"""Trains a DCGAN on the Voxel Art set and logs the evidence the project needs.

Every run writes to artifacts/runs/<name>/:
  config.json            exact arguments of the run
  losses.csv             per-iteration loss_D, loss_G, D(x), D(G(z))
  snapshots/epoch_XXXX.png  4x4 grid from the SAME fixed z for every run
  checkpoints/last.pt    resumable state (saved every --ckpt-every epochs)
  G_final.pt             generator weights at the end

Examples:
  python src/train.py --name base --loss bce
  python src/train.py --name hinge --loss hinge
  python src/train.py --name sn --loss bce --spectral
  python src/train.py --name sn_nobn --loss bce --spectral --no-d-bn
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from data import load_images
from losses import d_loss, g_loss
from models import Z_DIM, Discriminator, Generator, init_weights

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIXED_Z_PATH = ARTIFACTS / "fixed_z.pt"
FIXED_Z_SEED = 2026
N_FIXED = 16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--loss", choices=["bce", "minimax", "hinge"], default="bce")
    p.add_argument("--spectral", action="store_true", help="spectral norm on every conv of D")
    p.add_argument("--no-d-bn", action="store_true", help="remove BatchNorm from D")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=23242)
    p.add_argument("--snap-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fixed_noise(device: torch.device) -> torch.Tensor:
    """The evaluation noise is shared by every run so snapshots are comparable."""
    if not FIXED_Z_PATH.exists():
        FIXED_Z_PATH.parent.mkdir(parents=True, exist_ok=True)
        g = torch.Generator().manual_seed(FIXED_Z_SEED)
        torch.save(torch.randn(N_FIXED, Z_DIM, generator=g), FIXED_Z_PATH)
    return torch.load(FIXED_Z_PATH).to(device)


@torch.no_grad()
def snapshot(G: Generator, z: torch.Tensor, path: Path) -> None:
    # Eval mode uses BN running stats, so the same z always gives the same image.
    G.eval()
    save_image(G(z), path, nrow=4, normalize=True, value_range=(-1, 1))
    G.train()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = ARTIFACTS / "runs" / args.name
    (run_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    set_seed(args.seed)
    data = load_images().to(device)
    n = data.size(0)
    z_fixed = fixed_noise(device)

    G = Generator().to(device)
    D = Discriminator(spectral=args.spectral, batchnorm=not args.no_d_bn).to(device)
    G.apply(init_weights)
    D.apply(init_weights)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    start_epoch = 1
    ckpt_path = run_dir / "checkpoints" / "last.pt"
    log_mode = "w"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        torch.set_rng_state(ckpt["rng_cpu"])
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        start_epoch = ckpt["epoch"] + 1
        log_mode = "a"
        print(f"Resumed from epoch {ckpt['epoch']}")
    else:
        (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
        snapshot(G, z_fixed, run_dir / "snapshots" / "epoch_0000.png")

    log_file = open(run_dir / "losses.csv", log_mode, newline="")
    log = csv.writer(log_file)
    if log_mode == "w":
        log.writerow(["iter", "epoch", "loss_d", "loss_g", "d_real", "d_fake"])

    steps_per_epoch = n // args.batch
    it = (start_epoch - 1) * steps_per_epoch
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        perm = torch.randperm(n, device=device)
        for b in range(steps_per_epoch):
            real = data[perm[b * args.batch:(b + 1) * args.batch]]
            z = torch.randn(real.size(0), Z_DIM, device=device)

            # Discriminator step
            fake = G(z)
            real_logits = D(real)
            fake_logits = D(fake.detach())
            loss_d = d_loss(args.loss, real_logits, fake_logits)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # Generator step (reuses the same fake batch)
            fake_logits_g = D(fake)
            loss_g = g_loss(args.loss, fake_logits_g)
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            it += 1
            log.writerow([
                it, epoch, f"{loss_d.item():.5f}", f"{loss_g.item():.5f}",
                f"{torch.sigmoid(real_logits).mean().item():.4f}",
                f"{torch.sigmoid(fake_logits).mean().item():.4f}",
            ])

        if epoch % args.snap_every == 0 or epoch in (1, 2, 5) or epoch == args.epochs:
            snapshot(G, z_fixed, run_dir / "snapshots" / f"epoch_{epoch:04d}.png")
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            log_file.flush()
            torch.save({
                "epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
            }, ckpt_path)
        print(f"[{args.name}] epoch {epoch}/{args.epochs} "
              f"loss_d={loss_d.item():.3f} loss_g={loss_g.item():.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    log_file.close()
    torch.save(G.state_dict(), run_dir / "G_final.pt")
    print(f"Done: {run_dir}")


if __name__ == "__main__":
    main()

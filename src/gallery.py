"""Gallery generation and nearest-neighbor test.

Each candidate i is generated from its own seed (SEED_BASE + i), so any character can be
regenerated from the final weights and its seed alone.

  python src/gallery.py candidates --run sn            # 200 numbered candidates to choose from
  python src/gallery.py final --run sn --pick 3 17 ... # saves galeria/ and the NN figure
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from data import load_images
from features import FeatureExtractor
from models import Z_DIM, Generator

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "runs"
FIGS = ROOT / "artifacts" / "figures"
GALLERY = ROOT / "galeria"
SEED_BASE = 100_000
N_CANDIDATES = 200


def z_for_seed(seed: int) -> torch.Tensor:
    return torch.randn(1, Z_DIM, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def generate(run: str, seeds: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    G = Generator().to(device)
    G.load_state_dict(torch.load(RUNS / run / "G_final.pt", map_location=device))
    G.eval()
    z = torch.cat([z_for_seed(s) for s in seeds])
    return G(z.to(device)).cpu(), z


def to_img(x: torch.Tensor):
    return ((x.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()


def candidates(run: str, device: torch.device) -> None:
    seeds = [SEED_BASE + i for i in range(N_CANDIDATES)]
    imgs, _ = generate(run, seeds, device)
    cols = 20
    fig, axes = plt.subplots(N_CANDIDATES // cols, cols, figsize=(cols * 1.0, N_CANDIDATES // cols * 1.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(to_img(imgs[i]), interpolation="nearest")
        ax.set_title(str(i), fontsize=7, pad=1)
        ax.axis("off")
    fig.suptitle(f"{N_CANDIDATES} candidatos — corrida '{run}' (semilla = {SEED_BASE} + índice)")
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{run}_candidates.png", dpi=110)
    print(f"Saved {FIGS / f'{run}_candidates.png'}")


@torch.no_grad()
def final(run: str, picks: list[int], device: torch.device) -> None:
    seeds = [SEED_BASE + i for i in picks]
    imgs, z = generate(run, seeds, device)

    GALLERY.mkdir(exist_ok=True)
    meta = []
    for k, (img, seed, idx) in enumerate(zip(imgs, seeds, picks), start=1):
        name = f"personaje_{k:02d}"
        save_image(img, GALLERY / f"{name}.png", normalize=True, value_range=(-1, 1))
        # 4x nearest-neighbor upscale only for viewing; the 64x64 file is the model output.
        big = F.interpolate(img[None], scale_factor=4, mode="nearest")
        save_image(big, GALLERY / f"{name}_x4.png", normalize=True, value_range=(-1, 1))
        meta.append({"file": f"{name}.png", "candidate_index": idx, "seed": seed})
    torch.save(z, GALLERY / "z_vectors.pt")
    (GALLERY / "seeds.json").write_text(json.dumps(
        {"run": run, "weights": f"artifacts/runs/{run}/G_final.pt", "seed_base": SEED_BASE,
         "selection": f"{len(picks)} de {N_CANDIDATES}", "characters": meta}, indent=2))

    # Nearest neighbor in the training set, by Inception features (cosine) and by pixels (L2).
    real = load_images()
    fx = FeatureExtractor(device)
    f_real = F.normalize(fx(real), dim=1)
    f_gen = F.normalize(fx(imgs), dim=1)
    feat_dist = 1 - f_gen @ f_real.T                      # cosine distance
    pix_dist = torch.cdist(imgs.flatten(1), real.flatten(1)) / (64 * 64 * 3) ** 0.5  # RMS per pixel

    # Reference: how close real images are to their own nearest neighbor (excluding themselves).
    rr = 1 - f_real @ f_real.T
    rr.fill_diagonal_(float("inf"))
    real_nn = rr.min(1).values
    ref = {"p5": real_nn.quantile(0.05).item(), "median": real_nn.median().item()}

    rows = []
    fig, axes = plt.subplots(len(picks), 3, figsize=(6.6, 2.3 * len(picks)))
    for r, (k, img) in enumerate(zip(range(1, len(picks) + 1), imgs)):
        jf = int(feat_dist[r].argmin())
        jp = int(pix_dist[r].argmin())
        rows.append({"character": f"personaje_{k:02d}", "nn_feature_index": jf,
                     "feature_dist": round(feat_dist[r, jf].item(), 4),
                     "nn_pixel_index": jp, "pixel_rms": round(pix_dist[r, jp].item(), 4)})
        for ax, im, title in [
            (axes[r, 0], img, f"personaje_{k:02d}"),
            (axes[r, 1], real[jf], f"vecino CNN #{jf}\ndist={feat_dist[r, jf]:.3f}"),
            (axes[r, 2], real[jp], f"vecino píxel #{jp}\nRMS={pix_dist[r, jp]:.3f}"),
        ]:
            ax.imshow(to_img(im), interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.suptitle(f"Vecinos más cercanos en entrenamiento\nreferencia real↔real (CNN): "
                 f"p5={ref['p5']:.3f}, mediana={ref['median']:.3f}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGS / f"{run}_nearest_neighbors.png", dpi=120)
    (ROOT / "artifacts" / "nearest_neighbors.json").write_text(
        json.dumps({"run": run, "real_to_real_reference": ref, "rows": rows}, indent=2))
    print(json.dumps({"reference": ref, "rows": rows}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["candidates", "final"])
    p.add_argument("--run", required=True)
    p.add_argument("--pick", type=int, nargs="*", default=[])
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "candidates":
        candidates(args.run, device)
    else:
        if len(args.pick) < 10:
            raise SystemExit("Pick at least 10 candidates.")
        final(args.run, args.pick, device)


if __name__ == "__main__":
    main()

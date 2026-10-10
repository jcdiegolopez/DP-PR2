"""Compares runs with two numbers computed on the same samples for every run.

- FID (Inception-v3 from torchvision, 2,000 real vs 2,000 generated): lower = closer to the data.
  It is not comparable with published FIDs (different Inception weights and sample size),
  only between our own runs.
- Diversity ratio: mean pairwise feature distance among generated samples divided by the
  same quantity among real images. ~1 = as varied as the data; well below 1 = mode collapse.

Usage: python src/evaluate.py base hinge sn
Writes artifacts/metrics.csv (one row per run).
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

from data import load_images
from features import FeatureExtractor
from models import Z_DIM, Generator

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "runs"
N_SAMPLES = 2000
EVAL_SEED = 777  # same z for every run


def fid(a: np.ndarray, b: np.ndarray) -> float:
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = np.real(linalg.sqrtm(s1 @ s2))
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def mean_pairwise(f: torch.Tensor) -> float:
    d = torch.cdist(f, f)
    n = f.size(0)
    return float(d.sum() / (n * (n - 1)))


@torch.no_grad()
def sample(run: str, device: torch.device) -> torch.Tensor:
    G = Generator().to(device)
    G.load_state_dict(torch.load(RUNS / run / "G_final.pt", map_location=device))
    G.eval()
    g = torch.Generator(device="cpu").manual_seed(EVAL_SEED)
    z = torch.randn(N_SAMPLES, Z_DIM, generator=g).to(device)
    return G(z).cpu()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fx = FeatureExtractor(device)
    real = load_images()
    idx = torch.randperm(real.size(0), generator=torch.Generator().manual_seed(EVAL_SEED))[:N_SAMPLES]
    f_real = fx(real[idx])
    div_real = mean_pairwise(f_real)

    rows = []
    for run in sys.argv[1:]:
        f_fake = fx(sample(run, device))
        row = {
            "run": run,
            "fid": round(fid(f_real.numpy(), f_fake.numpy()), 2),
            "diversity_ratio": round(mean_pairwise(f_fake) / div_real, 3),
        }
        rows.append(row)
        print(row)

    out = ROOT / "artifacts" / "metrics.csv"
    existing = {}
    if out.exists():
        with open(out) as fh:
            existing = {r["run"]: r for r in csv.DictReader(fh)}
    existing.update({r["run"]: r for r in rows})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "fid", "diversity_ratio"])
        w.writeheader()
        w.writerows(existing.values())


if __name__ == "__main__":
    main()

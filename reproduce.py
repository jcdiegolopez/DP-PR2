"""Reproduces the project with one command.

  python reproduce.py          # data + gallery from the included weights, then checks it matches
  python reproduce.py --full   # also retrains the four runs (as <name>_repro) with figures and metrics

Retrained runs are saved as <name>_repro so they never overwrite the weights and evidence
used in the presentation. The Python environment (PyTorch with CUDA) must be installed first;
see README.md.
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
GALLERY = ROOT / "galeria"
PICKS = ["36", "100", "30", "13", "3", "62", "103", "102", "140", "151"]
RUNS = {
    "base": ["--loss", "bce"],
    "hinge": ["--loss", "hinge"],
    "sn": ["--loss", "bce", "--spectral"],
    "sn_nobn": ["--loss", "bce", "--spectral", "--no-d-bn"],
}


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def check_env() -> None:
    import torch

    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU found; generation works on CPU, training will be very slow.")
    if shutil.which("npm") is None and not dataset_ready():
        sys.exit("Node.js/npm is required to generate the dataset (https://nodejs.org).")


def dataset_ready() -> bool:
    return len(list((ROOT / "data" / "voxel64").glob("*.png"))) >= 5000


def gallery_hashes() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(GALLERY.glob("*.png"))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="also retrain the four runs")
    args = p.parse_args()
    check_env()

    # 1. Dataset
    if dataset_ready():
        print("Dataset already present: data/voxel64")
    else:
        npm = shutil.which("npm")
        assert npm is not None  # checked in check_env()
        run([npm, "install"], cwd=ROOT / "dataset")
        run(["node", "generate.mjs"], cwd=ROOT / "dataset")

    # 2. Optional retraining
    if args.full:
        names = [f"{n}_repro" for n in RUNS]
        for name, flags in RUNS.items():
            run([PY, "src/train.py", "--name", f"{name}_repro", *flags])
        run([PY, "src/plot_run.py", *names])
        run([PY, "src/evaluate.py", *names])
        run([PY, "src/gallery.py", "candidates", "--run", "hinge_repro"])

    # 3. Gallery from the included weights, checked against the committed images
    before = gallery_hashes()
    run([PY, "src/gallery.py", "final", "--run", "hinge", "--pick", *PICKS])
    after = gallery_hashes()
    if before and before != after:
        changed = [k for k in after if before.get(k) != after[k]]
        sys.exit(f"Gallery differs from the committed images: {changed}")
    print("\nOK: galeria/ regenerated" + (" and identical to the committed images." if before else "."))


if __name__ == "__main__":
    main()

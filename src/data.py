"""Loads the Voxel Art PNGs into a single tensor normalized to [-1, 1]."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "voxel64"


def load_images(data_dir: Path = DATA_DIR) -> torch.Tensor:
    """Returns a (N, 3, 64, 64) float tensor in [-1, 1].

    The full set is ~250 MB as float32, so it fits in GPU memory and avoids
    DataLoader worker overhead on Windows.
    """
    files = sorted(data_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNGs in {data_dir}. Run dataset/generate.mjs first.")
    # The background is already opaque, so dropping alpha loses nothing.
    arr = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return x

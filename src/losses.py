"""GAN losses computed on raw discriminator logits."""
import torch
import torch.nn.functional as F


def d_loss(kind: str, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    if kind in ("bce", "minimax"):
        # Same discriminator objective for both: classify real=1, fake=0.
        return (
            F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        )
    if kind == "hinge":
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    raise ValueError(f"Unknown loss: {kind}")


def g_loss(kind: str, fake_logits: torch.Tensor) -> torch.Tensor:
    if kind == "bce":
        # Non-saturating: maximize log D(G(z)).
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    if kind == "minimax":
        # Original saturating form: minimize log(1 - D(G(z))).
        return -F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    if kind == "hinge":
        return -fake_logits.mean()
    raise ValueError(f"Unknown loss: {kind}")

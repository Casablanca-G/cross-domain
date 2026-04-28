"""Task Complexity Modulation (TCM).

IBML Section 3.3.2: to equalize H(T|X^m) across domains, the strong domain's
input is perturbed so that its effective task difficulty rises. In CDSR this
translates to randomly masking items in the dominant domain's behaviour
sequence with probability proportional to its lead over the domain-mean rho.
"""
import torch


def apply_tcm_noise(seq: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Randomly zero out a fraction of non-padding items in `seq`.

    Args:
        seq: long tensor of item ids, shape (B, L). 0 is padding.
        mask_ratio: probability in [0, 1) of replacing a non-padding item with 0.
    Returns a new tensor (does not modify input).
    """
    if mask_ratio <= 0.0:
        return seq
    mask_ratio = min(float(mask_ratio), 0.9)
    drop = (seq > 0) & (torch.rand_like(seq, dtype=torch.float32) < mask_ratio)
    return seq.masked_fill(drop, 0)

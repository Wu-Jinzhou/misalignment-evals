"""Lightweight local package for the copied misalignment eval assets."""

from misalignment_evals.mgs import MGSResult, compute_mgs

__all__ = [
    "compute_mgs",
    "MGSResult",
]

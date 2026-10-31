"""Likelihood-ratio CUSUM extracted from the team Killswitch project."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Cusum:
    """S_t = max(0, S_{t-1} + llr_t); alarm strictly above threshold."""

    def init(self, shape):
        return np.zeros(shape)

    def step(self, s, r, llr):
        # r is unused; retain the upstream simulator's common detector interface.
        s = np.maximum(0.0, s + llr)
        return s, s

    def fires(self, stat, threshold):
        return stat > threshold


def cusum(llr):
    """Run without alarm resets; final axis is time, leading axes are series."""
    llr = np.asarray(llr, dtype=float)
    state = Cusum().init(llr.shape[:-1])
    out = np.empty_like(llr)
    for t in range(llr.shape[-1]):
        state, out[..., t] = Cusum().step(state, None, llr[..., t])
    return out

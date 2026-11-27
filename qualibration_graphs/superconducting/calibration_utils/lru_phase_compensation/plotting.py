"""Plot of the f-LRU phase compensation (node 26c)."""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from .analysis import wrap_turns


def plot_raw_data_with_fit(ds_fit: xr.Dataset, qubits: List) -> Figure:
    """P(e) against the swept virtual Z, with the fit and the frame that the new correction moves to zero."""
    n = len(qubits)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.8), squeeze=False)
    for ax, qubit in zip(axes[0], qubits):
        fit = ds_fit.sel(qubit=qubit.name)
        frames = fit.frame.values
        ax.plot(frames, fit.state.values, "o", ms=4, label="data")
        ax.plot(frames, fit.fitted_state.values, "-", lw=2, label="fit")
        # The peak sits where the swept frame cancels the residual phase
        ax.axvline(float(frames[np.argmax(fit.fitted_state.values)]), color="0.4", ls=":", lw=1)
        ax.set_xlabel("Virtual Z after the LRU (turns)")
        ax.set_ylabel("P(e)")
        ax.set_title(
            f"{qubit.name}: residual {float(fit.residual_phase) * 360:+.1f}° ± {float(fit.residual_phase_stderr) * 360:.1f}°\n"
            f"correction → {float(wrap_turns(fit.phase_shift)) * 360:+.1f}°"
        )
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("f-LRU phase compensation: x90 – LRU – Z(θ) – x90")
    fig.tight_layout()
    return fig

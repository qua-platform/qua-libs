"""Plot fitted ring-down phase and Ramsey contrast recovery."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_results(ds: xr.Dataset, fit_results):
    """Return the ring-down phase and contrast figure."""
    figure, axes = plt.subplots(ds.sizes["qubit"], 2, figsize=(12, 4 * ds.sizes["qubit"]), squeeze=False)
    for row, qubit_name in enumerate(ds.qubit.values):
        selected = ds.sel(qubit=qubit_name)
        result = fit_results[str(qubit_name)]
        axes[row, 0].plot(
            selected.ringdown_delay, np.rad2deg(selected.ramsey_phase_rad), "o", ms=4, label="Ramsey phase"
        )
        axes[row, 0].plot(
            selected.ringdown_delay, np.rad2deg(selected.ramsey_phase_fit_rad), "-", label="exponential fit"
        )
        axes[row, 0].set(
            title=f"{qubit_name}: tau={result['tau_ringdown_ns']:.1f} +/- {result['tau_ringdown_uncertainty_ns']:.1f} ns",
            xlabel="delay after readout (ns)",
            ylabel="Ramsey phase (deg)",
        )
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].plot(selected.ringdown_delay, selected.ramsey_contrast, "o-")
        axes[row, 1].set(
            title=f"{qubit_name} contrast recovery",
            xlabel="delay after readout (ns)",
            ylabel="Ramsey contrast",
            ylim=(0, 1.05),
        )
        for axis in axes[row]:
            axis.grid(alpha=0.25)
    figure.suptitle("Resonator ring-down measured with Ramsey")
    figure.tight_layout()
    return {"resonator_ringdown_ramsey": figure}

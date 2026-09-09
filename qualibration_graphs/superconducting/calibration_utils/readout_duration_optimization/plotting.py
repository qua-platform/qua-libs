"""Duration curves and selected IQ blobs/confusion matrices on the QUAM qubit grid."""

import numpy as np
from qualibration_libs.plotting import QubitGrid, grid_iter


def plot_results(ds, qubits, fits, blobs):
    figures = {}
    for kind in ("duration", "iq_blobs", "confusion_matrix"):
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, selection in grid_iter(grid):
            fit, blob = fits.sel(**selection), blobs.sel(**selection)
            ax.set_title(selection["qubit"])
            if kind == "duration":
                for metric, label in (("meas_fidelity", "Assignment fidelity"), ("non_outliers", "Non-outlier fraction")):
                    ax.plot(fit.duration, fit.fit_data.sel(fit_vals=metric), ".-", label=label)
                if np.isfinite(fit.optimal_duration):
                    ax.axvline(float(fit.optimal_duration), color="k", linestyle="--", label="Optimal duration")
                ax.set(xlabel="Readout duration [ns]", ylabel="Probability", ylim=(0, 1.03))
                ax.legend()
            elif not np.isfinite(fit.optimal_duration):
                ax.text(0.5, 0.5, "No valid duration", ha="center", transform=ax.transAxes)
            elif kind == "iq_blobs":
                for state, label in enumerate("gef"[:ds.sizes["state"]]):
                    ax.plot(1e3 * blob.I.sel(state=state), 1e3 * blob.Q.sel(state=state), ".",
                            alpha=0.2, markersize=2, label=label)
                ax.plot(1e3 * blob.centers.sel(quadrature="I"), 1e3 * blob.centers.sel(quadrature="Q"), "kx")
                if ds.sizes["state"] == 2:
                    ax.axvline(1e3 * float(blob.ge_threshold), color="r", linestyle="--", label="GE threshold")
                    ax.axvline(1e3 * float(blob.rus_threshold), color="k", linestyle=":", label="RUS threshold")
                ax.set(xlabel="I [mV]", ylabel="Q [mV]", aspect="equal")
                ax.legend()
            else:
                matrix = blob.confusion_matrix.values
                ax.imshow(matrix, vmin=0, vmax=1)
                for row, column in np.ndindex(matrix.shape):
                    ax.text(column, row, f"{100 * matrix[row, column]:.1f}%", ha="center", va="center")
                labels = list("gef"[:ds.sizes["state"]])
                ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
                       yticklabels=labels, xlabel="Measured", ylabel="Prepared")
        grid.fig.suptitle(kind.replace("_", " ").title())
        grid.fig.tight_layout()
        figures[kind] = grid.fig
    return figures

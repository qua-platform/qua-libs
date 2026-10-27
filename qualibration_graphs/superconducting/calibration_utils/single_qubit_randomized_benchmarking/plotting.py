from typing import List
import xarray as xr
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from calibration_utils.common_utils.rb import LeakageFit, rb_decay_curve
from calibration_utils.single_qubit_randomized_benchmarking.analysis import format_leakage_rate

u = unit(coerce_to_integer=True)

# Shared by the survival and leakage figures so both scale the same way with the number of qubits.
PANEL_SIZE = 5  # inches per qubit panel
TITLE_FONTSIZE = 11
LABEL_FONTSIZE = 10
LEGEND_FONTSIZE = 9


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset,
    *,
    log_x: bool = False,
    interleaved_gate: str | None = None,
):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.
    log_x : bool, optional
        Use a logarithmic circuit-depth axis (for log-spaced depths).
    interleaved_gate : str, optional
        The gate interleaved after every Clifford (11b): the fitted error is then per Clifford + gate step.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits], size=PANEL_SIZE)
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(
            ax, ds, qubit, fits.sel(qubit=qubit["qubit"]), log_x=log_x, interleaved_gate=interleaved_gate
        )

    interleaved = f"\n(interleaved with {interleaved_gate})" if interleaved_gate is not None else ""
    grid.fig.suptitle(f"Single qubit randomized benchmarking{interleaved}", fontsize=TITLE_FONTSIZE + 1)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    fit: xr.Dataset = None,
    *,
    log_x: bool = False,
    interleaved_gate: str | None = None,
):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """
    # Fitted decay, drawn on a dense depth grid so the curve is smooth between the measured depths
    depths = fit.depths.values
    dense = (np.geomspace if log_x else np.linspace)(depths.min(), depths.max(), 200)
    alpha = float(fit.fit_alpha.values)
    fitted = rb_decay_curve(dense, float(fit.fit_amplitude.values), alpha, float(fit.fit_offset.values))
    if hasattr(fit, "pg") or hasattr(fit, "state"):  # g/e/f or g/e state discrimination
        label = r"Probability to recover to $|0\rangle$"
    elif hasattr(fit, "I"):
        label = "I quadrature [mV]"
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")
    ax.errorbar(
        depths,
        fit.averaged_data,
        yerr=np.nan_to_num(fit.survival_stderr.values),
        fmt="o",
        ms=4,
        capsize=2,
        elinewidth=0.8,
        label="Mean over sequences",
    )
    fit_label = "Exponential fit" if interleaved_gate is None else "Interleaved fit"
    ax.plot(dense, fitted, "r--", label=f"{fit_label} (α = {alpha:.5f})")
    # 11b: the same A and B with the 11a reference alpha. The gate error comes only from the ratio of
    # the two alphas, so the data should fall below this curve by the gate's own error.
    if bool(fit.success.values) and "reference_alpha" in fit and np.isfinite(float(fit.reference_alpha.values)):
        alpha_ref = float(fit.reference_alpha.values)
        reference = rb_decay_curve(dense, float(fit.fit_amplitude.values), alpha_ref, float(fit.fit_offset.values))
        ax.plot(dense, reference, ":", color="0.4", label=f"11a reference decay (α = {alpha_ref:.5f})")
    ax.grid(True, which="both", alpha=0.4)
    epc = float(fit.error_per_clifford.values)
    epc_stderr = float(fit.error_per_clifford_stderr.values) if "error_per_clifford_stderr" in fit else None
    if not bool(fit.success.values):
        stats = "Fit failed"
    elif interleaved_gate is None:
        stats = (
            f"1Q Clifford Fidelity ($f_c$) = {_format_fraction(1 - epc, epc_stderr)}\n"
            f"Error Per Clifford (EPC) = $1 - f_c$ = {_format_fraction(epc, epc_stderr, as_error_rate=True)}"
        )
    elif "gate_error" in fit and np.isfinite(float(fit.gate_error.values)):
        r = float(fit.gate_error.values)
        r_stderr = float(fit.gate_error_stderr.values)
        r_stderr = r_stderr if np.isfinite(r_stderr) else None
        stats = (
            f"{interleaved_gate} fidelity = {_format_fraction(1 - r, r_stderr)}\n"
            f"{interleaved_gate} error = {_format_fraction(r, r_stderr, as_error_rate=True)}"
        )
    else:
        stats = (
            f"Clifford + {interleaved_gate} step fidelity = {_format_fraction(1 - epc, epc_stderr)}\n"
            f"(no 11a alpha stored: {interleaved_gate} alone not available)"
        )
    ax.set_title(f"Qubit: {qubit['qubit']}\n{stats}", fontsize=LABEL_FONTSIZE, linespacing=1.3)
    ax.set_xlabel("Circuit depth (Cliffords)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(label, fontsize=LABEL_FONTSIZE)
    if log_x:
        ax.set_xscale("log")
    ax.legend(fontsize=LEGEND_FONTSIZE)


def plot_leakage(
    ds_leakage: xr.Dataset,
    qubits: List[AnyTransmon],
    leakage_results: dict,
    *,
    log_x: bool = False,
    reset_type: str = "",
    interleaved_gate: str | None = None,
) -> Figure:
    """Plot the leaked population P(f) against circuit depth with its fit, one panel per qubit.

    ``leakage_results`` is ``node.results["leakage_results"]``: the ``asdict`` of each qubit's
    ``LeakageResult``. With ``interleaved_gate`` (11b) each panel title also gives the gate's own L1.
    """
    step = "Clifford" if interleaved_gate is None else f"Clifford + {interleaved_gate}"
    grid = QubitGrid(ds_leakage, [q.grid_location for q in qubits], size=PANEL_SIZE)
    for ax, qubit in grid_iter(grid):
        name = qubit["qubit"]
        result = leakage_results[name]
        fit = LeakageFit(**result["per_clifford"])
        fr = ds_leakage.sel(qubit=name)
        depths = fr.depths.values
        line = ax.errorbar(
            depths,
            fr.leaked.values,
            yerr=np.nan_to_num(fr.leaked_stderr.values),
            fmt="o",
            ms=4,
            capsize=2,
            label="Mean over sequences",
        )
        if fit.success and not result["failure"]:
            dense = (np.geomspace if log_x else np.linspace)(depths.min(), depths.max(), 200)
            ax.plot(dense, fit.curve(dense), "-", color=line[0].get_color(), lw=1, label=f"Leakage fit ({fit.method})")
        lines = [f"Qubit: {name}"]
        if result["failure"]:
            # The fitted rate means nothing when the readout is off.
            lines.append("FAILED: g/e/f readout suspect (not saved)")
        elif interleaved_gate is not None and result["per_gate"]:
            # The interleaved run is about the gate; its per-step rate is in the log.
            per_gate = result["per_gate"]
            rate = format_leakage_rate(per_gate["l1"], per_gate["l1_stderr"], pm="±")
            lines.append(f"L1 per {interleaved_gate} = {rate}")
        else:
            lines.append(f"L1 per {step} = {format_leakage_rate(fit.l1, fit.l1_stderr, pm='±')}")
        ax.grid(True, which="both", alpha=0.4)
        ax.set_title("\n".join(lines), fontsize=LABEL_FONTSIZE, linespacing=1.3)
        ax.set_xlabel("Circuit depth (Cliffords)", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Leaked population", fontsize=LABEL_FONTSIZE)
        if log_x:
            ax.set_xscale("log")
        ax.legend(fontsize=LEGEND_FONTSIZE)

    reset = f"{reset_type} reset, " if reset_type else ""
    interleaved = f"interleaved with {interleaved_gate}, " if interleaved_gate is not None else ""
    grid.fig.suptitle(
        f"Single qubit leakage randomized benchmarking\n({interleaved}{reset}g/e/f readout)",
        fontsize=TITLE_FONTSIZE + 1,
    )
    grid.fig.tight_layout()
    return grid.fig


def _format_fraction(value: float, stderr: float | None = None, *, as_error_rate: bool = False) -> str:
    """A fidelity as a percentage, or a small error rate in units of 10⁻³, with ±1σ when given (as the 2Q RB plots)."""
    if value is None or not np.isfinite(value):
        return "n/a"
    has_err = stderr is not None and np.isfinite(stderr) and stderr > 0
    if as_error_rate and value < 0.01:
        return f"{value * 1e3:.2f} ± {stderr * 1e3:.2f} × 10⁻³" if has_err else f"{value * 1e3:.2f} × 10⁻³"
    return f"{100 * value:.3f} ± {100 * stderr:.3f}%" if has_err else f"{100 * value:.3f}%"

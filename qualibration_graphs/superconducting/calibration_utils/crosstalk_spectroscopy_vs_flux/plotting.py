from typing import Dict, Any
import xarray as xr
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from mpl_toolkits.axes_grid1 import make_axes_locatable

from qualang_tools.units import unit

from qualibrate import QualibrationNode

from calibration_utils.crosstalk_spectroscopy_vs_flux.analysis import build_crosstalk_matrix
from calibration_utils.crosstalk_spectroscopy_vs_flux.program import get_expected_frequency_at_flux_detuning

u = unit(coerce_to_integer=True)

_LEGEND_FONT_SIZE = 8


def _select_pair_panel(ds: xr.Dataset, target_qubit_name: str, pair_idx: int) -> xr.Dataset:
    """Select one pair panel by acquisition index and attach physical sweep coordinates."""
    da = ds.sel(qubit=target_qubit_name, pair=pair_idx)
    return da.assign_coords(
        flux_bias=("flux_bias", da.flux_bias_sweep.values),
        detuning=("detuning", da.detuning_sweep.values),
        freq_GHz=(da.full_freq / 1e9),
    )


def _detuning_to_freq_ghz(detuning_hz: xr.DataArray, panel: xr.Dataset) -> xr.DataArray:
    """Map Lorentzian peak detuning (Hz) to absolute frequency (GHz) for overlay on the heatmap."""
    freq_offset_hz = panel.full_freq.isel(detuning=0) - panel.detuning.isel(detuning=0)
    return (detuning_hz + freq_offset_hz) / 1e9


def plot_analysis(
    ds: xr.Dataset,
    peak_results: Dict[str, Any],
    fit_results: Dict[str, Any],
    flux_detunings: Dict[str, Any],
    qubits: Dict[str, Any],
):
    """
    Plot the full analysis pipeline onto a 2D heatmap for each pair of qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    peak_results : Dict
        A dictionary of Lorentzian peak fitting results at each detuning for each qubit pair.
    fit_results : Dict
        A dictionary of linear fit results for each qubit pair.
    qubits : list of AnyTransmon
        A list of qubits to plot.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    target_qubits = ds.qubit.data
    n_pairs = ds.sizes["pair"]

    fig, axes = plt.subplots(
        len(target_qubits),
        n_pairs,
        figsize=(4 * n_pairs, 5 * len(target_qubits)),
        squeeze=False,
    )

    def get_ax(axes, i=0, j=0):
        if isinstance(axes, np.ndarray):
            if axes.ndim == 1:
                return axes[i]
            if axes.ndim == 2:
                return axes[i, j]
        return axes

    for i, target_qubit in enumerate(target_qubits):
        for j in range(n_pairs):
            aggressor_qubit = ds.aggressor.sel(qubit=target_qubit, pair=j).item()
            fit_result = fit_results[target_qubit][aggressor_qubit]
            peak_result = peak_results.sel(qubit=target_qubit, aggressor=aggressor_qubit)
            panel = _select_pair_panel(ds, target_qubit, j)

            ax = get_ax(axes, i, j)
            plot_individual_raw_data(ax, panel, target_qubit)
            plot_individual_peak_frequencies(ax, fit_result, peak_result, panel)
            plot_individual_linear_fit(ax, fit_result, peak_result, panel)

            if flux_detunings is not None:
                delta_phi = flux_detunings[target_qubit]
                delta_f = (
                    get_expected_frequency_at_flux_detuning(qubits[target_qubit], flux_detunings[target_qubit])
                    - qubits[target_qubit].xy.RF_frequency
                )
            else:
                delta_phi = np.nan
                delta_f = np.nan

            if fit_result.get("pair_type") == "self":
                role_label = f"Self response of {target_qubit}"
            else:
                role_label = f"{aggressor_qubit} acting on {target_qubit}"
                crosstalk_pct = 100 * fit_result.get("crosstalk_coefficient", np.nan)
                if np.isfinite(crosstalk_pct):
                    crosstalk_err_pct = 100 * fit_result.get("crosstalk_coefficient_uncertainty", np.nan)
                    if np.isfinite(crosstalk_err_pct):
                        role_label += f"\nCrosstalk: {crosstalk_pct:.2f}% ± {crosstalk_err_pct:.2f}%"
                    else:
                        role_label += f"\nCrosstalk: {crosstalk_pct:.2f}%"

            ax.set_title(
                f"{role_label}\n"
                f"$\Delta\phi_{{{target_qubit}}} = {1000 * delta_phi:.1f}\,$mV, "
                f"$\Delta f_{{{target_qubit}}} = {delta_f / 1e6:.2f}\,$MHz"
            )
            ax.set_xlabel(f"{aggressor_qubit} Flux Bias (V)")

    fig.suptitle("Crosstalk Spectroscopy")
    fig.tight_layout()
    return fig


def plot_individual_raw_data(ax: Axes, panel: xr.Dataset, target_qubit_name: str):
    """Plot individual qubit pair raw IQ_abs data on a given axis."""
    im = panel.IQ_abs.plot(
        ax=ax,
        x="flux_bias",
        y="freq_GHz",
        add_colorbar=False,
        robust=True,
    )

    ax.set_ylabel("Frequency (GHz)")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.get_figure().colorbar(im, cax=cax)
    cbar.set_label(
        rf"{target_qubit_name} Readout Amplitude = $\sqrt{{I^2 + Q^2}}$",
        fontsize=10,
        rotation=270,
        labelpad=15,
    )

    return im


def plot_individual_peak_frequencies(
    ax: Axes,
    fit_result: dict,
    peak_result: xr.Dataset,
    panel: xr.Dataset,
):
    """
    Plot individual qubit pair peak frequencies with error bars.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    fit_result : dict
        Dictionary containing linear fit metadata for the qubit pair.
    peak_result : xr.Dataset
        Peak frequency results for the qubit pair.
    panel : xr.Dataset
        Dataset slice for this qubit pair (for detuning -> GHz conversion).
    """
    mask = xr.DataArray(fit_result["linear_fit_inlier_mask"], dims="flux_bias")
    peak_result_inliers = peak_result.dropna("flux_bias").where(mask, drop=True)
    peak_result_outliers = peak_result.dropna("flux_bias").where(~mask, drop=True)

    peak_inliers_ghz = _detuning_to_freq_ghz(peak_result_inliers.peak_frequencies, panel)
    peak_outliers_ghz = _detuning_to_freq_ghz(peak_result_outliers.peak_frequencies, panel)
    freq_span_ghz = float(panel.freq_GHz.max() - panel.freq_GHz.min())
    max_err_ghz = 0.05 * freq_span_ghz
    peak_err_ghz = (peak_result_inliers.peak_frequency_errors / 1e9).clip(max=max_err_ghz)
    peak_err_outliers_ghz = (peak_result_outliers.peak_frequency_errors / 1e9).clip(max=max_err_ghz)

    ax.errorbar(
        peak_result_inliers.flux_bias,
        peak_inliers_ghz,
        yerr=peak_err_ghz,
        fmt="s",
        capsize=2,
        capthick=1,
        markersize=3,
        color="r",
        markerfacecolor="r",
        markeredgecolor="r",
        markeredgewidth=1,
        label="Peak Frequencies",
    )
    ax.errorbar(
        peak_result_outliers.flux_bias,
        peak_outliers_ghz,
        yerr=peak_err_outliers_ghz,
        fmt="s",
        capsize=2,
        capthick=1,
        markersize=3,
        color="C1",
        markerfacecolor="C1",
        markeredgecolor="C1",
        markeredgewidth=1,
        label="Outliers",
    )

    ax.legend(loc="upper right", fontsize=_LEGEND_FONT_SIZE)


def plot_individual_linear_fit(ax: Axes, fit_result: dict, peak_result: xr.Dataset, panel: xr.Dataset):
    """
    Plot individual qubit pair linear fit overlaid on peak frequency data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    fit_result : dict
        Fit results for the qubit pair.
    peak_result : xr.Dataset
        Peak frequency results for the qubit pair.
    panel : xr.Dataset
        Dataset slice for this qubit pair (for detuning -> GHz conversion).
    """
    flux_bias = peak_result.flux_bias.values

    if fit_result["success"] and len(flux_bias) > 0:
        flux_smooth = np.linspace(np.nanmin(flux_bias), np.nanmax(flux_bias), 100)
        freq_smooth_hz = fit_result["linear_fit_slope"] * flux_smooth + fit_result["linear_fit_intercept"]
        freq_smooth_ghz = _detuning_to_freq_ghz(xr.DataArray(freq_smooth_hz), panel).values
        slope_mhz_v = 1e-6 * fit_result["linear_fit_slope"]
        slope_err_mhz_v = 1e-6 * fit_result.get("linear_fit_slope_err", np.nan)
        if np.isfinite(slope_err_mhz_v):
            slope_label = f"Slope: {slope_mhz_v:.2f} ± {slope_err_mhz_v:.2f} MHz/V"
        else:
            slope_label = f"Slope: {slope_mhz_v:.2f} MHz/V"
        ax.plot(
            flux_smooth,
            freq_smooth_ghz,
            "r-",
            linewidth=3,
            label=f"Linear Fit\n{slope_label}",
        )

    ax.legend(loc="upper right", fontsize=_LEGEND_FONT_SIZE)


def plot_crosstalk_matrix(
    fit_results: Dict[str, Any],
    aggressor_qubits=None,
) -> Figure:
    """Plot fitted crosstalk coefficients as a labeled heatmap matrix."""
    target_names, aggressor_names, cells, values_pct, _ = build_crosstalk_matrix(
        fit_results, aggressor_qubits
    )

    n_targets = len(target_names)
    n_aggressors = len(aggressor_names)
    fig, ax = plt.subplots(figsize=(max(4, 1.5 * n_aggressors), max(3, n_targets + 1)))

    if n_targets == 0 or n_aggressors == 0:
        ax.text(0.5, 0.5, "No cross-talk pairs", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.suptitle("Crosstalk coefficient matrix (%)")
        return fig

    finite_values = values_pct[np.isfinite(values_pct)]
    if finite_values.size:
        color_limit = max(abs(finite_values.min()), abs(finite_values.max()), 1.0)
        norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
        display = np.ma.masked_invalid(values_pct)
        im = ax.imshow(display, cmap="RdBu_r", norm=norm, aspect="auto")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(im, cax=cax, label="Crosstalk (%)")
    else:
        color_limit = 1.0
        ax.imshow(np.zeros((n_targets, n_aggressors)), cmap="Greys", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(n_aggressors))
    ax.set_yticks(np.arange(n_targets))
    ax.set_xticklabels(aggressor_names)
    ax.set_yticklabels(target_names)
    ax.set_xlabel("Aggressor")
    ax.set_ylabel("Target")
    ax.set_title("Crosstalk coefficient matrix (%)")

    for i in range(n_targets):
        for j in range(n_aggressors):
            cell_text = cells[i][j]
            value = values_pct[i, j]
            if cell_text == "-":
                text_color = "0.45"
            elif cell_text == "FAIL":
                text_color = "0.15"
            elif np.isfinite(value) and abs(value) > 0.35 * color_limit:
                text_color = "white"
            else:
                text_color = "black"
            ax.text(j, i, cell_text, ha="center", va="center", color=text_color, fontsize=9)

    fig.tight_layout()
    return fig


def add_node_info_subtitle(node: QualibrationNode, fig: Figure = None, additional_info=None):
    """
    Add a standardized subtitle with node information to a matplotlib figure.
    If a suptitle already exists, the node info will be appended to it.

    Args:
        fig: matplotlib figure object. If None, uses plt.gcf()
        additional_info: Optional string with additional information to include

    Returns:
        str: The subtitle text that was added
    """
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.gcf()

    subtitle_parts = [f"#{node.storage_manager.snapshot_idx}"]

    if additional_info:
        subtitle_parts.append(additional_info)

    node_info_text = "\n".join(subtitle_parts)

    existing_suptitle = fig._suptitle
    if existing_suptitle is not None and existing_suptitle.get_text().strip():
        combined_text = f"{existing_suptitle.get_text()}\n{node_info_text}"
    else:
        combined_text = node_info_text

    fig.suptitle(combined_text, fontsize=10, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    return node_info_text

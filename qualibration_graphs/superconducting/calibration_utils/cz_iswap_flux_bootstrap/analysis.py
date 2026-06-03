"""Analysis for the CZ / iSWAP flux bootstrap (2D flux landscape).

Pipeline (per qubit pair)
-------------------------
1. **2D contrast** — ``target − control`` (CZ) or ``control − target`` (iSWAP).
2. **Qubit flux** — average contrast over coupler; CZ → argmax, iSWAP → argmin.
3. **1D cut** — contrast vs coupler flux at that qubit flux.
4. **Coarse (heavy Savitzky–Golay)** — sliding-window FFT → flat vs oscillation masks;
   decouple = min |contrast| in flat; gate coupler = first dip toward fringes from decouple.
5. **Refine (light smooth)** — local decouple and gate indices near coarse picks.
6. **Success** — decouple and gate found, distinct, not on sweep boundaries; qubit cut not on edge.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import oscillation

from .parameters import moving_qubit
from scipy.signal import find_peaks, savgol_filter

# --- Contrast-cut / region detection ---
_FFT_WINDOW = 30
_FRINGE_FREQ_LOW = 0.05
_FRINGE_FREQ_HIGH = 0.45
_OSC_POWER_THRESH = 0.05
_FLAT_POWER_THRESH = 0.03
_SAVGOL_WINDOW_COARSE = 21
_SAVGOL_WINDOW_FINE = 7
_SAVGOL_POLY = 3
_GUARD_FRACTION = 0.05
_GATE_DIP_PROMINENCE = 0.05
_MIN_FLAT_POINTS = 4
_REFINE_HALF_WIDTH_DECOUPLE = 20
_REFINE_HALF_WIDTH_GATE = 15


@dataclass
class FitParameters:
    """Flux operating points from the 2D landscape fit (one qubit pair)."""

    success: bool
    optimal_qubit_flux: float
    optimal_decouple_offset: float
    optimal_decouple_coupler_flux_rel: float = np.nan
    optimal_cz_coupler_flux: float = np.nan  # relative to decouple_offset
    optimal_cz_coupler_flux_total: float = np.nan  # absolute coupler flux
    # Optional 1D diagnostics (debug plot)
    contrast_coupler_rel: Optional[np.ndarray] = None
    contrast_coupler_full: Optional[np.ndarray] = None
    contrast_raw: Optional[np.ndarray] = None
    contrast_smoothed: Optional[np.ndarray] = None
    ac_power_norm: Optional[np.ndarray] = None
    osc_mask: Optional[np.ndarray] = None
    flat_mask: Optional[np.ndarray] = None


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log qubit flux, decouple offset, and gate coupler flux per pair."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, fit_result in fit_results.items():
        success = fit_result.get("success", False)
        qubit_v = fit_result.get("optimal_qubit_flux", np.nan)
        decouple_v = fit_result.get("optimal_decouple_offset", np.nan)
        gate_total = fit_result.get("optimal_cz_coupler_flux_total", np.nan)
        gate_rel = fit_result.get("optimal_cz_coupler_flux", np.nan)

        lines = [
            f"Results for qubit pair {qp_name}: {'SUCCESS' if success else 'FAIL'}",
            f"\tOptimal qubit flux: {qubit_v:.6f} V"
            if np.isfinite(qubit_v)
            else "\tOptimal qubit flux: N/A",
            f"\tDecouple offset (min |contrast| in flat): {decouple_v:.6f} V"
            if np.isfinite(decouple_v)
            else "\tDecouple offset: N/A",
            f"\tGate coupler flux (1st dip after decouple): {gate_total:.6f} V"
            if np.isfinite(gate_total)
            else "\tGate coupler flux: N/A",
            f"\tGate coupler pulse amplitude (rel. to decouple): {gate_rel:.6f} V"
            if np.isfinite(gate_rel)
            else "\tGate coupler pulse (rel.): N/A",
        ]
        log_callable("\n".join(lines))


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Add absolute flux coordinates and detuning for plotting."""
    detuning_mode = "quadratic"
    qubit_pairs = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]
    fluxes_qp = node.namespace["fluxes_qp"]
    fluxes_coupler = ds.coupler_flux.values
    qubit_flux_full = np.array([fluxes_qp[qp.name] for qp in qubit_pairs])
    ds = ds.assign_coords({"qubit_flux_full": (["qubit_pair", "qubit_flux"], qubit_flux_full)})

    coupler_flux_full = np.array([fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs])
    if detuning_mode == "quadratic":
        detuning = np.array(
            [-fluxes_qp[qp.name] ** 2 * moving_qubit(qp).freq_vs_flux_01_quad_term for qp in qubit_pairs]
        )
    elif detuning_mode == "cosine":
        detuning = np.array(
            [
                oscillation(
                    fluxes_qp,
                    moving_qubit(qp).extras["a"],
                    moving_qubit(qp).extras["f"],
                    moving_qubit(qp).extras["phi"],
                    moving_qubit(qp).extras["offset"],
                )
                for qp in qubit_pairs
            ]
        )
    else:
        raise ValueError(f"Invalid detuning_mode: {detuning_mode}. Must be 'quadratic' or 'cosine'")
    ds = ds.assign_coords({"coupler_flux_full": (["qubit_pair", "coupler_flux"], coupler_flux_full)})
    ds = ds.assign_coords({"detuning": (["qubit_pair", "qubit_flux"], detuning)})

    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """Fit decouple offset, qubit flux, and gate coupler flux for each qubit pair."""
    return _extract_fit_parameters(ds, node)


# ---------------------------------------------------------------------------
# Contrast map and qubit-flux cut
# ---------------------------------------------------------------------------


def _interaction_map(
    control: xr.DataArray, target: xr.DataArray, cz_or_iswap: str
) -> xr.DataArray:
    """2D interaction contrast (sign depends on gate)."""
    if cz_or_iswap == "cz":
        return target - control
    if cz_or_iswap == "iswap":
        return control - target
    raise ValueError(f"cz_or_iswap must be 'cz' or 'iswap', got {cz_or_iswap!r}")


def _qubit_flux_cut_index(contrast: xr.DataArray, cz_or_iswap: str) -> int:
    """Index along qubit_flux where the coupler-averaged contrast is optimal for the gate."""
    marginal = contrast.mean(dim="coupler_flux")
    if cz_or_iswap == "cz":
        return int(marginal.argmax(dim="qubit_flux").values)
    return int(marginal.argmin(dim="qubit_flux").values)


# ---------------------------------------------------------------------------
# 1D coupler cut: smoothing, FFT regions, dips
# ---------------------------------------------------------------------------


def _sliding_fringe_power(y: np.ndarray) -> np.ndarray:
    """Normalised sliding-window FFT power in the fringe band (0–1 after max scaling)."""
    half_w = _FFT_WINDOW // 2
    n = y.size
    freqs = np.fft.rfftfreq(_FFT_WINDOW)
    band = (freqs >= _FRINGE_FREQ_LOW) & (freqs <= _FRINGE_FREQ_HIGH)

    ac_power = np.zeros(n)
    for i in range(n):
        i0 = max(0, i - half_w)
        i1 = min(n, i + half_w)
        seg = y[i0:i1]
        seg_pad = np.pad(seg, (0, _FFT_WINDOW - len(seg)), mode="edge")
        spectrum = np.abs(np.fft.rfft(seg_pad - seg_pad.mean())) ** 2
        ac_power[i] = spectrum[band].mean()

    return ac_power / (ac_power.max() + 1e-12)


def _savgol_smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Savitzky–Golay smooth with odd window capped to trace length."""
    sw = min(window | 1, y.size)
    if sw % 2 == 0:
        sw -= 1
    sw = max(sw, _SAVGOL_POLY + 2)
    return savgol_filter(y, window_length=sw, polyorder=min(_SAVGOL_POLY, sw - 1))


def _region_masks(y_heavy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ac_power_norm, oscillation_mask, flat_mask) from the heavy-smoothed trace."""
    ac_power_norm = _sliding_fringe_power(y_heavy)
    osc_mask = ac_power_norm > _OSC_POWER_THRESH
    flat_mask = ac_power_norm < _FLAT_POWER_THRESH
    return ac_power_norm, osc_mask, flat_mask


def _decouple_index_in_flat_region(smoothed: np.ndarray, flat_mask: np.ndarray) -> int | None:
    """Coupler index of min |contrast| within the FFT-flat (idle) region."""
    flat_indices = np.where(flat_mask)[0]
    if flat_indices.size < _MIN_FLAT_POINTS:
        return None
    return int(flat_indices[np.argmin(np.abs(smoothed[flat_indices]))])


def _refine_decouple_index(
    y_fine: np.ndarray,
    flat_mask: np.ndarray,
    coarse_idx: int,
) -> int:
    """Refine decouple index using the lightly smoothed trace near the coarse flat minimum."""
    flat_indices = np.where(flat_mask)[0]
    if flat_indices.size < _MIN_FLAT_POINTS:
        return coarse_idx
    in_flat = flat_indices[
        (flat_indices >= max(flat_indices[0], coarse_idx - _REFINE_HALF_WIDTH_DECOUPLE))
        & (flat_indices <= min(flat_indices[-1], coarse_idx + _REFINE_HALF_WIDTH_DECOUPLE))
    ]
    if in_flat.size == 0:
        in_flat = flat_indices
    return int(in_flat[np.argmin(np.abs(y_fine[in_flat]))])


def _refine_gate_coupler_index(y_fine: np.ndarray, coarse_gate: int | None) -> int | None:
    """Refine gate-coupler dip index in a window around the coarse dip."""
    if coarse_gate is None:
        return None
    lo = max(0, coarse_gate - _REFINE_HALF_WIDTH_GATE)
    hi = min(len(y_fine) - 1, coarse_gate + _REFINE_HALF_WIDTH_GATE)
    seg = y_fine[lo : hi + 1]
    if seg.size < 3:
        return coarse_gate
    sig_range = float(seg.max() - seg.min())
    prom = max(_GATE_DIP_PROMINENCE, 0.05 * sig_range)
    dips, _ = find_peaks(-seg, prominence=prom)
    if len(dips) == 0:
        return coarse_gate
    global_dips = lo + dips.astype(int)
    return int(global_dips[np.argmin(np.abs(global_dips - coarse_gate))])


def _first_dip_beyond_decouple(
    seg_smoothed: np.ndarray,
    from_decouple_end: str,
    prominence: float,
    guard_pts: int,
) -> int | None:
    """First contrast dip searching away from the decouple index (left or right segment)."""
    n = len(seg_smoothed)
    if from_decouple_end == "right":
        s0, s1 = guard_pts, n
    else:
        s0, s1 = 0, max(guard_pts + 1, n - guard_pts)

    if s1 <= s0:
        return None

    search_region = seg_smoothed[s0:s1]
    sig_range = search_region.max() - search_region.min()
    prom = max(prominence, 0.05 * sig_range)
    dips, _ = find_peaks(-search_region, prominence=prom)
    if len(dips) == 0:
        return None

    local = int(dips[0] if from_decouple_end == "right" else dips[-1])
    return s0 + local


def _gate_coupler_index_from_contrast(
    smoothed: np.ndarray,
    decouple_idx: int,
    osc_mask: np.ndarray,
    coupler_rel: np.ndarray,
) -> int | None:
    """First contrast dip on the oscillation side of decouple (toward interaction fringes)."""
    osc_indices = np.where(osc_mask)[0]
    if osc_indices.size == 0:
        return None

    guard_pts = max(3, int(_GUARD_FRACTION * len(coupler_rel)))
    osc_center = float(coupler_rel[osc_indices].mean())
    decouple_v = float(coupler_rel[decouple_idx])

    dip_l = _first_dip_beyond_decouple(smoothed[: decouple_idx + 1], "left", _GATE_DIP_PROMINENCE, guard_pts)
    dip_r = _first_dip_beyond_decouple(smoothed[decouple_idx:], "right", _GATE_DIP_PROMINENCE, guard_pts)
    gate_l = dip_l
    gate_r = (decouple_idx + dip_r) if dip_r is not None else None

    if osc_center < decouple_v:
        return gate_l
    return gate_r


def _coarse_coupler_indices(
    y_heavy: np.ndarray,
    flat_mask: np.ndarray,
    osc_mask: np.ndarray,
    coupler_rel: np.ndarray,
) -> Tuple[int | None, int | None]:
    """Coarse decouple + gate coupler indices from heavy-smoothed contrast."""
    decouple = _decouple_index_in_flat_region(y_heavy, flat_mask)
    if decouple is None:
        return None, None
    gate = _gate_coupler_index_from_contrast(y_heavy, decouple, osc_mask, coupler_rel)
    return decouple, gate


def _refine_coupler_indices(
    y_fine: np.ndarray,
    flat_mask: np.ndarray,
    decouple_coarse: int | None,
    gate_coarse: int | None,
) -> Tuple[int | None, int | None]:
    """Refine decouple and gate indices on the lightly smoothed trace."""
    if decouple_coarse is None:
        return None, None
    decouple = _refine_decouple_index(y_fine, flat_mask, decouple_coarse)
    gate = _refine_gate_coupler_index(y_fine, gate_coarse)
    return decouple, gate


# ---------------------------------------------------------------------------
# Validation and per-pair fit
# ---------------------------------------------------------------------------


def _index_on_sweep_boundary(idx: int, size: int) -> bool:
    return size <= 1 or idx == 0 or idx == size - 1


def _coupler_fit_is_valid(
    decouple_idx: int | None, gate_idx: int | None, n_coupler: int
) -> bool:
    if decouple_idx is None or gate_idx is None:
        return False
    if decouple_idx == gate_idx:
        return False
    if _index_on_sweep_boundary(decouple_idx, n_coupler):
        return False
    if _index_on_sweep_boundary(gate_idx, n_coupler):
        return False
    return True


def _fit_pair_from_contrast_cut(
    control: xr.DataArray,
    target: xr.DataArray,
    coupler_rel: np.ndarray,
    coupler_full: xr.DataArray,
    qubit_full: xr.DataArray,
    cz_or_iswap: str,
) -> FitParameters:
    """Full contrast-cut fit for one pair (see module docstring pipeline)."""
    contrast = _interaction_map(control, target, cz_or_iswap)
    qubit_idx = _qubit_flux_cut_index(contrast, cz_or_iswap)
    optimal_qubit_flux = float(qubit_full.isel(qubit_flux=qubit_idx).values)

    y_raw = contrast.isel(qubit_flux=qubit_idx).values.ravel().astype(float)
    n_coupler = y_raw.size
    if n_coupler < 5:
        return FitParameters(
            success=False,
            optimal_qubit_flux=optimal_qubit_flux,
            optimal_decouple_offset=np.nan,
        )

    y_heavy = _savgol_smooth(y_raw, _SAVGOL_WINDOW_COARSE)
    y_fine = _savgol_smooth(y_raw, _SAVGOL_WINDOW_FINE)
    ac_power_norm, osc_mask, flat_mask = _region_masks(y_heavy)

    decouple_coarse, gate_coarse = _coarse_coupler_indices(
        y_heavy, flat_mask, osc_mask, coupler_rel
    )
    decouple_idx, gate_idx = _refine_coupler_indices(
        y_fine, flat_mask, decouple_coarse, gate_coarse
    )

    success = (
        not _index_on_sweep_boundary(qubit_idx, control.sizes["qubit_flux"])
        and _coupler_fit_is_valid(decouple_idx, gate_idx, n_coupler)
    )

    return FitParameters(
        success=success,
        optimal_qubit_flux=optimal_qubit_flux,
        optimal_decouple_offset=(
            float(coupler_full.isel(coupler_flux=decouple_idx).values)
            if decouple_idx is not None
            else np.nan
        ),
        optimal_decouple_coupler_flux_rel=(
            float(coupler_rel[decouple_idx]) if decouple_idx is not None else np.nan
        ),
        optimal_cz_coupler_flux=(
            float(coupler_rel[gate_idx]) if gate_idx is not None else np.nan
        ),
        optimal_cz_coupler_flux_total=(
            float(coupler_full.isel(coupler_flux=gate_idx).values)
            if gate_idx is not None
            else np.nan
        ),
        contrast_coupler_rel=coupler_rel.copy(),
        contrast_coupler_full=np.asarray(coupler_full).ravel().astype(float).copy(),
        contrast_raw=y_raw.copy(),
        contrast_smoothed=y_fine.copy(),
        ac_power_norm=ac_power_norm.copy(),
        osc_mask=osc_mask.copy(),
        flat_mask=flat_mask.copy(),
    )


def _extract_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Run the contrast-cut fit for every qubit pair in the dataset."""
    use_sd = node.parameters.use_state_discrimination
    cz_or_iswap = node.parameters.cz_or_iswap
    coupler_rel = np.asarray(fit.coupler_flux).astype(float)

    fit_results = {}
    for qp_name in fit.qubit_pair.values:
        qp_name = str(qp_name)
        if use_sd and "state_control" in fit:
            control = fit.state_control.sel(qubit_pair=qp_name)
            target = fit.state_target.sel(qubit_pair=qp_name)
        else:
            control = fit.I_control.sel(qubit_pair=qp_name)
            target = fit.I_target.sel(qubit_pair=qp_name)

        fit_results[qp_name] = _fit_pair_from_contrast_cut(
            control,
            target,
            coupler_rel,
            fit.coupler_flux_full.sel(qubit_pair=qp_name),
            fit.qubit_flux_full.sel(qubit_pair=qp_name),
            cz_or_iswap,
        )

    return fit, fit_results

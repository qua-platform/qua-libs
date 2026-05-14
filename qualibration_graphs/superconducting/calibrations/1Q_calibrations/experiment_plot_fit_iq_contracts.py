"""
No-hardware checks for plotting ↔ fitting ↔ IQ naming contracts on qubit spectroscopy.

Run from repo superconducting root with PYTHONPATH set to that folder:

    cd qualibration_graphs/superconducting
    set PYTHONPATH=%CD%
    python calibrations/1Q_calibrations/experiment_plot_fit_iq_contracts.py
"""

from __future__ import annotations

import inspect
import os
import sys

# -----------------------------------------------------------------------------
# Imports (need superconducting on path for calibration_utils)
# -----------------------------------------------------------------------------
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.qubit_spectroscopy import plotting as qs_plot


def _static_plot_uses_fit_for_traces() -> None:
    src = inspect.getsource(qs_plot.plot_individual_data_with_fit)
    # Main matplotlib .plot / .plot( calls use the fit dataset, not raw IQ_abs / I / Q on ds.
    assert "ds.IQ_abs" not in src, "Expected ds.IQ_abs never referenced in plot_individual_data_with_fit"
    assert "ds.I_rot" not in src, "Expected no ds.I_rot trace (I_rot comes from fit pipeline)"
    assert "fit.I_rot" in src or "fit.assign_coords" in src, "Expected rotated trace from fit side"
    assert "ds.detuning" in src, "Expected lorentzian x grid still taken from ds.detuning"
    print("[static] plot_individual_data_with_fit: main traces from fit.*; ds only ds.detuning for lorentzian x.")


def _static_node_wires_plot_to_fit_output() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent / "03a_qubit_spectroscopy.py"
    text = path.read_text(encoding="utf-8")
    assert 'plot_raw_data_with_fit(node.results["ds_raw"]' in text
    assert 'node.results["ds_fit"]' in text
    assert "plot_raw_data_with_fit" in text
    # plot_data always receives ds_fit produced in analyse_data; no alternate code path.
    idx_plot = text.index("def plot_data")
    idx_analyse = text.index("def analyse_data")
    assert idx_analyse < idx_plot, "analyse_data must be defined before plot_data in file order (run_action order follows registration)"
    print("[static] 03a plot_data always passes ds_raw + ds_fit from analyse_data; no raw-only plot API.")


def _runtime_fit_none_crashes() -> None:
    det = np.linspace(-40e6, 40e6, 9)
    ds = xr.Dataset(coords={"detuning": ("detuning", det)})
    fig, ax = plt.subplots()
    try:
        qs_plot.plot_individual_data_with_fit(ax, ds, {"qubit": "q1"}, fit=None)
    except (AttributeError, TypeError) as e:
        print(f"[runtime] fit=None -> {type(e).__name__}: (expected) {e}")
    else:
        raise AssertionError("Expected plot_individual_data_with_fit to fail when fit=None")


def _runtime_plot_works_with_minimal_ds_if_fit_has_I_rot() -> None:
    """ds lacks I/Q/IQ_abs; only detuning coord shared with lorentzian x — still plots via fit.I_rot."""
    det = np.linspace(-50e6, 50e6, 21)
    ds = xr.Dataset(coords={"detuning": ("detuning", det)})
    fit = xr.Dataset(
        data_vars={
            "I_rot": ("detuning", np.exp(-((det - 5e6) / 15e6) ** 2)),
            "full_freq": ("detuning", 4.3e9 + det),
            "amplitude": np.array(1.2),
            "position": np.array(5e6),
            "width": np.array(18e6),
            "base_line": ("detuning", np.full_like(det, 0.01, dtype=float)),
        },
        coords={"detuning": det},
    )
    fig, ax = plt.subplots()
    qs_plot.plot_individual_data_with_fit(ax, ds, {"qubit": "q1"}, fit)
    plt.close(fig)
    print("[runtime] Plot succeeded with ds carrying only detuning coord; trace data came from fit (I_rot).")


def _runtime_fit_missing_I_rot_raises() -> None:
    det = np.linspace(-10e6, 10e6, 5)
    ds = xr.Dataset(coords={"detuning": ("detuning", det)})
    fit = xr.Dataset(
        data_vars={
            "full_freq": ("detuning", 4e9 + det),
            "amplitude": np.array(1.0),
            "position": np.array(0.0),
            "width": np.array(5e6),
            "base_line": ("detuning", np.zeros_like(det)),
        },
        coords={"detuning": det},
    )
    fig, ax = plt.subplots()
    try:
        qs_plot.plot_individual_data_with_fit(ax, ds, {"qubit": "q1"}, fit)
    except (KeyError, AttributeError) as e:
        print(f"[runtime] fit without I_rot -> {type(e).__name__}: (expected) {e}")
    else:
        plt.close(fig)
        raise AssertionError("Expected failure when I_rot missing on fit")
    plt.close(fig)


def main() -> int:
    print("=== Qubit spectroscopy plot / fit / IQ contract experiment (no hardware) ===\n")
    _static_plot_uses_fit_for_traces()
    _static_node_wires_plot_to_fit_output()
    print()
    _runtime_fit_none_crashes()
    _runtime_fit_missing_I_rot_raises()
    _runtime_plot_works_with_minimal_ds_if_fit_has_I_rot()
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

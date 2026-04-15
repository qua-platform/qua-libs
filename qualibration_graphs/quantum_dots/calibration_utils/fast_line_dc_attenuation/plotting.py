"""Plotting utilities for fast-line DC attenuation calibration."""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _plot_sensor_dot_ax(
    ax: plt.Axes,
    dc_values: np.ndarray,
    signal: np.ndarray,
    key: str,
    fit_result: dict | None,
) -> None:
    """Plot double-peak trace (sensor dot) with peak markers."""
    ax.plot(dc_values, signal, "b-", lw=1, alpha=0.8)
    ax.scatter(dc_values, signal, c="b", s=6, alpha=0.5, zorder=3)
    ax.set_xlabel("DC voltage (V)")
    ax.set_ylabel("Signal amplitude")
    ax.set_title(f"{key} (sensor dot)")

    if fit_result and fit_result.get("success"):
        for label, pos_key in [("Peak 1", "peak_position_1"), ("Peak 2", "peak_position_2")]:
            pos = fit_result.get(pos_key)
            if pos is not None and np.isfinite(pos):
                ax.axvline(pos, color="lime", ls="--", lw=1.2, alpha=0.9, label=f"{label} = {pos:.6f} V")
        sep = fit_result.get("dc_separation", 0)
        ax.annotate(
            f"sep = {sep:.6f} V",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )
    ax.legend(loc="upper right", fontsize=7)


def _plot_non_sensor_ax(
    ax_signal: plt.Axes,
    ax_deriv: plt.Axes,
    dc_values: np.ndarray,
    signal: np.ndarray,
    key: str,
    fit_result: dict | None,
) -> None:
    """Plot step trace (non-sensor) with derivative and peak markers."""
    ax_signal.plot(dc_values, signal, "b-", lw=1, alpha=0.8)
    ax_signal.scatter(dc_values, signal, c="b", s=6, alpha=0.5, zorder=3)
    ax_signal.set_xlabel("DC voltage (V)")
    ax_signal.set_ylabel("Signal amplitude")
    ax_signal.set_title(f"{key} (signal)")

    derivative = (fit_result or {}).get("_derivative")
    if derivative is not None:
        deriv_abs = np.abs(derivative)
        ax_deriv.plot(dc_values, deriv_abs, "b-", lw=1, alpha=0.8)
        ax_deriv.scatter(dc_values, deriv_abs, c="b", s=6, alpha=0.5, zorder=3)

    if fit_result and fit_result.get("success"):
        for label, pos_key in [("Peak 1", "peak_position_1"), ("Peak 2", "peak_position_2")]:
            pos = fit_result.get(pos_key)
            if pos is not None and np.isfinite(pos):
                ax_deriv.axvline(pos, color="lime", ls="--", lw=1.2, alpha=0.9, label=f"{label} = {pos:.6f} V")
                ax_signal.axvline(pos, color="lime", ls="--", lw=1.2, alpha=0.5)
        ax_deriv.legend(loc="upper right", fontsize=7)
        sep = fit_result.get("dc_separation", 0)
        ax_deriv.annotate(
            f"sep = {sep:.6f} V",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    ax_deriv.set_xlabel("DC voltage (V)")
    ax_deriv.set_ylabel("|d(signal)/dV|")
    ax_deriv.set_title(f"{key} (|derivative|)")


def _plot_iq_ax(
    ax_i: plt.Axes,
    ax_q: plt.Axes,
    dc_values: np.ndarray,
    fit_result: dict | None,
    sensor_name: str,
) -> None:
    """Plot raw I and Q quadratures."""
    I = (fit_result or {}).get("_I")
    Q = (fit_result or {}).get("_Q")
    if I is not None:
        ax_i.plot(dc_values, I, "b-", lw=1, alpha=0.8)
        ax_i.set_ylabel(f"{sensor_name}")
        ax_i.set_xlabel("DC voltage (V)")
    if Q is not None:
        ax_q.plot(dc_values, Q, "b-", lw=1, alpha=0.8)
        ax_q.set_xlabel("DC voltage (V)")


def plot_raw_data_with_fit(
    ds,
    ds_fit,
    node,
    fit_results: Dict[str, Dict[str, Any]],
) -> Dict[str, plt.Figure]:
    """Create per-component figures with analysis overlays.

    Returns a dict mapping ``"figure_<comp>"`` to :class:`~matplotlib.figure.Figure`.
    """
    components: List[str] = node.namespace["components"]
    sensors = node.namespace["sensor_names"]

    figures: Dict[str, plt.Figure] = {}

    for comp in components:
        sensor_dot_names = set(node.machine.sensor_dots.keys()) if hasattr(node.machine, "sensor_dots") else set()
        is_sensor_dot = comp in sensor_dot_names

        n_sensors = len(sensors)
        if is_sensor_dot:
            ncol = 3  # I, Q, fit
        else:
            ncol = 4  # I, Q, signal+steps, derivative+fit

        fig, axes = plt.subplots(n_sensors, ncol, figsize=(5 * ncol, 4 * n_sensors), squeeze=False)
        fig.suptitle(f"Component: {comp}  ({'sensor dot' if is_sensor_dot else 'non-sensor'})")

        for i, sensor in enumerate(sensors):
            key = f"{comp}__{sensor.name}"
            fr = fit_results.get(key, {})
            dc_values = fr.get("_dc_values")
            if dc_values is None:
                continue

            _plot_iq_ax(axes[i, 0], axes[i, 1], dc_values, fr, sensor.name)
            if i == 0:
                axes[i, 0].set_title("I")
                axes[i, 1].set_title("Q")

            signal = fr.get("_signal")
            if signal is None:
                continue

            if is_sensor_dot:
                _plot_sensor_dot_ax(axes[i, 2], dc_values, signal, key, fr)
            else:
                _plot_non_sensor_ax(axes[i, 2], axes[i, 3], dc_values, signal, key, fr)

        fig.tight_layout()
        figures[f"figure_{comp}"] = fig

    return figures

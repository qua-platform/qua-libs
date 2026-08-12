from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from qualibration_libs.parameters import get_qubits

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

__all__ = ["generate_simulated_dataset"]


def _gaussian_2d(
    hold_grid: np.ndarray,
    ramp_grid: np.ndarray,
    *,
    hold_center: float,
    ramp_center: float,
    hold_sigma: float,
    ramp_sigma: float,
) -> np.ndarray:
    """Return a unit-height separable 2D Gaussian."""
    hold_term = ((hold_grid - hold_center) / max(float(hold_sigma), 1.0)) ** 2
    ramp_term = ((ramp_grid - ramp_center) / max(float(ramp_sigma), 1.0)) ** 2
    return np.exp(-0.5 * (hold_term + ramp_term))


def generate_simulated_dataset(node: QualibrationNode) -> xr.Dataset:
    """Generate 08c-style raw maps for the real processing and plotting path.

    The returned dataset intentionally matches the post-fetch, pre-processing
    layout produced by the current real OPX path for 08c, including the
    collapsed ``qubit`` dimension created from handles such as
    ``I_no_pulse_q1``/``I_no_pulse_q4``. This lets the existing
    ``process_raw_data()``, ``analyse_data()``, and ``plot_data()`` actions run
    unchanged on simulated data.
    """
    node.namespace["qubits"] = qubits = get_qubits(node)

    hold_durations = np.arange(
        node.parameters.hold_duration_start,
        node.parameters.hold_duration_stop,
        node.parameters.hold_duration_step,
        dtype=float,
    )
    ramp_durations = np.arange(
        node.parameters.ramp_duration_start,
        node.parameters.ramp_duration_stop,
        node.parameters.ramp_duration_step,
        dtype=float,
    )

    if len(hold_durations) == 0:
        hold_durations = np.array([float(node.parameters.hold_duration_start)])
    if len(ramp_durations) == 0:
        ramp_durations = np.array([float(node.parameters.ramp_duration_start)])

    hold_grid, ramp_grid = np.meshgrid(
        hold_durations,
        ramp_durations,
        indexing="ij",
    )

    rng = np.random.default_rng(seed=42)
    qubit_names = [qubit.name for qubit in qubits]
    shape = (len(qubits), len(hold_durations), len(ramp_durations))

    i_no_pulse = np.empty(shape, dtype=float)
    q_no_pulse = np.empty(shape, dtype=float)
    i_pulse = np.empty(shape, dtype=float)
    q_pulse = np.empty(shape, dtype=float)
    state_no_pulse = np.empty(shape, dtype=float)
    state_pulse = np.empty(shape, dtype=float)

    hold_span = max(float(np.ptp(hold_durations)), 1.0)
    ramp_span = max(float(np.ptp(ramp_durations)), 1.0)

    for index, qubit in enumerate(qubits):
        hold_center = float(
            hold_durations[min(len(hold_durations) - 1, len(hold_durations) // 3 + index)]
        )
        ramp_center = float(
            ramp_durations[min(len(ramp_durations) - 1, len(ramp_durations) // 2 + index)]
        )

        response = _gaussian_2d(
            hold_grid,
            ramp_grid,
            hold_center=hold_center,
            ramp_center=ramp_center,
            hold_sigma=max(hold_span / 6.0, float(node.parameters.hold_duration_step)),
            ramp_sigma=max(ramp_span / 6.0, float(node.parameters.ramp_duration_step)),
        )

        i_baseline = (
            0.015 * index
            + 0.01 * (hold_grid - hold_durations.mean()) / hold_span
            + rng.normal(scale=0.002, size=hold_grid.shape)
        )
        q_baseline = (
            -0.01 * index
            + 0.01 * (ramp_grid - ramp_durations.mean()) / ramp_span
            + rng.normal(scale=0.002, size=hold_grid.shape)
        )
        state_baseline = np.clip(
            0.08 + 0.02 * index + rng.normal(scale=0.005, size=hold_grid.shape),
            0.0,
            1.0,
        )

        i_response = 0.12 * response
        q_response = 0.07 * response
        state_response = 0.55 * response

        i_no_pulse[index] = i_baseline
        q_no_pulse[index] = q_baseline
        i_pulse[index] = i_baseline + i_response
        q_pulse[index] = q_baseline + q_response
        state_no_pulse[index] = state_baseline
        state_pulse[index] = np.clip(state_baseline + state_response, 0.0, 1.0)

    coords = {
        "qubit": xr.DataArray(qubit_names, dims="qubit"),
        "hold_durations": xr.DataArray(
            hold_durations,
            dims="hold_durations",
            attrs={"long_name": "Init hold duration", "units": "ns"},
        ),
        "ramp_durations": xr.DataArray(
            ramp_durations,
            dims="ramp_durations",
            attrs={"long_name": "Init ramp duration", "units": "ns"},
        ),
    }

    return xr.Dataset(
        {
            "I_no_pulse_q": (
                ["qubit", "hold_durations", "ramp_durations"],
                i_no_pulse,
            ),
            "Q_no_pulse_q": (
                ["qubit", "hold_durations", "ramp_durations"],
                q_no_pulse,
            ),
            "I_pulse_q": (
                ["qubit", "hold_durations", "ramp_durations"],
                i_pulse,
            ),
            "Q_pulse_q": (
                ["qubit", "hold_durations", "ramp_durations"],
                q_pulse,
            ),
            "state_no_pulse_q": (
                ["qubit", "hold_durations", "ramp_durations"],
                state_no_pulse,
            ),
            "state_pulse_q": (
                ["qubit", "hold_durations", "ramp_durations"],
                state_pulse,
            ),
        },
        coords=coords,
    )

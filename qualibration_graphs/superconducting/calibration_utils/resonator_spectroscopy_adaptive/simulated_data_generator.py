from __future__ import annotations

import numpy as np
import xarray as xr
from qualang_tools.units import unit
from qualibration_libs.parameters import get_qubits


def generate_simulated_dataset(node) -> xr.Dataset:
    """Generate a seeded notch-resonator response for every selected qubit."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    detuning = np.arange(-span / 2, span / 2, step)
    names = qubits.get_names()
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(names),
        "detuning": xr.DataArray(detuning, attrs={"long_name": "readout frequency", "units": "Hz"}),
    }

    rng = np.random.default_rng(42)
    signals = []
    for index, _ in enumerate(names):
        center = rng.uniform(-0.06 * span, 0.06 * span)
        linewidth = rng.uniform(0.8e6, 1.8e6)
        delta = detuning - center
        s11 = 1 - 0.75 * (linewidth / 2) / (linewidth / 2 + 1j * delta)
        ripple = 1 + 0.025 * np.sin(2 * np.pi * detuning / (0.3 * span) + index)
        signal = 1e-3 * ripple * s11 * np.exp(1j * (0.4 + 0.2 * index))
        signal += rng.normal(0, 4e-6, detuning.size) + 1j * rng.normal(0, 4e-6, detuning.size)
        signals.append(signal)

    signal = np.asarray(signals)
    return xr.Dataset(
        data_vars={
            "I": (("qubit", "detuning"), signal.real),
            "Q": (("qubit", "detuning"), signal.imag),
        },
        coords={"qubit": names, "detuning": detuning},
    )

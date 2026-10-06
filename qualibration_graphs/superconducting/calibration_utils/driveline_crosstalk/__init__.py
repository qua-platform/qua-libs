"""Public helpers for the drive-line crosstalk calibrations."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
from qualibrate.core.config import get_config_path, get_settings
from qualibrate.core.storage.local_storage_manager import LocalStorageManager
from qualibration_libs.parameters import get_qubits

from .analysis import check_compensation, fit_amplitude, fit_phase, log_validation_results, select_phase
from .parameters import AmplitudeParameters, CheckParameters, PhaseParameters, amplitude_factors
from .plotting import plot_check, plot_matrix, plot_phase, plot_rabi, plot_validation_table

__all__ = [
    "AmplitudeParameters",
    "PhaseParameters",
    "CheckParameters",
    "amplitude_factors",
    "fit_amplitude",
    "fit_phase",
    "select_phase",
    "check_compensation",
    "log_validation_results",
    "plot_rabi",
    "plot_matrix",
    "plot_phase",
    "plot_check",
    "plot_validation_table",
    "prepare_machine",
    "read_matrix",
    "save_matrix",
]


def _configure_storage(node):
    if node.storage_manager is None:
        node.storage_manager = LocalStorageManager(
            get_settings(get_config_path()).storage.location, active_machine_path=None
        )
    node.storage_manager.active_machine_path = None


def prepare_machine(node):
    """Use the refreshed state with a common LO, without writing active state."""
    p = node.parameters
    type(p).model_validate(p.model_dump())
    _configure_storage(node)
    if len(p.qubits) < 2 or len(set(p.qubits)) != len(p.qubits):
        raise ValueError("Select at least two distinct qubits")
    fingerprint = hashlib.sha256(json.dumps(node.machine.to_dict(), sort_keys=True).encode()).hexdigest()
    qubits = list(get_qubits(node))
    for q in qubits:
        pulse = q.xy.operations[p.operation]
        if "SquarePulse" not in type(pulse).__name__ or pulse.amplitude <= 0:
            raise ValueError(f"{q.name}: use a positive square-pulse operation")
        port = q.xy.opx_output
        if port.upconverters is None:
            port.upconverter_frequency = p.common_lo_frequency_hz
        else:
            port.upconverters[q.xy.upconverter].frequency = p.common_lo_frequency_hz
        if not np.isclose(q.xy.LO_frequency, p.common_lo_frequency_hz):
            raise ValueError(f"{q.name}: could not set common LO")
        if abs(q.xy.RF_frequency - q.xy.LO_frequency) > p.max_drive_if_in_mhz * 1e6:
            raise ValueError(f"{q.name}: RF is outside the requested IF range")
    node.namespace["qubits"] = qubits
    names = [q.name for q in qubits]
    context = xr.Dataset(coords={"qubit": names, "drive_qubit": names})
    context["target_rf_hz"] = ("qubit", [q.xy.RF_frequency for q in qubits])
    context["drive_base_amplitude"] = ("drive_qubit", [q.xy.operations[p.operation].amplitude for q in qubits])
    context["drive_mV_per_prefactor"] = (
        "drive_qubit",
        [
            q.xy.operations[p.operation].amplitude
            * np.sqrt(0.1 * 10 ** (q.xy.opx_output.full_scale_power_dbm / 10))
            * 1000
            for q in qubits
        ],
    )
    context.attrs.update(
        state_fingerprint=fingerprint,
        operation=p.operation,
        pulse_length_ns=p.pulse_length_ns,
        common_lo_frequency_hz=p.common_lo_frequency_hz,
        phase_reference="reset_global_phase + reset_if_phase; target frame relative to source",
        matrix_orientation="rows: drive_qubit; columns: qubit (target)",
    )
    node.namespace["context"] = context


def read_matrix(node, name):
    path = Path(node.parameters.matrix_directory) / f"{name}_matrix.h5"
    ds = xr.load_dataset(path)
    context = node.namespace["context"]
    for key, value in context.attrs.items():
        if ds.attrs.get(key) != value:
            raise ValueError(f"{path.name}: {key} changed; rerun amplitude then phase calibration")
    for dim in ("qubit", "drive_qubit"):
        if not np.array_equal(ds[dim], context[dim]):
            raise ValueError(f"{path.name}: selected qubits/order differ")
    node.namespace[f"{name}_matrix_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    node.results[f"{name}_source"] = str(path)
    return ds


def save_matrix(node, name):
    """Publish an exact copy of the normal node's H5 dataset; never a CSV."""
    _configure_storage(node)
    node.save()
    if node.parameters.simulate or node.parameters.load_data_id is not None:
        return
    source = Path(node.storage_manager.data_handler.path) / "ds_fit.h5"
    directory = Path(node.parameters.matrix_directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".{name}_matrix.h5"
    shutil.copyfile(source, temporary)
    temporary.replace(directory / f"{name}_matrix.h5")
    node.log(f"Saved #{node.snapshot_idx}: {source.parent}; {name}_matrix.h5 updated")

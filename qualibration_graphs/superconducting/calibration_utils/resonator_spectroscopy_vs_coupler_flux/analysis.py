"""Analysis functions for resonator spectroscopy versus coupler flux calibration."""

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase

from calibration_utils.resonator_spectroscopy_vs_flux.analysis import (
    fit_raw_data,
    log_fitted_results,
)

__all__ = ["process_raw_dataset", "fit_raw_data", "log_fitted_results"]


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Process the raw dataset for coupler flux spectroscopy.

    Identical in purpose to the qubit-flux version, but the IQ→V conversion
    is indexed by qubit-pair (coupler) names rather than measured-qubit names.
    This avoids a duplicate-index error when two different qubit pairs share
    the same measured qubit.
    """
    qubit_pairs = node.namespace["qubit_pairs"]
    measured_qubits = node.namespace["measured_qubits"]

    # Build readout lengths keyed by the coupler name so the coordinate
    # matches the 'qubit' dimension already in ds (which uses coupler names).
    readout_lengths = xr.DataArray(
        [q.resonator.operations["readout"].length for q in measured_qubits],
        coords=[("qubit", [qp.name for qp in qubit_pairs])],
    )
    # Convert I and Q from demodulation units to Volts (same formula as
    # convert_IQ_to_V with single_demod=False).
    ds = ds.assign({key: ds[key] * 2**12 / readout_lengths for key in ("I", "Q")})
    # Add the amplitude and phase to the raw dataset
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    # Add the RF frequency as a per-(qubit, detuning) coordinate.
    full_freq = np.array([ds.detuning.values + q.resonator.RF_frequency for q in measured_qubits])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    # Add the coupler current axis as a per-flux_bias coordinate.
    current = ds.flux_bias / node.parameters.input_line_impedance_in_ohm
    ds = ds.assign_coords({"current": (["flux_bias"], current.data)})
    ds.current.attrs = {"long_name": "Current", "units": "A"}
    # Add attenuated current to dataset
    attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
    attenuated_current = ds.current * attenuation_factor
    ds = ds.assign_coords({"attenuated_current": (["flux_bias"], attenuated_current.values)})
    ds.attenuated_current.attrs = {"long_name": "Attenuated Current", "units": "A"}

    return ds

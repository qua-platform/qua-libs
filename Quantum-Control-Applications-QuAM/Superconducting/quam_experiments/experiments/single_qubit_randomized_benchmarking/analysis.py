import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.qua_datasets import add_amplitude_and_phase, convert_IQ_to_V
from quam_experiments.analysis.fit import peaks_dips
from quam_config.instrument_limits import instrument_limits


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    success: bool
    qubit_name: Optional[str] = ""


def log_fitted_results(fit_results: Dict, logger=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    # for q in fit_results.keys():
    #     s_qubit = f"Results for qubit {q}: "
    #     s_freq = f"\tQubit frequency: {1e-9 * fit_results[q]['frequency']:.3f} GHz | "
    #     s_fwhm = f"FWHM: {1e-3 * fit_results[q]['fwhm']:.1f} kHz | "
    #     s_angle = (
    #         f"The integration weight angle: {fit_results[q]['iw_angle']:.3f} rad\n "
    #     )
    #     s_saturation = f"To get the desired FWHM, the saturation amplitude is updated to: {1e3 * fit_results[q]['saturation_amp']:.1f} mV | "
    #     s_x180 = f"To get the desired x180 gate, the x180 amplitude is updated to: {1e3 * fit_results[q]['x180_amp']:.1f} mV\n "
    #     if fit_results[q]["success"]:
    #         s_qubit += " SUCCESS!\n"
    #     else:
    #         s_qubit += " FAIL!\n"
    #     logger.info(
    #         s_qubit + s_freq + s_fwhm + s_freq + s_angle + s_saturation + s_x180
    #     )
    pass


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    # ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    # ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    # full_freq = np.array(
    #     [ds.detuning + q.xy.RF_frequency for q in node.namespace["qubits"]]
    # )
    # ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    # ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    pass
    # Add the qubit pulse absolute alpha coefficient to the dataset
    # ds = ds.assign_coords(
    #     {
    #         "alpha": (
    #             ["qubit", "amp"],
    #             np.array([q.xy.operations[operation].alpha * amps for q in qubits]),
    #         )
    #     }
    # )
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency and FWHM for each qubit in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    # # %% {Data_analysis}
    # da_state = 1 - ds["state"].mean(dim="sequence")
    # da_state: xr.DataArray
    # da_state.attrs = {"long_name": "p(0)"}
    # da_state = da_state.assign_coords(depths=da_state.depths - 1)
    # da_state = da_state.rename(depths="m")
    # da_state.m.attrs = {"long_name": "no. of Cliffords"}
    # # Fit the exponential decay
    # da_fit = fit_decay_exp(da_state, "m")
    # # Extract the decay rate
    # alpha = np.exp(da_fit.sel(fit_vals="decay"))
    # # average_gate_per_clifford = 45/24 = 1.875
    # average_gate_per_clifford = (1 * 3 + 9 * 2 + 1 * 4 + 2 * 3 + 4 * 2 + 2 * 3) / 24
    # # EPC from here: https://qiskit.org/textbook/ch-quantum-hardware/randomized-benchmarking.html#Step-5:-Fit-the-results
    # EPC = (1 - alpha) - (1 - alpha) / 2
    # EPG = EPC / average_gate_per_clifford

    ds_fit = ds
    fit_results = {
        q: FitParameters(
            success=False,
        )
        for q in ds_fit.qubit.values
    }
    return ds_fit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    pass
    # # Save fitting results
    # node.results["fit_results"] = {}
    # for q in qubits:
    #     node.results["fit_results"][q.name] = {}
    #     node.results["fit_results"][q.name]["EPC"] = EPC.sel(qubit=q.name).values
    #     node.results["fit_results"][q.name]["EPG"] = EPG.sel(qubit=q.name).values
    #     print(f"{q.name}: EPC={EPC.sel(qubit=q.name).values}")
    #     print(f"{q.name}: EPG={EPG.sel(qubit=q.name).values}")
    # node.outcomes = {q.name: "successful" for q in node.namespace["qubits"]}

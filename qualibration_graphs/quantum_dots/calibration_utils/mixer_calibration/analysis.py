import logging
from dataclasses import dataclass
from typing import Dict, Optional

from qualibrate import QualibrationNode
from qualang_tools.octave_tools.calibration_result_plotter import CalibrationResultPlotter


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    resonator: Optional[dict] = None
    xy_drive: Optional[dict] = None
    success: Optional[bool] = True


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all elements from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all elements.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for element_name in fit_results.keys():
        s_element = f"Results for {element_name}: "
        if fit_results[element_name]["resonator"] is not None:
            s_res = f"\tresonator -> LO leakage suppression: {fit_results[element_name]['resonator']['lo_leakage']:.1f} dB | image rejection: {fit_results[element_name]['resonator']['image_rejection']:.1f} dB.\n"
        else:
            s_res = ""
        if fit_results[element_name]["xy_drive"] is not None:
            s_xy = f"\txy_drive  -> LO leakage suppression: {fit_results[element_name]['xy_drive']['lo_leakage']:.1f} dB | image rejection: {fit_results[element_name]['xy_drive']['image_rejection']:.1f} dB.\n"
        else:
            s_xy = ""
        if fit_results[element_name]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_res + s_xy)


def extract_relevant_fit_parameters(node: QualibrationNode):
    """Add metadata to the dataset and fit results."""

    cal_results = node.namespace["calibration_results"]
    fit_results = {
        q: FitParameters(
            resonator=(
                {
                    "lo_leakage": CalibrationResultPlotter(cal_results[q]["resonator"]).get_lo_leakage_rejection(),
                    "image_rejection": CalibrationResultPlotter(cal_results[q]["resonator"]).get_image_rejection(),
                }
                if node.parameters.calibrate_resonator
                else None
            ),
            xy_drive=(
                {
                    "lo_leakage": CalibrationResultPlotter(cal_results[q]["xy_drive"]).get_lo_leakage_rejection(),
                    "image_rejection": CalibrationResultPlotter(cal_results[q]["xy_drive"]).get_image_rejection(),
                }
                if node.parameters.calibrate_drive
                else None
            ),
            success=True,
        )
        for q in node.namespace["calibration_results"].keys()
    }
    return fit_results

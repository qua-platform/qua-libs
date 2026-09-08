import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from qualibrate.core import QualibrationNode
from qualang_tools.octave_tools.calibration_result_plotter import (
    CalibrationResultPlotter,
)


@dataclass
class FitParameters:
    """Stores Octave mixer calibration metrics for a single element (sensor or qubit)."""

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
            s_res = (
                f"\tresonator -> LO leakage suppression: "
                f"{fit_results[element_name]['resonator']['lo_leakage']:.1f} dB | "
                f"image rejection: {fit_results[element_name]['resonator']['image_rejection']:.1f} dB.\n"
            )
        else:
            s_res = ""
        if fit_results[element_name]["xy_drive"] is not None:
            s_xy = (
                f"\txy_drive  -> LO leakage suppression: "
                f"{fit_results[element_name]['xy_drive']['lo_leakage']:.1f} dB | "
                f"image rejection: {fit_results[element_name]['xy_drive']['image_rejection']:.1f} dB.\n"
            )
        else:
            s_xy = ""
        if fit_results[element_name]["success"]:
            s_element += " SUCCESS!\n"
        else:
            s_element += " FAIL!\n"
        log_callable(s_element + s_res + s_xy)


def _metrics_from_calibration_result(cal_result: Any) -> Optional[dict]:
    """Extract LO leakage / image rejection, or ``None`` if unavailable."""
    if cal_result is None:
        return None
    try:
        plotter = CalibrationResultPlotter(cal_result)
        return {
            "lo_leakage": plotter.get_lo_leakage_rejection(),
            "image_rejection": plotter.get_image_rejection(),
        }
    except Exception:
        return None


def extract_relevant_fit_parameters(node: QualibrationNode):
    """Build per-element fit parameters from whatever channels were actually calibrated."""
    fit_results = {}
    for element_name, element_cal in node.namespace["calibration_results"].items():
        resonator = None
        if "resonator" in element_cal and node.parameters.calibrate_resonator:
            resonator = _metrics_from_calibration_result(element_cal.get("resonator"))

        xy_drive = None
        if "xy_drive" in element_cal and node.parameters.calibrate_drive:
            xy_drive = _metrics_from_calibration_result(element_cal.get("xy_drive"))

        success = resonator is not None or xy_drive is not None
        fit_results[element_name] = FitParameters(
            resonator=resonator,
            xy_drive=xy_drive,
            success=success,
        )
    return fit_results

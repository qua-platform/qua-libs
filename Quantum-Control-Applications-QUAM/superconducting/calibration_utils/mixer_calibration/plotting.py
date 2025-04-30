from typing import Dict
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualang_tools.octave_tools.calibration_result_plotter import CalibrationResultPlotter
from qualibrate import QualibrationNode
from qm.octave.octave_mixer_calibration import MixerCalibrationResults

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(node: QualibrationNode):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.
    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    figures = {}
    for qubit in node.namespace["qubits"]:
        figs = plot_individual_data_with_fit(node.namespace["calibration_results"], qubit.name)
        figures["qubit.name"] = figs
    return figures


def plot_individual_data_with_fit(calibration_results: Dict[str, Dict[str, MixerCalibrationResults]], qubit_name: str):
    """
    Plots individual qubit data on a given axis with optional fit.
    """
    figs = {}
    for key in ["resonator", "xy_drive"]:
        if calibration_results[qubit_name][key] is not None:
            plotter = CalibrationResultPlotter(calibration_results[qubit_name][key])
            # LO leakage
            fig_lo_leakage = plotter.show_lo_leakage_calibration_result()
            fig_lo_leakage.suptitle(qubit_name + "." + key + ": " + fig_lo_leakage._suptitle.get_text())
            # Image rejection
            fig_image_rejection = plotter.show_image_rejection_calibration_result()
            fig_image_rejection.suptitle(qubit_name + "." + key + ": " + fig_image_rejection._suptitle.get_text())
            figs[key] = {"lo_leakage": fig_lo_leakage, "image_rejection": fig_image_rejection}
    return figs

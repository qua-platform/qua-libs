"""
Signal processing utilities for Qualibrate nodes.
"""

from scipy.signal import savgol_filter
import numpy as np


from qualibrate import QualibrationNode

from qualang_tools.units import unit

# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)


def process_raw_data(fetched_data: dict):
    """
    Convert raw ADC traces into volts.

    Args:
        fetched_data (dict): Dictionary of raw results (typically from the execution node).

    Modifies:
        fetched_data (in-place): All NumPy array values are converted to volts using unit conversions.
    """
    # Convert raw ADC traces into volts
    for key, value in fetched_data.items():
        if isinstance(value, np.ndarray):
            fetched_data[key] = u.raw2volts(value)


def fit_raw_data(fetched_data: dict, node: QualibrationNode):
    """
    Analyze raw ADC data to extract features like mean and pulse arrival delay.

    Performs the following steps:
    - Computes and stores the mean of each signal (under "mean_values").
    - Mean-centers each ADC trace.
    - Detects pulse arrival time via Savitzky-Golay filter + threshold.

    Args:
        fetched_data (dict): Dictionary of preprocessed results (in volts).
        node (QualibrationNode): Node containing metadata (especially `node.parameters.resonators`).

    Modifies:
        fetched_data (in-place):
            - Adds "mean_values" dict
            - Adds "delay{i}" entries for each resonator
    """
    # Derive the average values
    mean_values = {}
    for key, value in fetched_data.items():
        if isinstance(value, np.ndarray):
            mean_values[key] = value.mean()
        elif isinstance(value, (int, float)):
            mean_values[key] = value
    # Save the average values
    fetched_data["mean_values"] = mean_values

    # Remove the average values
    for key, value in fetched_data.items():
        if isinstance(value, np.ndarray):
            fetched_data[key] = value - mean_values[key]

    # Filter the data to get the pulse arrival time
    for i in range(len(node.parameters.resonators)):
        signal = savgol_filter(np.abs(fetched_data[f"adcI{i + 1}"] + 1j * fetched_data[f"adcQ{i + 1}"]), 11, 3)
        # Detect the arrival of the readout signal
        th = (np.mean(signal[:100]) + np.mean(signal[:-100])) / 2
        delay = np.where(signal > th)[0][0]
        fetched_data[f"delay{i + 1}"] = np.round(delay / 4) * 4  # Find the closest multiple integer of 4ns

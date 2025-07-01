"""
Signal processing utilities for Qualibrate nodes.
"""

from scipy.signal import savgol_filter
import numpy as np

from qualibrate import QualibrationNode

from qualang_tools.units import unit

# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)


def process_raw_data(raw_data: dict) -> dict:
    """
    Convert raw ADC traces into volts.

    Args:
        raw_data (dict): Dictionary of raw results.

    Returns:
        processed_data (dict): Same keys as raw_data, but every NumPy array converted to volts.
    """
    processed_data = {}
    # Convert raw ADC traces into volts
    for key, value in raw_data.items():
        if isinstance(value, np.ndarray):
            processed_data[key] = u.raw2volts(value) if isinstance(value, np.ndarray) else value

    return processed_data

def fit_raw_data(processed_data: dict, num_resonators: int) -> dict:
    """
    Analyze raw ADC data to extract features like mean and pulse arrival delay.

    Performs the following steps:
    - Computes and stores the mean of each signal (under "mean_values").
    - Mean-centers each ADC trace.
    - Detects pulse arrival time via Savitzky-Golay filter + threshold.

    Args:
        processed_data (dict): Dictionary of preprocessed results (in volts).
        num_resonators (int): Number of resonators (or qubits).

    Returns:
        ds_fit (dict): A dictionary with the following keys:
            - 'mean_values': Per-signal mean offset (used for centering).
            - 'processed_data_without_mean': Mean-centered ADC traces.
            - 'delay{i}': Extracted pulse arrival delay (in ns) for each resonator.

    """
    ds_fit = {}

    # Derive the average values
    mean_values = {}
    for key, value in processed_data.items():
        if isinstance(value, np.ndarray):
            mean_values[key] = value.mean()
        elif isinstance(value, (int, float)):
            mean_values[key] = value
    # Save the average values
    ds_fit["mean_values"] = mean_values

    # Remove the average values
    ds_fit["processed_data_without_mean"] = {}
    for key, value in processed_data.items():
        if isinstance(value, np.ndarray):
            ds_fit["processed_data_without_mean"][key] = value - ds_fit["mean_values"][key]
        elif isinstance(value, (int, float)):
            ds_fit["processed_data_without_mean"][key] = value

    # Filter the data to get the pulse arrival time
    for i in range(num_resonators):
        I = ds_fit["processed_data_without_mean"][f"adcI{i + 1}"]
        Q = ds_fit["processed_data_without_mean"][f"adcQ{i + 1}"]
        signal = savgol_filter(np.abs(I + 1j * Q), 11, 3)
        # Detect the arrival of the readout signal
        th = (np.mean(signal[:100]) + np.mean(signal[:-100])) / 2
        delay = np.where(signal > th)[0][0]
        ds_fit[f"delay{i + 1}"] = int(np.round(delay / 4) * 4)  # Find the closest multiple integer of 4ns

    return ds_fit
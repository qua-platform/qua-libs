import numpy as np
from matplotlib import pyplot as plt
from qualang_tools.units import unit


u = unit(coerce_to_integer=True)


def plot_single_run_with_fit(num_resonators, data: dict):
    """
    Plot single-shot I/Q traces for each qubit with threshold and delay markers.

    Each subplot displays:
        - The single-run I (blue) and Q (red) quadrature signals.
        - Horizontal dashed lines at +0.5 and -0.5 as visual thresholds.
        - Dashed lines representing the mean offset of I and Q.
        - A vertical dashed line marking the extracted delay.

    Args:
        num_resonators (int): Number of resonators (or qubits) to plot.
        data (dict): Dictionary containing processed signal data, including:
            - 'adc_single_runI{i}'
            - 'adc_single_runQ{i}'
            - 'mean_values'
            - 'delay{i}'

    Returns:
        matplotlib.figure.Figure: The figure object containing the subplots.
    """
    fig_single_run, axs1 = plt.subplots(num_resonators, 1, figsize=(10, 4 * num_resonators))

    for i in range(num_resonators):
        q = i + 1
        adcI = data['raw_data'][f"adc_single_runI{q}"]
        adcQ = data['raw_data'][f"adc_single_runQ{q}"]
        adcI_mean = data["mean_values"][f"adcI{q}"]
        adcQ_mean = data["mean_values"][f"adcQ{q}"]
        delay = data[f"delay{q}"]

        ax = axs1[i] if num_resonators > 1 else axs1
        ax.set_title(f"Resonator {q} - Single Run")
        ax.plot(adcI, "b", label="I")
        ax.plot(adcQ, "r", label="Q")
        ax.axhline(0.5, color="gray", linestyle="-")
        ax.axhline(-0.5, color="gray", linestyle="-")
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        ax.plot(xl, adcI_mean * np.ones(2), "k--")
        ax.plot(xl, adcQ_mean * np.ones(2), "k--")
        ax.plot([delay, delay], yl, "k--")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Signal amplitude [V]")
        ax.legend()
        ax.grid(True)

    fig_single_run.tight_layout()

    return fig_single_run


def plot_averaged_run_with_fit(num_resonators, data: dict):
    """
    Plot averaged I/Q traces for each qubit with mean lines and delay markers.

    Each subplot displays:
        - Averaged I (blue) and Q (red) readout signals.
        - Dashed lines at their respective means (I and Q).
        - A vertical dashed line at the detected signal delay.

    Args:
        num_resonators (int): Number of resonators (or qubits) to plot.
        data (dict): Dictionary containing averaged signal data, including:
            - 'adcI{i}'
            - 'adcQ{i}'
            - 'mean_values'
            - 'delay{i}'

    Returns:
        matplotlib.figure.Figure: The figure object containing the subplots.
    """
    fig_averaged_run, axs2 = plt.subplots(num_resonators, 1, figsize=(10, 4 * num_resonators))

    for i in range(num_resonators):
        q = i + 1
        adcI = data['raw_data'][f"adcI{q}"]
        adcQ = data['raw_data'][f"adcQ{q}"]
        adcI_mean = data["mean_values"][f"adcI{q}"]
        adcQ_mean = data["mean_values"][f"adcQ{q}"]
        delay = data[f"delay{q}"]

        ax = axs2[i] if num_resonators > 1 else axs2
        ax.set_title(f"Resonator {q} - Averaged Run")
        ax.plot(adcI, "b", label="I")
        ax.plot(adcQ, "r", label="Q")
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        ax.plot(xl, adcI_mean * np.ones(2), "k--")
        ax.plot(xl, adcQ_mean * np.ones(2), "k--")
        ax.plot([delay, delay], yl, "k--")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Signal amplitude [V]")
        ax.legend()
        ax.grid(True)

    fig_averaged_run.tight_layout()

    return fig_averaged_run

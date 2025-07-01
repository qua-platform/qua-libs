import numpy as np
from matplotlib import pyplot as plt
from qualang_tools.units import unit


u = unit(coerce_to_integer=True)


def plot_single_run_with_fit(data: dict, num_resonators: int):
    """
    Plot single-shot I/Q traces for each qubit with threshold and delay markers.

    Each subplot displays:
        - The single-run I (blue) and Q (red) quadrature signals.
        - Horizontal dashed lines at +0.5 and -0.5 as visual thresholds.
        - Dashed lines representing the mean offset of I and Q.
        - A vertical dashed line marking the extracted delay.

    Args:
        data (dict): Dictionary containing processed signal data and fitted data, including:
            - data["processed_data"]["adc_single_runI{i}"]: Single-run I trace for resonator i.
            - data["processed_data"]["adc_single_runQ{i}"]: Single-run Q trace for resonator i.
            - data["fitted_data"]["mean_values"]: Dict with mean offsets for I and Q.
            - data["fitted_data"]["delay{i}"]: Estimated arrival delay for resonator i (in ns).

        num_resonators (int): Number of resonators (or qubits) to plot.

    Returns:
        matplotlib.figure.Figure: The figure object with subplots for each resonator.
    """
    fig_single_run, axs1 = plt.subplots(num_resonators, 1, figsize=(10, 4 * num_resonators))

    for i in range(num_resonators):
        q = i + 1
        I = data["processed_data"][f"adc_single_runI{q}"]
        Q = data["processed_data"][f"adc_single_runQ{q}"]
        I_mean = data["fitted_data"]["mean_values"][f"adcI{q}"]
        Q_mean = data["fitted_data"]["mean_values"][f"adcQ{q}"]
        delay = data["fitted_data"][f"delay{q}"]

        ax = axs1[i] if num_resonators > 1 else axs1
        ax.set_title(f"Resonator {q} - Single Run")
        ax.plot(I, "b", label="I")
        ax.plot(Q, "r", label="Q")
        ax.axhline(0.5, color="gray", linestyle="-")
        ax.axhline(-0.5, color="gray", linestyle="-")
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        ax.plot(xl, I_mean * np.ones(2), "k--")
        ax.plot(xl, Q_mean * np.ones(2), "k--")
        ax.plot([delay, delay], yl, "k--")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Signal amplitude [V]")
        ax.legend()
        ax.grid(True)

    fig_single_run.tight_layout()

    return fig_single_run


def plot_averaged_run_with_fit(data: dict, num_resonators: int):
    """
    Plot averaged I/Q traces for each qubit with mean lines and delay markers.

    Each subplot displays:
        - Averaged I (blue) and Q (red) readout signals.
        - Dashed lines at their respective means (I and Q).
        - A vertical dashed line at the detected signal delay.

    Args:
        data (dict): Dictionary containing processed signal data and fitted data, including:
            - data["processed_data"]["adc_single_runI{i}"]: Single-run I trace for resonator i.
            - data["processed_data"]["adc_single_runQ{i}"]: Single-run Q trace for resonator i.
            - data["fitted_data"]["mean_values"]: Dict with mean offsets for I and Q.
            - data["fitted_data"]["delay{i}"]: Estimated arrival delay for resonator i (in ns).
        num_resonators (int): Number of resonators (or qubits) to plot.

    Returns:
        matplotlib.figure.Figure: The figure object with subplots for each resonator.
    """
    fig_averaged_run, axs2 = plt.subplots(num_resonators, 1, figsize=(10, 4 * num_resonators))

    for i in range(num_resonators):
        q = i + 1
        I = data["processed_data"][f"adc_single_runI{q}"]
        Q = data["processed_data"][f"adc_single_runQ{q}"]
        I_mean = data["fitted_data"]["mean_values"][f"adcI{q}"]
        Q_mean = data["fitted_data"]["mean_values"][f"adcQ{q}"]
        delay = data["fitted_data"][f"delay{q}"]

        ax = axs2[i] if num_resonators > 1 else axs2
        ax.set_title(f"Resonator {q} - Averaged Run")
        ax.plot(I, "b", label="I")
        ax.plot(Q, "r", label="Q")
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        ax.plot(xl, I_mean * np.ones(2), "k--")
        ax.plot(xl, Q_mean * np.ones(2), "k--")
        ax.plot([delay, delay], yl, "k--")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Signal amplitude [V]")
        ax.legend()
        ax.grid(True)

    fig_averaged_run.tight_layout()

    return fig_averaged_run

"""
CHARGE STABILITY DIAGRAM
"""

from qm.qua import *
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from typing import Union
from numpy.typing import NDArray
import numpy as np


def round_to_fixed(x, number_of_bits=12):
    """
    function which rounds 'x' to 'number_of_bits' of precision to help reduce the accumulation of fixed point arithmetic errors
    """
    return round((2**number_of_bits) * x) / (2**number_of_bits)


def RF_reflectometry_macro(
    operation: str = "readout",
    element: str = "tank_circuit",
    element_output: str = "out1",
    I=None,
    Q=None,
    I_st=None,
    Q_st=None,
):
    if I is None:
        I = declare(fixed)
    if Q is None:
        Q = declare(fixed)
    if I_st is None:
        I_st = declare_stream()
    if Q_st is None:
        Q_st = declare_stream()
    measure(operation, element, None, demod.full("cos", I, element_output), demod.full("sin", Q, element_output))
    save(I, I_st)
    save(Q, Q_st)
    return I, Q, I_st, Q_st


def DC_current_sensing_macro(
    operation: str = "readout", element: str = "TIA", element_output: str = "out2", dc_signal=None, dc_signal_st=None
):
    if dc_signal is None:
        dc_signal = declare(fixed)
    if dc_signal_st is None:
        dc_signal_st = declare_stream()
    measure(operation, element, None, integration.full("constant", dc_signal, element_output))
    save(dc_signal, dc_signal_st)
    return dc_signal, dc_signal_st


def get_filtered_voltage(
    voltage_list: Union[NDArray, list], step_duration: float, bias_tee_cut_off_frequency: float, plot: bool = False
):
    """Get the voltage after filtering through the bias-tee

    :param voltage_list: List of voltages outputted by the OPX in V.
    :param step_duration: Duration of each step in s.
    :param bias_tee_cut_off_frequency: Cut-off frequency of the bias-tee in Hz.
    :param plot: Flag to plot the voltage values if set to True.
    :return: the filtered and unfiltered voltage lists with 1Gs/s sampling rate.
    """

    def high_pass(data, f_cutoff):
        res = butter(1, f_cutoff, btype="high", analog=False)
        return lfilter(res[0], res[1], data)

    y = [val for val in voltage_list for _ in range(int(step_duration * 1e9))]
    y_filtered = high_pass(y, bias_tee_cut_off_frequency * 1e-9)
    if plot:
        # plt.figure()
        plt.plot(y, label="before bias-tee")
        plt.plot(y_filtered, label="after bias-tee")
        plt.xlabel("Time [ns]")
        plt.ylabel("Voltage [V]")
        plt.legend()
    print(f"Error: {np.mean(np.abs((y-y_filtered)/(max(y)-min(y))))*100:.2f} %")
    return y, y_filtered

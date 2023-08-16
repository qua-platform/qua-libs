import numpy as np


# Exponential decay
def expdecay(x, a, t):
    """Exponential decay defined as 1 + a * np.exp(-x / t).

    :param x: numpy array for the time vector in ns
    :param a: float for the exponential amplitude
    :param t: float for the exponential decay time in ns
    :return: numpy array for the exponential decay
    """
    return 1 + a * np.exp(-x / t)


# Theoretical IIR and FIR taps based on exponential decay coefficients
def exponential_correction(A, tau, Ts=1e-9):
    """Derive FIR and IIR filter taps based on a the exponential coefficients A and tau from 1 + a * np.exp(-x / t).

    :param A: amplitude of the exponential decay
    :param tau: decay time of the exponential decay
    :param Ts: sampling period. Default is 1e-9
    :return: FIR and IIR taps
    """
    tau = tau * Ts
    k1 = Ts + 2 * tau * (A + 1)
    k2 = Ts - 2 * tau * (A + 1)
    c1 = Ts + 2 * tau
    c2 = Ts - 2 * tau
    feedback_tap = k2 / k1
    feedforward_taps = np.array([c1, c2]) / k1
    return feedforward_taps, feedback_tap


# FIR and IIR taps calculation
def filter_calc(exponential):
    """Derive FIR and IIR filter taps based on a list of exponential coefficients.

    :param exponential: exponential coefficients defined as [(A1, tau1), (A2, tau2)]
    :return: FIR and IIR taps as [fir], [iir]
    """
    # Initialization based on the number of exponential coefficients
    b = np.zeros((2, len(exponential)))
    feedback_taps = np.zeros(len(exponential))
    # Derive feedback tap for each set of exponential coefficients
    for i, (A, tau) in enumerate(exponential):
        b[:, i], feedback_taps[i] = exponential_correction(A, tau)
    # Derive feddback tap for each set of exponential coefficients
    feedforward_taps = b[:, 0]
    for i in range(len(exponential) - 1):
        feedforward_taps = np.convolve(feedforward_taps, b[:, i + 1])
    # feedforward taps are bounded to +/- 2
    if np.abs(max(feedforward_taps)) >= 2:
        feedforward_taps = 2 * feedforward_taps / max(feedforward_taps)

    return feedforward_taps, feedback_taps

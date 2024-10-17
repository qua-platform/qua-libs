"""
This file contains useful QUA macros meant to simplify and ease QUA programs.
All the macros below have been written and tested with the basic configuration. If you modify this configuration
(elements, operations, integration weights...) these macros will need to be modified accordingly.
"""

from qm.qua import *
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.loops import from_array


##############
# QUA macros #
##############


def reset_qubit(method, **kwargs):
    """
    Macro to reset the qubit state.

    If method is 'cooldown', then the variable cooldown_time (in clock cycles) must be provided as a python integer > 4.

    **Example**: reset_qubit('cooldown', cooldown_times=500)

    If method is 'active', then 3 parameters are available as listed below.

    **Example**: reset_qubit('active', threshold=-0.003, max_tries=3)

    :param method: Method the reset the qubit state. Can be either 'cooldown' or 'active'.
    :type method: str
    :key cooldown_time: qubit relaxation time in clock cycle, needed if method is 'cooldown'. Must be an integer > 4.
    :key threshold: threshold to discriminate between the ground and excited state, needed if method is 'active'.
    :key max_tries: python integer for the maximum number of tries used to perform active reset,
        needed if method is 'active'. Must be an integer > 0 and default value is 1.
    :key Ig: A QUA variable for the information in the `I` quadrature used for active reset. If not given, a new
        variable will be created. Must be of type `Fixed`.
    :return:
    """
    if method == "cooldown":
        # Check cooldown_time
        cooldown_time = kwargs.get("cooldown_time", None)
        if (cooldown_time is None) or (cooldown_time < 4):
            raise Exception("'cooldown_time' must be an integer > 4 clock cycles")
        # Reset qubit state
        wait(cooldown_time, "qubit")
    elif method == "active":
        # Check threshold
        threshold = kwargs.get("threshold", None)
        if threshold is None:
            raise Exception("'threshold' must be specified for active reset.")
        # Check max_tries
        max_tries = kwargs.get("max_tries", 1)
        if (max_tries is None) or (not float(max_tries).is_integer()) or (max_tries < 1):
            raise Exception("'max_tries' must be an integer > 0.")
        # Check Ig
        Ig = kwargs.get("Ig", None)
        # Reset qubit state
        return active_reset(threshold, max_tries=max_tries, Ig=Ig)


# Macro for performing active reset until successful for a given number of tries.
def active_reset(threshold, max_tries=1, Ig=None):
    """Macro for performing active reset until successful for a given number of tries.

    :param threshold: threshold for the 'I' quadrature discriminating between ground and excited state.
    :param max_tries: python integer for the maximum number of tries used to perform active reset. Must >= 1.
    :param Ig: A QUA variable for the information in the `I` quadrature. Should be of type `Fixed`. If not given, a new
        variable will be created
    :return: A QUA variable for the information in the `I` quadrature and the number of tries after success.
    """
    if Ig is None:
        Ig = declare(fixed)
    if (max_tries < 1) or (not float(max_tries).is_integer()):
        raise Exception("max_count must be an integer >= 1.")
    # Initialize Ig to be > threshold
    assign(Ig, threshold + 2**-28)
    # Number of tries for active reset
    counter = declare(int)
    # Reset the number of tries
    assign(counter, 0)

    # Perform active feedback
    align("qubit", "resonator")
    # Use a while loop and counter for other protocols and tests
    with while_((Ig > threshold) & (counter < max_tries)):
        # Measure the resonator
        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full("rotated_cos", "rotated_sin", Ig),
        )
        # Play a pi pulse to get back to the ground state
        play("pi", "qubit", condition=(Ig > threshold))
        # Increment the number of tries
        assign(counter, counter + 1)
    return Ig, counter


# Single shot readout macro
def readout_macro(threshold=None, state=None, I=None, Q=None):
    """
    A macro for performing the readout, with the ability to perform state discrimination.
    If `threshold` is given, the information in the `I` quadrature will be compared against the threshold and `state`
    would be `True` if `I > threshold`.
    Note that it is assumed that the results are rotated such that all the information is in the `I` quadrature.

    :param threshold: Optional. The threshold to compare `I` against.
    :param state: A QUA variable for the state information, only used when a threshold is given.
        Should be of type `bool`. If not given, a new variable will be created
    :param I: A QUA variable for the information in the `I` quadrature. Should be of type `Fixed`. If not given, a new
        variable will be created
    :param Q: A QUA variable for the information in the `Q` quadrature. Should be of type `Fixed`. If not given, a new
        variable will be created
    :return: Three QUA variables populated with the results of the readout: (`state`, `I`, `Q`)
    """
    if I is None:
        I = declare(fixed)
    if Q is None:
        Q = declare(fixed)
    if threshold is not None and state is None:
        state = declare(bool)
    measure(
        "readout",
        "resonator",
        None,
        dual_demod.full("rotated_cos", "rotated_sin", I),
        dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
    )
    if threshold is not None:
        assign(state, I > threshold)
    return state, I, Q


# Frequency tracking class
class qubit_frequency_tracking:
    def __init__(self, qubit, rr, f_res, ge_threshold, frame_rotation_flag=False):
        """Frequency tracking class

        :param str qubit: The qubit element from the configuration
        :param str rr: The readout element from the configuration
        :param int f_res: The initial guess for the qubit resonance frequency in Hz
        :param float ge_threshold: Threshold to discriminate between ground and excited (with single shot readout)
        :param bool frame_rotation_flag: Flag to perform the Ramsey scans by dephasing the 2nd pi/2 pulse instead of applying a detuning.
        """
        # The qubit element
        self.qubit = qubit
        # The readout resonator element
        self.rr = rr
        # The qubit resonance frequency
        self.f_res = f_res
        # Threshold to discriminate between ground and excited (with single shot readout)
        self.ge_threshold = ge_threshold
        # Ramsey dephasing (idle) time in clock cycles (4ns)
        self.dephasing_time = None
        # Dephasing time vector for time domain Ramsey
        self.tau_vec = None
        # Detuning to apply for time domain Ramsey
        self.f_det = None
        # Qubit detuning vector for frequency domain Ramsey
        self.f_vec = None
        # HWHM of the frequency domain Ramsey central fringe around the qubit resonance
        self.delta = None
        # Fitted amplitude of the frequency domain oscillations used to derive the scale factor in two_point_ramsey
        self.frequency_sweep_amp = None
        # Flag to perform the Ramsey scans by dephasing the second pi/2 pulse instead of applying a detuning
        self.frame_rotation = frame_rotation_flag
        # Flag to declare the QUA variable and initialize state_estimation_st_idx during the first run
        self.init = True

    def _qua_declaration(self):
        # I & Q data
        self.I = declare(fixed)
        self.Q = declare(fixed)
        # Qubit state after a measurement (True or False)
        self.res = declare(bool)
        # Qubit state after a measurement (0.0 or 1.0)
        self.state_estimation = declare(fixed)
        # Stream for state_estimation as a buffer of streams if multiple sweeps are performed in the same program
        self.state_estimation_st = [declare_stream() for i in range(10)]
        # Initialize the index for the buffer of state_estimation streams (python variable)
        self.state_estimation_st_idx = 0
        # Variable for averaging
        self.n = declare(int)
        # Variable for scanning the dephasing time
        self.tau = declare(int)
        # Variable for scanning the qubit detuning
        self.f = declare(int)
        # Vector containing the data for the two_point_ramsey
        self.two_point_vec = declare(fixed, size=2)
        # Variable to switch from the left to the right side of the fringe in two_point_ramsey
        self.idx = declare(int)
        # Frequency correction to apply in order to track the qubit resonance
        self.corr = declare(int, value=0)
        # Stream for corr
        self.corr_st = declare_stream()
        # Qubit frequency after correction with two_point_ramsey
        self.f_res_corr = declare(int, value=round(self.f_res))
        # Stream for f_res_corr
        self.f_res_corr_st = declare_stream()
        # Detuning used to derive the phase of the second pi/2 pulse when using frame rotation
        self.frame_rotation_detuning = declare(fixed)
        # Conversion factor from GHz to Hz
        self.Hz_to_GHz = declare(fixed, value=1e-9)

    def initialization(self):
        self._qua_declaration()

    @staticmethod
    def _fit_ramsey(x, y):
        w = np.fft.fft(y)
        freq = np.fft.fftfreq(len(x))
        new_w = w[1 : len(freq // 2)]
        new_f = freq[1 : len(freq // 2)]

        ind = new_f > 0
        new_f = new_f[ind]
        new_w = new_w[ind]

        yy = np.abs(new_w)
        first_read_data_ind = np.where(yy[1:] - yy[:-1] > 0)[0][0]  # away from the DC peak

        new_f = new_f[first_read_data_ind:]
        new_w = new_w[first_read_data_ind:]

        out_freq = new_f[np.argmax(np.abs(new_w))]
        new_w_arg = new_w[np.argmax(np.abs(new_w))]

        omega = out_freq * 2 * np.pi / (x[1] - x[0])  # get gauss for frequency #here

        cycle = int(np.ceil(1 / out_freq))
        peaks = np.array([np.std(y[i * cycle : (i + 1) * cycle]) for i in range(int(len(y) / cycle))]) * np.sqrt(2) * 2

        initial_offset = np.mean(y[:cycle])
        cycles_wait = np.where(peaks > peaks[0] * 0.37)[0][-1]

        post_decay_mean = np.mean(y[-cycle:])

        decay_gauss = (
            np.log(peaks[0] / peaks[cycles_wait]) / (cycles_wait * cycle) / (x[1] - x[0])
        )  # get gauss for decay #here

        fit_type = lambda x, a: post_decay_mean * a[4] * (1 - np.exp(-x * decay_gauss * a[1])) + peaks[0] / 2 * a[2] * (
            np.exp(-x * decay_gauss * a[1])
            * (a[5] * initial_offset / peaks[0] * 2 + np.cos(2 * np.pi * a[0] * omega / (2 * np.pi) * x + a[3]))
        )  # here problem, removed the 1+

        def curve_fit3(f, x, y, a0):
            def opt(x, y, a):
                return np.sum(np.abs(f(x, a) - y) ** 2)

            out = optimize.minimize(lambda a: opt(x, y, a), a0)
            return out["x"]

        angle0 = np.angle(new_w_arg) - omega * x[0]

        popt = curve_fit3(
            fit_type,
            x,
            y,
            [1, 1, 1, angle0, 1, 1, 1],
        )

        print(
            f"f = {popt[0] * omega / (2 * np.pi)}, phase = {popt[3] % (2 * np.pi)}, tau = {1 / (decay_gauss * popt[1])}, amp = {peaks[0] * popt[2]}, uncertainty population = {post_decay_mean * popt[4]},initial offset = {popt[5] * initial_offset}"
        )
        out = {
            "fit_func": lambda x: fit_type(x, popt),
            "f": popt[0] * omega / (2 * np.pi),
            "phase": popt[3] % (2 * np.pi),
            "tau": 1 / (decay_gauss * popt[1]),
            "amp": peaks[0] * popt[2],
            "uncertainty_population": post_decay_mean * popt[4],
            "initial_offset": popt[5] * initial_offset,
        }

        plt.plot(x, fit_type(x, [1, 1, 1, angle0, 1, 1, 1]), "--r", linewidth=1, label="Fit initial guess")
        return out

    def time_domain_ramsey_full_sweep(self, n_avg, f_det, tau_vec, correct=False):
        """QUA program to perform a time-domain Ramsey sequence with `n_avg` averages and scanning the idle time over `tau_vec`.

        :param int n_avg: python integer for the number of averaging loops
        :param int f_det: python integer for the detuning to apply in Hz
        :param tau_vec: numpy array of integers for the idle times to be scanned in clock cycles (4ns)
        :param bool correct: boolean flag for choosing to use the initial qubit frequency or the corrected one
        :return: None
        """
        # Declare the QUA variables once
        if self.init:
            self._qua_declaration()
            self.init = False

        self.f_det = f_det
        self.tau_vec = tau_vec
        if self.frame_rotation:
            if correct:
                update_frequency(self.qubit, self.f_res_corr)
            else:
                update_frequency(self.qubit, self.f_res)
        else:
            if correct:
                update_frequency(self.qubit, self.f_res_corr + self.f_det)
            else:
                update_frequency(self.qubit, self.f_res + self.f_det)

        with for_(self.n, 0, self.n < n_avg, self.n + 1):
            with for_(*from_array(self.tau, tau_vec)):
                # Qubit initialization
                reset_qubit("cooldown", cooldown_time=1000)
                # Ramsey sequence (time-domain)
                play("x90", self.qubit)
                wait(self.tau, self.qubit)
                # Perform Time domain Ramsey with a frame rotation instead of detuning
                # 4*tau because tau was in clock cycles and 1e-9 because tau is ns
                if self.frame_rotation:
                    frame_rotation_2pi(Cast.mul_fixed_by_int(self.f_det * 1e-9, 4 * self.tau), self.qubit)
                play("x90", self.qubit)

                align(self.qubit, self.rr)
                # should be replaced by the readout procedure of the qubit. A boolean value should be assigned into
                # the QUA variable "self.res". True for the qubit in the excited.
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("rotated_cos", "rotated_sin", self.I),
                )
                assign(self.res, self.I > self.ge_threshold)
                ####################################################################################################
                # Convert bool to fixed to perform the average
                assign(self.state_estimation, Cast.to_fixed(self.res))
                save(
                    self.state_estimation,
                    self.state_estimation_st[self.state_estimation_st_idx],
                )

        self.state_estimation_st_idx += 1

    def time_domain_ramsey_full_sweep_analysis(self, result_handles, stream_name):
        # Get the average excited population
        Pe = result_handles.get(stream_name).fetch_all()
        # Get the idle time vector in ns
        t = np.array(self.tau_vec) * 4
        # Plot raw data
        plt.plot(t, Pe, ".", label="Experimental data")
        # Fit data
        out = qubit_frequency_tracking._fit_ramsey(t, Pe)  # in [ns]
        # Plot fit
        plt.plot(t, out["fit_func"](t), "m", label="Fit")
        plt.xlabel("time[ns]")
        plt.ylabel("P(|e>)")
        # New intermediate frequency: f_res - (fitted_detuning - f_det)
        self.f_res = self.f_res - int(out["f"] * 1e9 - self.f_det)
        print(f"shifting by {out['f'] * 1e9 - self.f_det:.0f} Hz, and now f_res = {self.f_res} Hz")

        # Dephasing time leading to a phase-shift of 2*pi for a frequency detuning f_det
        tau_2pi = int(1 / self.f_det / 4e-9)
        plt.plot(
            tau_2pi * 4,
            out["fit_func"](tau_2pi * 4),
            "r*",
            label="Ideal first peak location",
        )
        plt.legend()

    def freq_domain_ramsey_full_sweep(self, n_avg, f_vec, oscillation_number=1):
        """QUA program to perform a frequency-domain Ramsey sequence with `n_avg` averages and scanning the frequency over `f_vec`.

        :param int n_avg: python integer for the number of averaging loops
        :param f_vec: numpy array of integers for the qubit detuning to be scanned in Hz
        :param oscillation_number: number of oscillations to capture used to define the idle time.
        :return:
        """

        # Declare the QUA variables once
        if self.init:
            self._qua_declaration()
            self.init = False
        self.f_vec = f_vec
        # Dephasing time to get a given number of oscillations in the frequency range given by f_vec
        self.dephasing_time = max(oscillation_number * int(1 / (2 * (max(f_vec) - self.f_res)) / 4e-9), 4)

        with for_(self.n, 0, self.n < n_avg, self.n + 1):
            with for_(*from_array(self.f, f_vec)):
                # Qubit initialization
                # Note: if you are using active reset, you might want to do it with the new corrected frequency
                reset_qubit("cooldown", cooldown_time=1000)
                # Update the frequency
                if self.frame_rotation:
                    update_frequency(self.qubit, self.f_res)
                else:
                    update_frequency(self.qubit, self.f)
                # Ramsey sequence
                play("x90", self.qubit)

                if self.frame_rotation:
                    assign(self.frame_rotation_detuning, Cast.mul_fixed_by_int(self.Hz_to_GHz, self.f - self.f_res))
                    frame_rotation_2pi(
                        Cast.mul_fixed_by_int(self.frame_rotation_detuning, 4 * self.dephasing_time), self.qubit
                    )
                wait(self.dephasing_time, self.qubit)
                play("x90", self.qubit)

                align(self.qubit, self.rr)
                # should be replaced by the readout procedure of the qubit. A boolean value should be assigned into
                # the QUA variable "self.res". True for the qubit in the excited.
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "sin", self.I),
                )
                if self.frame_rotation:
                    reset_frame(self.qubit)
                assign(self.res, self.I > self.ge_threshold)
                ####################################################################################################
                # Convert bool to fixed to perform the average
                assign(self.state_estimation, Cast.to_fixed(self.res))
                save(self.state_estimation, self.state_estimation_st[self.state_estimation_st_idx])
        # Increment state_estimation_st_idx in case other full sweeps are performed within the same program.
        self.state_estimation_st_idx += 1

    def freq_domain_ramsey_full_sweep_analysis(self, result_handles, stream_name):
        # Get the average excited population
        Pe = result_handles.get(stream_name).fetch_all()
        # Plot raw data
        plt.plot(self.f_vec - self.f_res, Pe, ".", label="Experimental data")
        # Fit data
        out = qubit_frequency_tracking._fit_ramsey(self.f_vec - self.f_res, Pe)
        # amplitude of the frequency domain oscillations used to derive the scale factor in two_point_ramsey
        self.frequency_sweep_amp = out["amp"]
        # HWHM of the frequency domain Ramsey central fringe around the qubit resonance
        # i.e. detuning to go from resonance to  half fringe
        self.delta = int(1 / (self.dephasing_time * 4e-9) / 4)  # the last 4 is for 1/4 of a cycle (dephasing of pi/2)
        # Plot fit
        plt.plot(self.f_vec - self.f_res, out["fit_func"](self.f_vec - self.f_res), "m", label="fit")
        # Plot specific points at half the central fringe
        plt.plot(
            [-self.delta, self.delta],
            out["fit_func"](np.array([-self.delta, self.delta])),
            "r*",
        )
        plt.xlabel("Detuning from resonance [Hz]")
        plt.ylabel("P(|e>)")
        plt.legend()

    def two_points_ramsey(self, n_avg_power_of_2):
        """
        Sequence consisting of measuring successively the left and right sides of the Ramsey central fringe around
        resonance to track the qubit frequency drifts.

        :param int n_avg_power_of_2: power of two defining the number of averages as n_avg=2**n_avg_power_of_2
        :return:
        """
        if n_avg_power_of_2 > 20 or not np.log2(2**n_avg_power_of_2).is_integer():
            raise ValueError(
                "'n_avg_power_of_2' must be defined as the power of two defining the number of averages (n_avg=2**n_avg_power_of_2)"
            )
        # Declare the QUA variables once
        if self.init:
            self._qua_declaration()
            self.init = False

        # Scale factor to convert amplitude to frequency change: frequency_sweep_amp is the amplitude of the frequency
        # domain oscillation. The factor 4e-9 is to convert tau from clock cycles to sec.
        scale_factor = int(
            1 / (2 * np.pi * self.dephasing_time * 4e-9 * self.frequency_sweep_amp)
        )  # in Hz per unit of I, Q or state
        # Average value of the measured quantity (I, state, np.sqrt(I**2+Q**2)...) on both sides of the central fringe.
        assign(self.two_point_vec[0], 0)  # Left side
        assign(self.two_point_vec[1], 0)  # Right side
        # Number of averages defined as a power of 2 to perform the average on the FPGA using bit-shifts.
        with for_(self.n, 0, self.n < 2**n_avg_power_of_2, self.n + 1):
            # Go to the left side of the central fringe
            assign(self.f, self.f_res_corr - self.delta)
            # Alternate between left and right sides
            with for_(self.idx, 0, self.idx < 2, self.idx + 1):
                # Qubit initialization
                # Note: if you are using active reset, you might want to do it with the new corrected frequency
                reset_qubit("cooldown", cooldown_time=1000)
                ####################################################################################################
                # Set qubit frequency
                if self.frame_rotation:
                    update_frequency(self.qubit, self.f_res_corr)
                else:
                    update_frequency(self.qubit, self.f)
                # Ramsey sequence
                play("x90", self.qubit)
                wait(self.dephasing_time, self.qubit)
                if self.frame_rotation:
                    assign(
                        self.frame_rotation_detuning,
                        Cast.mul_fixed_by_int(self.Hz_to_GHz, self.f - self.f_res_corr),
                    )
                    frame_rotation_2pi(
                        Cast.mul_fixed_by_int(self.frame_rotation_detuning, 4 * self.dephasing_time), self.qubit
                    )
                play("x90", self.qubit)

                align(self.qubit, self.rr)
                # should be replaced by the readout procedure of the qubit. A boolean value should be assigned into
                # the QUA variable "self.res". True for the qubit in the excited.

                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "sin", self.I),
                )
                if self.frame_rotation:
                    reset_frame(self.qubit)
                assign(self.res, self.I > self.ge_threshold)
                ####################################################################################################
                # Sum the results and divide by the number of iterations to get the average on the fly
                assign(
                    self.two_point_vec[self.idx],
                    self.two_point_vec[self.idx] + (Cast.to_fixed(self.res) >> n_avg_power_of_2),
                )
                # Go to the right side of the central fringe
                assign(self.f, self.f + 2 * self.delta)

        # Derive the frequency shift
        assign(self.corr, Cast.mul_int_by_fixed(scale_factor, (self.two_point_vec[0] - self.two_point_vec[1])))
        # To keep track of the qubit frequency over time
        assign(self.f_res_corr, self.f_res_corr - self.corr)

        save(self.f_res_corr, self.f_res_corr_st)
        save(self.corr, self.corr_st)

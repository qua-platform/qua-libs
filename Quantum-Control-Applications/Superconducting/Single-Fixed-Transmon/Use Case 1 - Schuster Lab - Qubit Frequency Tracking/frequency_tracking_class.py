from qm import QuantumMachinesManager
from qm.qua import *
from qm.simulate import SimulationConfig


import numpy as np
import matplotlib.pyplot as plt
from configuration import config, ge_IF, qubit_freq, disc_file_jpa
from TwoStateDiscriminator_2103 import TwoStateDiscriminator

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, "rr", disc_file_jpa, lsb=True)


class qubit_frequency_tracking:
    def __init__(self, qubit, rr, f_res):
        self.qubit = qubit
        self.rr = rr
        self.fres = f_res
        self.t2 = None
        self.tau0 = None
        self.phase = None
        self.tau_vec = None
        self.f_ref = None
        self.fvec = None
        self.delta = None
        self.frequency_sweep_amp = None

    def _fit_ramsey(self, x, y):
        # y = Pe
        # taus = self.tau_vec

        from scipy import optimize

        # taus = np.arange(0, 400, 1) + 0
        #         # y = 0.4 * (0.4 * 2 * (1 - np.exp(-taus / 100)) + np.exp(-taus / 100) * (
        #         #             1 + np.cos(1.8 + 2 * np.pi * (1 / 10) * taus))) + 0.0 * (np.random.rand(len(taus)) - 0.5)
        #
        # taus = np.array(self.tau_vec)
        # y = Pe
        #
        # plt.clf()
        # plt.plot(taus, y, label="MicroSec")
        # plt.legend()
        #
        # ax = plt.gca()
        # x = ax.get_lines()[0].get_xdata()  # get the X, Y data from graph - can skip this
        # y = ax.get_lines()[0].get_ydata()

        w = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(x))
        new_w = w[1 : len(freqs // 2)]
        new_f = freqs[1 : len(freqs // 2)]

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

        # amp_guess = max(y[0:cycle])-min(y[0:cycle]) #get gauss for amplitude
        # post_decay = max(y[cycle * cycles_wait : cycle * (cycles_wait+1)])-min(y[cycle * cycles_wait : cycle * (cycles_wait+1)])
        post_decay_mean = np.mean(y[-cycle:])

        # decay_gauss = np.log(amp_guess/post_decay)/(cycles_wait*cycle) / (x[1]-x[0]) #get gauss for decay #here
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

        angle0 = np.angle(new_w_arg) - omega * x[0]  # here

        popt = curve_fit3(
            fit_type,
            x,
            y,
            [1, 1, 1, angle0, 1, 1, 1],
        )

        print(
            f"f = {popt[0]*omega/(2*np.pi)}, phase = {popt[3] % (2*np.pi)}, tau = {1/(decay_gauss*popt[1])}, amp = {peaks[0]*popt[2]}, uncertainty population = {post_decay_mean*popt[4]},initial offset = {popt[5] * initial_offset}"
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

        # (l,) = plt.plot(x, fit_func(x), "m", linewidth=2)
        plt.plot(x, fit_type(x, [1, 1, 1, angle0, 1, 1, 1]), "--r", linewidth=1)
        return out

    def qua_declarations(self):
        self.I = declare(fixed)
        self.Q = declare(fixed)
        self.state_estimation = declare(fixed)
        self.state_estimation_st = [declare_stream() for i in range(10)]
        self.state_estimation_st_idx = 0

        self.res = declare(bool)

        self.n = declare(int)
        self.tau = declare(int)

        self.m = declare(int)
        self.f = declare(int)

        self.p = declare(int)
        self.if_total = declare(int, value=0)
        self.se_vec = declare(fixed, size=3)
        self.idx = declare(int)
        self.fres_corr = declare(int, value=int(self.fres + 0.5))
        self.fres_corr_st = declare_stream()
        self.corr = declare(int, value=0)
        self.corr_st = declare_stream()

    def time_domain_ramesy_full_sweep(self, reps, f_ref, tau_min, tau_max, dtau, stream_name, correct=False):
        self.f_ref = f_ref
        self.tau_vec = np.arange(tau_min, tau_max, dtau).astype(int).tolist()
        print(f"fres is {self.fres}, playing {self.fres + self.f_ref}")
        if correct:
            update_frequency(self.qubit, self.fres + self.f_ref - self.corr)
        else:
            update_frequency(self.qubit, self.fres + self.f_ref)
        with for_(self.n, 0, self.n < reps, self.n + 1):
            with for_(self.tau, tau_min, self.tau < tau_max, self.tau + dtau):
                play("pump_square", "jpa_pump")
                discriminator.measure_state("readout", "out1", "out2", self.res)
                align(self.qubit, self.rr, "jpa_pump")
                wait(500, self.qubit)
                play("pi", self.qubit, condition=self.res)
                wait(500, self.qubit)

                play("pi2", self.qubit)
                wait(self.tau, self.qubit)
                play("pi2", self.qubit)

                align(self.qubit, self.rr, "jpa_pump")
                play("pump_square", "jpa_pump")
                discriminator.measure_state("readout", "out1", "out2", self.res)

                assign(self.state_estimation, Cast.to_fixed(self.res))  # True: Excited
                save(
                    self.state_estimation,
                    self.state_estimation_st[self.state_estimation_st_idx],
                )

        # with stream_processing():
        #     self.state_estimation_st[self.state_estimation_st_idx].buffer(reps, len(self.tau_vec)).map(FUNCTIONS.average()).save(stream_name)

        self.state_estimation_st_idx = self.state_estimation_st_idx + 1
        # Expect: P(e, tau) = A*(exp(-tau/T2)*(1/4+cos(2*pi*(f_ref+x)*tau)/4)+1/2)
        # f_res = f_res + x
        # find A and T2

    def time_domain_ramesy_full_sweep_analysis(self, result_handles, stream_name):
        Pe = result_handles.get(stream_name).fetch_all()
        t = np.array(self.tau_vec) * 4
        plt.plot(t, Pe)
        out = qubit_frequency_tracking._fit_ramsey(self, t, Pe)  # in [ns]
        plt.plot(t, out["fit_func"](t), "m")
        plt.xlabel("time[ns]")
        plt.ylabel("P(|e>)")

        self.fres = self.fres - (out["f"] * 1e9 - self.f_ref)  # Intermediate frequency [Hz]
        print(f"shifting by {out['f'] * 1e9 - self.f_ref}, and now f_res = {self.fres}")

        self.t2 = out["tau"]
        self.phase = out["phase"]
        # This is to choose the actual peak
        # self.tau0 = int((1 - ((self.phase + np.pi) % (2*np.pi) - np.pi)/(2*np.pi))/out["f"]/4)  # [clockcycles]
        # This is to choose the expected peak
        self.tau0 = int(1 / self.f_ref / 4e-9)
        plt.plot(
            self.tau0 * 4,
            out["fit_func"](self.tau0 * 4),
            "r*",
            label="ideal first peak location",
        )
        plt.legend()

    def freq_domain_ramsey_full_sweep(self, reps, fmin, fmax, df, stream_name, oscillation_number=1, correct=False):
        self.tau0 = oscillation_number * int(1 / self.f_ref / 4e-9)
        self.delta = 1 / (self.tau0 * 4e-9) / 4  # the last 4 is for 1/4 of a cycle
        self.fvec = np.arange(fmin, fmax, df).astype(int).tolist()

        with for_(self.m, 0, self.m < reps, self.m + 1):
            with for_(self.f, fmin, self.f < fmax, self.f + df):
                if correct:
                    update_frequency(self.qubit, self.fres_corr)
                else:
                    update_frequency(self.qubit, self.fres)
                play("pump_square", "jpa_pump")
                discriminator.measure_state("readout", "out1", "out2", self.res)
                align(self.qubit, self.rr, "jpa_pump")
                wait(500, self.qubit)
                play("pi", self.qubit, condition=self.res)
                wait(500, self.qubit)

                if correct:
                    update_frequency(self.qubit, self.f + self.corr)
                else:
                    update_frequency(self.qubit, self.f + self.corr)
                play("pi2", self.qubit)
                wait(self.tau0, self.qubit)
                play("pi2", self.qubit)

                align(self.qubit, self.rr, "jpa_pump")
                play("pump_square", "jpa_pump")
                discriminator.measure_state("readout", "out1", "out2", self.res)

                assign(self.state_estimation, Cast.to_fixed(self.res))
                save(
                    self.state_estimation,
                    self.state_estimation_st[self.state_estimation_st_idx],
                )

        self.state_estimation_st_idx = self.state_estimation_st_idx + 1
        # Expect: p(e, f) = A * exp(-tau0/T2) * (1/2+cos(2*pi*(f-f_res)*tau0)/2)
        # Peak at f_res, with A * exp(-tau0/T2)

    def freq_domain_ramsey_full_sweep_analysis(self, result_handles, stream_name):
        # self.delta = self.f_ref/4
        Pe = result_handles.get(stream_name).fetch_all()
        f = np.array(self.fvec)
        plt.plot(f - self.fres, Pe)
        out = qubit_frequency_tracking._fit_ramsey(self, f - self.fres, Pe)  # in Hz
        self.frequency_sweep_amp = out["amp"]
        plt.plot(f - self.fres, out["fit_func"](f - self.fres), "m")
        plt.plot(
            [-self.delta, self.delta],
            out["fit_func"](np.array([-self.delta, self.delta])),
            "r*",
        )
        plt.xlabel("detuning from resonance[Hz]")
        plt.ylabel("P(|e>)")

    def two_points_ramsey(self):
        c = int(1 / (2 * np.pi * self.tau0 * 4e-9 * self.frequency_sweep_amp))
        print(f"c = {c}")
        assign(self.se_vec[0], 0)
        assign(self.se_vec[1], 0)

        with for_(self.p, 0, self.p < 32768, self.p + 1):
            assign(self.f, self.fres - self.delta)

            with for_(self.idx, 0, self.idx < 2, self.idx + 1):
                update_frequency(self.qubit, self.fres)

                play("pump_square", "jpa_pump")
                discriminator.measure_state("readout", "out1", "out2", self.res)
                align(self.qubit, self.rr, "jpa_pump")
                wait(500, self.qubit)
                play("pi", self.qubit, condition=self.res)
                wait(500, self.qubit)

                update_frequency(self.qubit, self.f)
                play("pi2", self.qubit)
                wait(self.tau0, self.qubit)
                play("pi2", self.qubit)

                align(self.qubit, self.rr, "jpa_pump")
                play("pump_square", "jpa_pump")
                discriminator.measure_state("readout", "out1", "out2", self.res)

                assign(
                    self.se_vec[self.idx],
                    self.se_vec[self.idx] + (Cast.to_fixed(self.res) >> 15),
                )
                assign(self.f, self.f + 2 * self.delta)

        assign(self.corr, Cast.mul_int_by_fixed(c, (self.se_vec[0] - self.se_vec[1])))
        assign(self.fres_corr, self.fres_corr + self.corr)
        # update_frequency(self.qubit, self.fres_corr)

        save(self.fres_corr, self.fres_corr_st)
        save(self.corr, self.corr_st)

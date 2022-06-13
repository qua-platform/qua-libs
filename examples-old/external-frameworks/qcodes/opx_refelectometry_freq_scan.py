from qcodes.utils.validators import Arrays
from opx_driver import *
from qm.qua import *


# noinspection PyAbstractClass
class OPXSpectrumScan(OPX):
    def __init__(self, config: Dict, name: str = "OPX", host=None, port=None, **kwargs):
        super().__init__(config, name, host=host, port=port, **kwargs)
        self.add_parameter(
            "f_start",
            initial_value=30e6,
            unit="Hz",
            label="f start",
            vals=Numbers(-400e6, 400e6),
            get_cmd=None,
            set_cmd=None,
        )

        self.add_parameter(
            "f_stop",
            initial_value=70e6,
            unit="Hz",
            label="f stop",
            vals=Numbers(-400e6, 400e6),
            get_cmd=None,
            set_cmd=None,
        )

        self.add_parameter(
            "n_points",
            initial_value=100,
            unit="",
            vals=Numbers(1, 1e9),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "t_meas",
            unit="s",
            initial_value=0.01,
            vals=Numbers(0, 1),
            get_cmd=None,
            set_cmd=None,
        )

        self.add_parameter(
            "freq_axis",
            unit="Hz",
            label="Freq Axis",
            parameter_class=GeneratedSetPoints,
            startparam=self.f_start,
            stopparam=self.f_stop,
            numpointsparam=self.n_points,
            vals=Arrays(shape=(self.n_points.get_latest,)),
        )
        self.add_parameter(
            "amp",
            unit="",
            initial_value=1,
            vals=Numbers(0, 2),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "readout_pulse_length",
            unit="ns",
            vals=Numbers(16, 1e7),
            get_cmd=None,
            set_cmd=None,
        )

    def get_prog(self):
        df = (self.f_stop() - self.f_start()) / self.n_points()
        n_avg = round(self.t_meas() * 1e9 / self.readout_pulse_length())
        with program() as prog:
            n = declare(int)
            f = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_st = declare_stream()
            Q_st = declare_stream()
            with for_(n, 0, n < n_avg, n + 1):
                with for_(f, self.f_start(), f < self.f_stop(), f + df):
                    update_frequency("readout", f)
                    measure(
                        "cw_reflectometry" * amp(self.amp()),
                        "readout",
                        None,
                        demod.full("cos", I, "out1"),
                        demod.full("sin", Q, "out1"),
                    )
                    save(I, I_st)
                    save(Q, Q_st)

            with stream_processing():
                I_st.buffer(self.n_points()).average().save("I")
                Q_st.buffer(self.n_points()).average().save("Q")

        return prog

    def run_exp(self):
        self.execute_prog(self.get_prog())

    def get_res(self):
        if self.result_handles is None:
            n = self.n_points()
            return {"I": (0,) * n, "Q": (0,) * n, "R": (0,) * n, "Phi": (0,) * n}
        else:
            self.result_handles.wait_for_all_values()
            I = self.result_handles.get("I").fetch_all() / self.readout_pulse_length() * 2**12 * 2
            Q = self.result_handles.get("Q").fetch_all() / self.readout_pulse_length() * 2**12 * 2
            R = np.sqrt(I**2 + Q**2)
            phase = np.unwrap(np.angle(I + 1j * Q)) * 180 / np.pi
            return {"I": I, "Q": Q, "R": R, "Phi": phase}

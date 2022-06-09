from opx_driver import *
from qm.qua import *


# noinspection PyAbstractClass
class OPXLineScan(OPX):
    def __init__(self, config: Dict, name: str = "OPX", host=None, port=None, **kwargs):
        super().__init__(config, name, host=host, port=port, **kwargs)
        self.counter = 0
        self.add_parameter(
            "f",
            initial_value=300e6,
            unit="Hz",
            label="freq",
            vals=Numbers(-400e6, 400e6),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "t_meas",
            unit="s",
            initial_value=0.013,
            vals=Numbers(0, 1),
            get_cmd=None,
            set_cmd=None,
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
        n_avg = round(self.t_meas() * 1e9 / self.readout_pulse_length())
        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_st = declare_stream()
            Q_st = declare_stream()
            update_frequency("readout", self.f())
            with infinite_loop_():
                pause()
                with for_(n, 0, n < n_avg, n + 1):
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
                I_st.buffer(n_avg).map(FUNCTIONS.average()).save_all("I")
                Q_st.buffer(n_avg).map(FUNCTIONS.average()).save_all("Q")

        return prog

    def run_exp(self):
        self.execute_prog(self.get_prog())
        self.counter = 0

    def resume(self):
        self.qm.resume()
        self.counter += 1

    def get_res(self):
        if self.result_handles is None:
            return 0, 0, 0, 0
        else:
            self.result_handles.get("I").wait_for_values(self.counter)
            self.result_handles.get("Q").wait_for_values(self.counter)
            I = (
                self.result_handles.get("I").fetch(self.counter - 1)["value"]
                / self.readout_pulse_length()
                * 2**12
                * 2
            )
            Q = (
                self.result_handles.get("Q").fetch(self.counter - 1)["value"]
                / self.readout_pulse_length()
                * 2**12
                * 2
            )
            R = np.sqrt(I**2 + Q**2)
            phase = np.unwrap(np.angle(I + 1j * Q)) * 180 / np.pi
            return I, Q, R, phase

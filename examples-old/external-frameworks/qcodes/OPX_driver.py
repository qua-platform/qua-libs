from time import sleep, time
from typing import Dict

import numpy as np
import ctypes  # only for DLL-based instrument
import time
import qcodes as qc
from qcodes.utils.validators import Numbers, Arrays
from qm.qua import *

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from qm.program._Program import _Program

from qcodes import (
    Instrument,
    VisaInstrument,
    Parameter,
    ManualParameter,
    MultiParameter,
    validators as vals,
    ParameterWithSetpoints,
)
from qcodes.instrument.channel import InstrumentChannel


class OPX(Instrument):
    """
    Driver for interacting with QM OPX
    """

    def __init__(self, config: Dict, name: str = "OPX", **kwargs) -> None:
        """
        Args:
            name: Name to use internally in QCoDeS
        """
        super().__init__(name, **kwargs)

        self.set_config(config=config)
        self._connect()
        self.result_handles = None
        # self.simulation_duration=kwargs['simulation_duration']
        # self.add_parameter('job')

        self.add_parameter("results", label="results", get_cmd=self.result_handles)

        self.add_parameter(
            "sim_time",
            unit="ns",
            label="sim_time",
            initial_value=100000,
            vals=Numbers(
                4,
            ),
            get_cmd=None,
            set_cmd=None,
        )

    def simulate_and_read(self, prog):
        self.simulate_prog(prog, duration=self.sim_time())
        return self.get_res()

    def get_res(self):
        return self.job.result_handles.result.fetch_all()["value"]

    def execute_prog(self, prog):
        self.job = self.qm1.execute(prog)
        self.result_handels = self.job.result_handles

    def simulate_prog(self, prog, duration=1000):
        self.job = self.qm1.simulate(prog, SimulationConfig(duration))

    def set_config(self, config):
        self.config = config

    def _connect(self):
        begin_time = time.time()
        self.QMm = QuantumMachinesManager()
        self.QMm.close_all_quantum_machines()
        self.qm1 = self.QMm.open_qm(self.config)
        idn = {"vendor": "Quantum Machines", "model": "OPX"}
        idn.update(self.QMm.version())
        t = time.time() - (begin_time or self._t0)

        con_msg = (
            "Connected to: {vendor} {model}, client ver. = {client}, server ver. ={server} "
            "in {t:.2f}s".format(t=t, **idn)
        )
        print(con_msg)
        self.log.info(f"Connected to instrument: {idn}")


class GeneratedSetPoints(Parameter):
    """
    A parameter that generates a setpoint array from start, stop and num points
    parameters.
    """

    def __init__(self, startparam, stopparam, numpointsparam, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._startparam = startparam
        self._stopparam = stopparam
        self._numpointsparam = numpointsparam

    def get_raw(self):
        return np.linspace(
            self._startparam(), self._stopparam(), self._numpointsparam()
        )


class Spectrum(ParameterWithSetpoints):
    def get_raw(self):
        npoints = self.root_instrument.n_points.get_latest()
        # self.root_instrument.simulate_prog(self.root_instrument.get_prog())

        return self.root_instrument.simulate_and_read(self.root_instrument.get_prog())


class OPX_SpectrumScan(OPX):
    def __init__(self, config: Dict, name: str = "OPX", **kwargs):
        super().__init__(config, name, **kwargs)
        self.add_parameter(
            "f_start",
            initial_value=0,
            unit="Hz",
            label="f start",
            vals=Numbers(0, 1e3),
            get_cmd=None,
            set_cmd=None,
        )

        self.add_parameter(
            "f_stop",
            unit="Hz",
            label="f stop",
            vals=Numbers(1, 1e3),
            get_cmd=None,
            set_cmd=None,
        )

        self.add_parameter(
            "n_points",
            unit="",
            initial_value=10,
            vals=Numbers(1, 1e3),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "n_avg",
            unit="",
            initial_value=3,
            vals=Numbers(1, 1e3),
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
            "ext_v",
            unit="V",
            label="ext_v",
            vals=Numbers(0, 1e3),
            get_cmd=None,
            set_cmd=None,
        )

        self.add_parameter(
            "spectrum",
            unit="V",
            setpoints=(self.freq_axis,),
            label="Spectrum",
            parameter_class=Spectrum,
            vals=Arrays(shape=(self.n_points.get_latest,)),
        )

    def get_prog(self):
        df = (self.f_stop() - self.f_start()) / self.n_points()
        with program() as prog:
            r = Random()
            vn = declare(int)
            N = declare(int)
            f = declare(int)
            I = declare(fixed)
            result_str = declare_stream()
            # with for_(vn,0,vn<voltage_pts,vn+1):
            with for_(f, self.f_start(), f < self.f_stop(), f + df):
                update_frequency("qe1", f)
                with for_(N, 0, N < self.n_avg(), N + 1):
                    measure("readout", "qe1", None, demod.full("x", I))
                    assign(I, r.rand_fixed())
                    save(I, result_str)

            with stream_processing():
                result_str.buffer(self.n_avg()).map(FUNCTIONS.average()).save_all(
                    "result"
                )

        return prog

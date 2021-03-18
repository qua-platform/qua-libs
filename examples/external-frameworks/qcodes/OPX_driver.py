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

from qcodes import (Instrument, VisaInstrument,
                    Parameter, ManualParameter, MultiParameter,
                    validators as vals)
from qcodes.instrument.channel import InstrumentChannel


class OPX(Instrument):
    """
    Driver for interacting with QM OPX
    """

    def __init__(self, config: Dict, name: str = 'OPX', **kwargs) -> None:
        """
        Args:
            name: Name to use internally in QCoDeS
        """
        super().__init__(name, **kwargs)

        self.set_config(config=config)
        self._connect()
        self.result_handles=None
        # self.add_parameter('job')


        self.add_parameter('results',
                       label='results',
                       get_cmd=self.result_handles)
    def get_res(self):
        return self.job.result_handles.result.fetch_all()['value']

    def execute_prog(self, prog):
        self.job =self.qm1.execute(prog)
        self.result_handels=self.job.result_handles

    def simulate_prog(self, prog, duration=1000):
        self.job=self.qm1.simulate(prog, SimulationConfig(duration))


    def set_config(self, config):
        self.config = config

    def _connect(self):
        begin_time = time.time()
        self.QMm = QuantumMachinesManager()
        self.qm1 = self.QMm.open_qm(self.config)
        idn = {'vendor': 'Quantum Machines', 'model': 'OPX'}
        idn.update(self.QMm.version())
        t = time.time() - (begin_time or self._t0)

        con_msg = ('Connected to: {vendor} {model}, client ver. = {client}, server ver. ={server} '
                   'in {t:.2f}s'.format(t=t, **idn))
        print(con_msg)
        self.log.info(f"Connected to instrument: {idn}")







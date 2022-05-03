import InstrumentDriver
import numpy as np
from qm.qua import *
from configuration import config
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
import json


class Driver(InstrumentDriver.InstrumentWorker):
    """This class implements a simple signal generator driver"""

    def performOpen(self, options={}):
        """Perform the operation of opening the instrument connection"""
        QMm = QuantumMachinesManager()

        my_config = self.getValue("config")
        my_prog = self.getValue("program")

        f = open(my_config, "r")
        config = json.load(f)
        self.QM1 = QMm.open_qm(config)

    def qua_prog_1(self, a=1, f=1e6):
        with program() as prog:
            update_frequency("qe1", f)
            play("playOp" * amp(a), "qe1")
            return prog

    def qua_prog_2(self, a=1, f=1e6):
        with program() as prog:
            update_frequency("qe1", 2 * f)
            play("playOp" * amp(a), "qe1")
            return prog

    def performClose(self, bError=False, options={}):
        """Perform the close instrument connection operation"""
        pass

    def performSetValue(self, quant, value, sweepRate=0.0, options={}):
        """Perform the Set Value instrument operation. This function should
        return the actual value set by the instrument"""
        # just return the value
        return value

    def performGetValue(self, quant, options={}):
        """Perform the Get Value instrument operation"""
        # proceed depending on quantity
        if quant.name == "Signal":
            prog_name = self.getValue("program")
            amp = self.getValue("Amplitude")
            freq = self.getValue("Frequency")
            if prog_name == "qua_prog_1":
                prog = self.qua_prog_1
            elif prog_name == "qua_prog_2":
                prog = self.qua_prog_2

            job = self.QM1.simulate(prog(a=amp, f=freq), SimulationConfig(int(1000)))  # in clock cycles, 4 ns
            samples = job.get_simulated_samples()

            # if asking for signal, start with getting values of other controls

            # calculate time vector from 0 to 1 with 1000 elements

            signal = samples.con1.analog["1"]
            time = np.linspace(0, 1, len(signal))

            trace = quant.getTraceDict(signal, t0=0.0, dt=time[1] - time[0])
            # finally, return the trace object
            return trace
        if quant.name == "run":
            print("here")
            return trace
        else:
            # for other quantities, just return current value of control
            return quant.getValue()

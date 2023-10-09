import os
# qcodes imports
import qcodes as qc
from qcodes import initialise_or_create_database_at, load_or_create_experiment

from qcodes.instrument.parameter import Parameter
# QUA imports
from qualang_tools.external_frameworks.qcodes.opx_driver import OPX
from configuration import *
from qm.qua import *

#######################################
#          Setting-up QCoDeS          #
#######################################
def qcodes_setup(db_name: str="database.db", sample_name:str = "sample", exp_name:str = "experiment"):
    db_file_path = os.path.join(os.getcwd(), db_name)
    qc.config.core.db_location = db_file_path
    initialise_or_create_database_at(db_file_path)

    experiment = load_or_create_experiment(experiment_name = exp_name,
                                           sample_name = sample_name)

    station = qc.Station()

    # Add the OPX to the station
    # Create the OPX instrument and add it to the qcodes station
    opx_instrument = OPX(config, name="OPX_instrument", host=qop_ip, cluster_name=cluster_name, octave=octave_config)
    station.add_component(opx_instrument)
    return station, experiment

def OPX_measurement(readout_type:str = "reflectometry", I=None, I_st=None, Q=None, Q_st=None):
    if I is None:
        I = declare(fixed)
    if I_st is None:
        I_st = declare_stream()
    if readout_type == "reflectometry":
        if Q is None:
            Q = declare(fixed)
        if Q_st is None:
            Q_st = declare_stream()
        measure("readout", "tank_circuit", None, demod.full("cos", I, "out1"), demod.full("sin", Q, "out1"))
        save(I, I_st)
        save(Q, Q_st)
    elif readout_type == "dc_current":
        measure("readout", "TIA", None, integration.full("cos", I, "out1"))
        save(I, I_st)

# Define dummy parameters for demonstration purposes - ti be replaced by real parameters from other instruments
class DummyParameter(Parameter):
    def __init__(self, name, label):
        # only name is required
        super().__init__(
            name=name,
            label=label,
            unit="V",
            docstring="Dummy counter for scanning a variable with qcodes",
        )
        self._count = 0

    # you must provide a get method, a set method, or both.
    def get_raw(self):
        # self._count += 1
        return self._count

    def set_raw(self, val):
        self._count = val
        return self._count
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pprint import pprint
from time import sleep

# qcodes imports
import qcodes as qc
from qcodes import (Measurement,
                    experiments,
                    initialise_database,
                    initialise_or_create_database_at,
                    load_by_guid,
                    load_by_run_spec,
                    load_experiment,
                    load_last_experiment,
                    load_or_create_experiment,
                    new_experiment,
                    ManualParameter)
from qcodes.utils.dataset.doNd import do1d, do2d, do0d
from qcodes.dataset.plotting import plot_dataset
from qcodes.logger.logger import start_all_logging
from qcodes.tests.instrument_mocks import DummyInstrument, DummyInstrumentWithMeasurement
from qcodes.instrument.specialized_parameters import ElapsedTimeParameter
from qcodes.utils.validators import Numbers, Arrays
from qcodes.utils.metadata import diff_param_values
from qcodes.instrument.parameter import ParameterWithSetpoints, Parameter, ScaledParameter
from qcodes.interactive_widget import experiments_widget

# OPX import
# QUA imports
from qm.qua import *
from qualang_tools.external_frameworks.qcodes.opx_driver import OPX
from importlib import reload
import configuration
reload(configuration)
from configuration import *
from qualang_tools.loops import from_array

from helpers import qcodes_setup, DummyParameter, OPX_measurement


#######################################
#          Setting-up QCoDeS          #
#######################################
station, experiment, opx_instrument = qcodes_setup()

#Dummy parameters
V1 = DummyParameter("Plunger1", "Vp1")
V2 = DummyParameter("Plunger2", "Vp2")


#######################################
#             QUA program             #
#######################################
n_avg = 1_000  # Number of averages
# Measurement type to select the correct measure statement - can be either "reflectometry" or "dc_current"
measurement_type = "reflectometry"
simulate = False
#     We are now defining a QUA program  which applies N pulse cycles, averages them n_avg times, for IF frequencies in f_vec
def single_point_readout(simulate=False):
    with program() as prog:
        n = declare(int)
        I = declare(fixed)
        Q = declare(fixed)
        I_st = declare_stream()
        Q_st = declare_stream()

        with infinite_loop_():
            if not simulate:
                pause()

            with for_(n, 0, n < n_avg, n + 1):
                OPX_measurement(measurement_type, I, I_st, Q, Q_st)
                wait(1000)
        with stream_processing():
            I_st.buffer(n_avg).map(FUNCTIONS.average()).save_all("I")
            if measurement_type == "reflectometry":
                Q_st.buffer(n_avg).map(FUNCTIONS.average()).save_all("Q")
    return prog


opx_instrument.qua_program = single_point_readout(simulate=False)
if measurement_type == "reflectometry":
    opx_instrument.readout_pulse_length(reflectometry_readout_length)
else:
    opx_instrument.readout_pulse_length(readout_len)

do2d(
    V1,
    0.02,
    0.04,
    10,
    0.1,
    V2,
    0.220,
    0.240,
    11,
    0.1,
    opx_instrument.resume,
    opx_instrument.get_measurement_parameter(),
    enter_actions=[opx_instrument.run_exp],
    exit_actions=[opx_instrument.halt],
    show_progress=True,
    do_plot=True,
    exp=experiment,
)

import os

import qcodes as qc
from qcodes import initialise_or_create_database_at, load_or_create_experiment

from opx_cw_readout import *
from qcodes.utils.dataset.doNd import do2d
from configuration import *

db_name = "QM_demo_reflectometry.db"  # Database name
sample_name = "demo"  # Sample name
exp_name = "2d_scan_opx_single_cw"  # Experiment name

db_file_path = os.path.join(os.getcwd(), db_name)
qc.config.core.db_location = db_file_path
initialise_or_create_database_at(db_file_path)

experiment = load_or_create_experiment(experiment_name=exp_name, sample_name=sample_name)

opx_single_point = OPXCWReadout(config)
opx_single_point.f(300.8509e6)
opx_single_point.t_meas(0.010)
opx_single_point.amp(1.0)
opx_single_point.readout_pulse_length(readout_pulse_length)
full_data = QMDemodParameters(
    opx_single_point,
    ["I", "Q", "R", "Phi"],
    "Spectrum",
    names=["I", "Q", "R", "Phi"],
    units=["V", "V", "V", "°"],
)
station = qc.Station()
station.add_component(opx_single_point)
do2d(
    VP1,
    -0.55,
    -0.85,
    100,
    1,
    VP2,
    -0.65,
    -0.9,
    100,
    0.0135,
    full_data,
    enter_actions=opx_single_point.run_exp(),
    exit_actions=opx_single_point.halt(),
    show_progress=True,
)

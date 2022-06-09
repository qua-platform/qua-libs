import os

import qcodes as qc
from qcodes import initialise_or_create_database_at, load_or_create_experiment

from opx_single_point_readout import *
from qcodes.utils.dataset.doNd import do2d, do1d
from configuration import *

db_name = "QM_demo_reflectometry.db"  # Database name
sample_name = "demo"  # Sample name
exp_name = "2d_scan_opx_line"  # Experiment name

db_file_path = os.path.join(os.getcwd(), db_name)
qc.config.core.db_location = db_file_path
initialise_or_create_database_at(db_file_path)

experiment = load_or_create_experiment(experiment_name=exp_name, sample_name=sample_name)

opx_line_scan = OPXSinglePointReadout(config)
opx_line_scan.f(300.8509e6)
opx_line_scan.t_meas(0.010)
opx_line_scan.amp(1.0)
opx_line_scan.readout_pulse_length(readout_pulse_length)
full_data = QMDemodParameters(
    opx_line_scan,
    ["I", "Q", "R", "Phi"],
    "Spectrum",
    names=["I", "Q", "R", "Phi"],
    units=["V", "V", "V", "°"],
    # The following two lines tell it that it will return a vector of items!
    shapes=((opx_line_scan.n_points(),),) * 4,
    setpoints=((opx_line_scan.freq_axis(),),) * 4,
)
station = qc.Station()
station.add_component(opx_line_scan)
do1d(
    VP1,
    -0.55,
    -0.85,
    100,
    1,
    opx_line_scan.resume(),
    full_data,
    enter_actions=opx_line_scan.run_exp(),
    exit_actions=opx_line_scan.halt(),
    show_progress=True,
)

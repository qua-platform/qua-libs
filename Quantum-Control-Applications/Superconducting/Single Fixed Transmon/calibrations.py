"""
calibrations.py: template for easily performing single qubit calibration protocols.
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
from qualang_tools.addons.calibration.calibrations import QUA_calibrations


# Relevant configuration parameters
#############################################################
resonator_element = "resonator"
resonator_operation = "readout"
qubit_element = "qubit"
qubit_operation = "gauss"
flux_line_element = None
flux_line_operation = None
int_w = ["cos", "sin", "minus_sin", "cos"]
outs = ["out1", "out2"]
#############################################################

# Initialize the calibration API with the relevant configuration parameters
my_calib = QUA_calibrations(
    configuration=config,
    readout=(resonator_element, resonator_operation),
    qubit=(qubit_element, qubit_operation),
    integration_weights=int_w,
    outputs=outs,
)

n_avg = 1000
# Resonator spectroscopy scan
scans = [
    ("frequency", np.arange(100e6, 300e6, step=0.1e6)),
]
my_calib.set_resonator_spectroscopy(scan_variables=scans, iterations=n_avg, cooldown_time=0)

# Rabi scan
scans = [
    ("duration", np.arange(40, 4000, step=100)),
    ("amplitude", np.linspace(0.5, 1.99, num=51)),
]
# my_calib.set_rabi(scan_variables=scans, iterations=n_avg, cooldown_time=0)

# T1 scan
scans = [
    ("duration", np.arange(5, 4000, step=10)),
]
# my_calib.set_T1(scan_variables=scans, iterations=n_avg, cooldown_time=0)


# Ramsey scan
scans = [
    ("frequency", np.linspace(100e6, 200e6, num=101)),
    ("duration", np.arange(40, 4000, step=10)),
]
# my_calib.set_ramsey(scan_variables=scans, iterations=n_avg, cooldown_time=0, idle_time=10)

# Raw traces
# my_calib.set_raw_traces(iterations=n_avg, cooldown_time=0)

# Time of flight
# my_calib.set_time_of_flight(iterations=n_avg, cooldown_time=0)

################################
# Open communication with QOP  #
################################
qmm = QuantumMachinesManager(qop_ip)
qm = qmm.open_qm(config)

# Run calibrations
options = {
    "fontsize": 14,
    "color": "b",
    "marker": ".",
    "linewidth": 1,
    "figsize": (12, 15),
}
my_calib.simulate_calibrations(machine=qm, simulation_duration=5000)
# my_calib.run_calibrations(quantum_machine=qm, plot="live", plot_options=options)
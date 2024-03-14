from qm import QuantumMachinesManager
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
#############################################################
my_calib = QUA_calibrations(
    configuration=config,
    readout=(resonator_element, resonator_operation),
    qubit=(qubit_element, qubit_operation),
    integration_weights=int_w,
    outputs=outs,
)

# Set scan parameters
#############################################################
n_avg = 1000
# Resonator spectroscopy scan
scans_resonator_spectroscopy = [("frequency", np.arange(100e6, 300e6, step=0.1e6))]
# Rabi scan
scans_rabi = [
    ("duration", np.arange(40, 4000, step=100)),
    ("amplitude", np.linspace(0.5, 1.99, num=51)),
]
# T1 scan
scans_T1 = [("duration", np.arange(5, 4000, step=10))]
# Ramsey scan
scans_ramsey = [
    ("frequency", np.linspace(100e6, 200e6, num=101)),
    ("duration", np.arange(40, 4000, step=10)),
]

# Set calibrations | Just uncomment the calibrations you want to run
#############################################################
# --> Raw traces
# my_calib.set_raw_traces(iterations=n_avg, cooldown_time=0)
# --> Time of flight
# my_calib.set_time_of_flight(iterations=n_avg, cooldown_time=0)
# --> Resonator spectroscopy scan
# my_calib.set_resonator_spectroscopy(scan_variables=scans_resonator_spectroscopy, iterations=n_avg, cooldown_time=0)
# --> Rabi scan
# my_calib.set_rabi(scan_variables=scans_rabi, iterations=n_avg, cooldown_time=0)
# --> T1 scan
# my_calib.set_T1(scan_variables=scans_T1, iterations=n_avg, cooldown_time=0)
# --> Ramsey scan
# my_calib.set_ramsey(scan_variables=scans_ramsey, iterations=n_avg, cooldown_time=0, idle_time=10)

# Open communication with QOP
#############################################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)
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

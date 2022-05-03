# Tested on Keysight FieldFox N9917A

from qm.QuantumMachinesManager import QuantumMachinesManager
import time
import matplotlib.pyplot as plt
from configuration import *
import numpy as np
import scipy.optimize as opti
from auto_mixer_tools_visa import KeysightFieldFox

##############
# Parameters #
##############
# Important Parameters:
address = "TCPIP0::192.168.1.9::inst0::INSTR"  # The address for the SA, opened using visa.
bDoSweeps = True  # If True, performs a large sweep before and after the optimization.
method = 1  # If set to 1, checks power using a channel power measurement. If set to 2, checks power using a marker.

# Parameters for SA - Measurement:
measBW = 100  # Measurement bandwidth
measNumPoints = 101

# Parameters for SA - Sweep:
sweepBW = 1e3
fullNumPoints = 1201
fullSpan = int(abs(qubit_IF * 4.1))  # Larger than 4 such that we'll see spurs
startFreq = qubit_LO - fullSpan / 2
stopFreq = qubit_LO + fullSpan / 2
freq_vec = np.linspace(float(startFreq), float(stopFreq), int(fullNumPoints))

# Parameters for Nelder-Mead
initial_simplex = np.zeros([3, 2])
initial_simplex[0, :] = [0, 0]
initial_simplex[1, :] = [0, 0.1]
initial_simplex[2, :] = [0.1, 0]
xatol = 1e-4  # 1e-4 change in DC offset or gain/phase
fatol = 3  # dB change tolerance
maxiter = 50  # 50 iterations should be more then enough, but can be changed.

##########
# Execute:
##########

# Execute the mixer_cal program:
qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

calib = KeysightFieldFox(address, qm)
calib.method = method

# Set video bandwidth to be automatic and bandwidth to be manual. Disable continuous mode
calib.set_automatic_video_bandwidth(1)
calib.set_automatic_bandwidth(0)
calib.set_cont_off()

if bDoSweeps:
    # Set Bandwidth and start/stop freq of SA for a large sweep
    calib.set_bandwidth(sweepBW)
    calib.set_sweep_points(fullNumPoints)
    calib.set_center_freq(qubit_LO)
    calib.set_span(fullSpan)

    # Do a single read
    calib.get_single_trigger()

    # Query the FieldFox response data
    amp = calib.get_full_trace()

    plt.figure("Full Spectrum")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude (dBm)")
    plt.plot(freq_vec, amp)
    plt.show()

# Configure measure
if method == 1:  # Channel power
    calib.enable_measurement()
    calib.sets_measurement_integration_bw(10 * measBW)
    calib.disables_measurement_averaging()
elif method == 2:  # Marker
    calib.get_single_trigger()
    calib.active_marker(1)
calib.set_sweep_points(measNumPoints)
calib.set_span(10 * measBW)
calib.set_bandwidth(measBW)

# Get Signal
calib.set_center_freq(qubit_LO + qubit_IF)
if method == 2:  # Marker
    calib.set_marker_freq(1, qubit_LO + qubit_IF)
signal = int(calib.get_amp())

# Optimize LO leakage
calib.set_center_freq(qubit_LO)
if method == 2:  # Marker
    calib.set_marker_freq(1, qubit_LO)
start_time = time.time()
fun_leakage = lambda x: calib.get_leakage(x[0], x[1])
res_leakage = opti.minimize(
    fun_leakage,
    [0, 0],
    method="Nelder-Mead",
    options={
        "xatol": xatol,
        "fatol": fatol,
        "initial_simplex": initial_simplex,
        "maxiter": maxiter,
    },
)
print(
    f"LO Leakage Results: Found a minimum of {int(res_leakage.fun)} dBm at I0 = {res_leakage.x[0]:.5f}, Q0 = {res_leakage.x[1]:.5f} in "
    f"{int(time.time() - start_time)} seconds --- {signal - int(res_leakage.fun)} dBc"
)

# Optimize image
calib.set_center_freq(qubit_LO - qubit_IF)
if method == 2:  # Marker
    calib.set_marker_freq(1, qubit_LO - qubit_IF)
start_time = time.time()
fun_image = lambda x: calib.get_image(x[0], x[1])
res_image = opti.minimize(
    fun_image,
    [0, 0],
    method="Nelder-Mead",
    options={
        "xatol": xatol,
        "fatol": fatol,
        "initial_simplex": initial_simplex,
        "maxiter": maxiter,
    },
)
print(
    f"Image Rejection Results: Found a minimum of {int(res_image.fun)} dBm at I0 = {res_image.x[0]:.5f}, Q0 = {res_image.x[1]:.5f} in "
    f"{int(time.time() - start_time)} seconds --- {signal - int(res_image.fun)} dBc"
)

# Turn measurement off
if method == 1:  # Channel power
    calib.disables_measurement()

# Set parameters back for a large sweep
if bDoSweeps:
    # Set Bandwidth and start/stop freq of SA for a large sweep
    calib.set_bandwidth(sweepBW)
    calib.set_sweep_points(fullNumPoints)
    calib.set_center_freq(qubit_LO)
    calib.set_span(fullSpan)

    # Do a single read
    calib.get_single_trigger()

    # Query the FieldFox response data
    amp = calib.get_full_trace()

    plt.figure("Full Spectrum")
    plt.plot(freq_vec, amp)

    plt.legend(["Before", "After"])
    plt.show()

# Return the FieldFox back to continuous mode
calib.set_cont_on()

# On exit clean a few items up.
calib.__del__()

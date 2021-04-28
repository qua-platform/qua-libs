# Tested on Keysight FieldFox N9917A

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import time
import matplotlib.pyplot as plt
from configuration import *
import numpy as np
import pyvisa as visa
import scipy.optimize as opti

##############
# Parameters #
##############

# Parameters for Nelder-Mead
initial_simplex = np.zeros([3, 2])
initial_simplex[0, :] = [0, 0]
initial_simplex[1, :] = [0, 0.1]
initial_simplex[2, :] = [0.1, 0]
xatol = 1e-4  # 1e-4 change in DC offset or gain/phase
fatol = 3  # dB change tolerance
maxiter = 50  # 50 iterations should be more then enough, but can be changed.

# Parameters for SA - Measurement:
measBW = 100  # Measurement integration bandwidth
measNumPoints = 101

# Parameters for SA - Sweep:
bDoSweeps = True  # If True, performs a large sweep before and after the optimization.
fullNumPoints = 1201
fullSpan = int(abs(qubit_IF * 4.1))  # Larger than 4 such that we'll see spurs
startFreq = qubit_LO - fullSpan / 2
stopFreq = qubit_LO + fullSpan / 2
freq_vec = np.linspace(float(startFreq), float(stopFreq), int(fullNumPoints))

# Open connection to SA
rm = visa.ResourceManager()
sa = rm.open_resource('TCPIP0::192.168.1.9::inst0::INSTR')
sa.timeout = 100000

#############
# Functions #
#############

with program() as mixer_cal:
    with infinite_loop_():
        play("test_pulse", "qubit")


def get_amp():
    sa.quert("INIT:IMM;*OPC?")
    sig = float(sa.query('CALC:MARK1:Y?'))
    return sig


def get_leakage(i0, q0):
    qm.set_dc_offset_by_qe("qubit", "I", i0)
    qm.set_dc_offset_by_qe("qubit", "Q", q0)
    amp_ = get_amp()
    return amp_


def get_image(g, p):
    job.set_element_correction('qubit', IQ_imbalance_correction(g, p))
    amp_ = get_amp()
    return amp_


##########
# Execute:
##########

# Execute the mixer_cal program:
qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
job = qm.execute(mixer_cal)

# Set video bandwidth to be automatic and bandwidth to be manual. Disable continuous mode
sa.write('SENS:BAND:VID:AUTO 1')
sa.write('SENS:BAND:AUTO 0')
sa.query("INIT:CONT OFF;*OPC?")

if bDoSweeps:
    # Set Bandwidth and start/stop freq of SA for a large sweep
    sa.write('SENS:BAND 1e3')
    sa.write("SENS:SWE:POIN " + str(fullNumPoints))
    sa.write("SENS:FREQ:START " + str(startFreq))
    sa.write("SENS:FREQ:STOP " + str(stopFreq))

    # Do a single read
    sa.query("INIT:IMM;*OPC?")

    # Query the FieldFox response data
    sa.write("TRACE:DATA?")
    ff_SA_Trace_Data = sa.read()
    # Data from the Fieldfox comes out as a string separated by ',':
    # '-1.97854112E+01,-3.97854112E+01,-2.97454112E+01,-4.92543112E+01,-5.17254112E+01,-1.91254112E+01...\n'
    # The code below turns it into an a python list of floats

    # Use split to turn long string to an array of values
    ff_SA_Trace_Data_Array = ff_SA_Trace_Data.split(",")
    amp = [float(i) for i in ff_SA_Trace_Data_Array]

    plt.figure('Full Spectrum')
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude (dBm)")
    plt.plot(freq_vec, amp)
    plt.show()

# Configure measurement
sa.write("SENS:SWE:POIN " + str(measNumPoints))
sa.write(f"SENS:FREQ:SPAN {10 * measBW}")
sa.write(f'SENS:BAND {int(measBW)}')
sa.query("INIT:IMM;*OPC?")
sa.write('CALC:MARK1:ACT')
sa.write(f'CALC:MARK1:X {qubit_LO}')  # Leakage

# Get Signal
sa.write(f"SENS:FREQ:CENT {qubit_LO + qubit_IF}")
signal = int(get_amp())

# Optimize LO leakage
sa.write(f"SENS:FREQ:CENT {qubit_LO}")
start_time = time.time()
fun_leakage = lambda x: get_leakage(x[0], x[1])
res_leakage = opti.minimize(fun_leakage, [0, 0], method='Nelder-Mead', options={'xatol': xatol, 'fatol': fatol, 'initial_simplex': initial_simplex, 'maxiter': maxiter})
print(f"LO Leakage Results: Found a minimum of {int(res_leakage.fun)} dBm at I0 = {res_leakage.x[0]:.4f}, Q0 = {res_leakage.x[1]:.4f} in "
      f"{int(time.time() - start_time)} seconds --- {signal - int(res_leakage.fun)} dBc")

# Optimize image
sa.write(f"SENS:FREQ:CENT {qubit_LO - qubit_IF}")
start_time = time.time()
fun_image = lambda x: get_image(x[0], x[1])
res_image = opti.minimize(fun_image, [0, 0], method='Nelder-Mead', options={'xatol': xatol, 'fatol': fatol, 'initial_simplex': initial_simplex, 'maxiter': maxiter})
print(f"Image Results: Found a minimum of {int(res_image.fun)} dBm at g = {res_image.x[0]:.4f}, phi = {res_image.x[1]:.4f} in "
      f"{int(time.time() - start_time)} seconds --- {signal - int(res_image.fun)} dBc")

# Set parameters back for a large sweep
if bDoSweeps:
    sa.write('SENS:BAND 1e3')
    sa.write("SENS:SWE:POIN " + str(fullNumPoints))
    sa.write("SENS:FREQ:START " + str(startFreq))
    sa.write("SENS:FREQ:STOP " + str(stopFreq))
    sa.query("INIT:IMM;*OPC?")

    # Do a large sweep
    sa.write("TRACE:DATA?")
    ff_SA_Trace_Data = sa.read()
    ff_SA_Trace_Data_Array = ff_SA_Trace_Data.split(",")
    amp = [float(i) for i in ff_SA_Trace_Data_Array]
    plt.figure('Full Spectrum')
    plt.plot(freq_vec, amp)

    plt.legend(['Before', 'After'])
    plt.show()

# Return the FieldFox back to continuous mode
sa.write("INIT:CONT ON")

# On exit clean a few items up.
sa.clear()
sa.close()

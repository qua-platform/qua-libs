# Works on Keysight FieldFox, tested on N9917A

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import time
import matplotlib.pyplot as plt
from configuration import *
import numpy as np
import pyvisa as visa
import scipy.optimize as opti


#############
# Functions #
#############

with program() as mixer_cal:
    with infinite_loop_():
        play("test_pulse", "qubit")


def get_amp():
    sa.write("INIT:IMM;*OPC?")
    sa.read()
    sa.write(f'CALC:MARK1:Y?')
    sig = float(sa.read())
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

# Configure the SA :
numPoints = 1201
span = int(abs(qubit_IF * 4.1))
startFreq = qubit_LO-span/2
stopFreq = qubit_LO+span/2
freq_vec = np.linspace(float(startFreq), float(stopFreq), int(numPoints))

rm = visa.ResourceManager()
sa = rm.open_resource('TCPIP0::192.168.1.9::inst0::INSTR')
sa.timeout = 100000
# Set Bandwidth and start/stop freq of SA
sa.write('SENS:BAND:VID:AUTO 1')
sa.write('SENS:BAND:AUTO 0')
sa.write('SENS:BAND 1e3')
sa.write("SENS:SWE:POIN " + str(numPoints))
sa.write("SENS:FREQ:START " + str(startFreq))
sa.write("SENS:FREQ:STOP " + str(stopFreq))

time.sleep(2)

# Do a single read
sa.write("INIT:CONT OFF;*OPC?")
sa.read()

sa.write("INIT:IMM;*OPC?")
sa.read()

# Query the FieldFox response data
sa.write("TRACE:DATA?")
ff_SA_Trace_Data = sa.read()

# Use split to turn long string to an array of values
ff_SA_Trace_Data_Array = ff_SA_Trace_Data.split(",")
amp = [float(i) for i in ff_SA_Trace_Data_Array]
plt.figure('Full Spectrum')
plt.xlabel("Frequency")
plt.ylabel("Amplitude (dBm)")
plt.plot(freq_vec, amp)
plt.show()

# Get signal - change BW and start stop to be around image
sa.write('SENS:BAND 1e1')
sa.write(f"SENS:SWE:POIN 41")
sa.write(f"SENS:FREQ:START {qubit_LO+qubit_IF-100}")
sa.write(f"SENS:FREQ:STOP {qubit_LO+qubit_IF+100}")

# Set marker
sa.write("INIT:IMM;*OPC?")
sa.read()
sa.write('CALC:MARK1:ACT')
sa.write(f'CALC:MARK1:X {qubit_LO + qubit_IF}')  # Signal

signal = int(get_amp())

# LO leakage optimize - change BW and start stop to be around LO leakage
sa.write('SENS:BAND 1e1')
sa.write(f"SENS:SWE:POIN 41")
sa.write(f"SENS:FREQ:START {qubit_LO-100}")
sa.write(f"SENS:FREQ:STOP {qubit_LO+100}")

# Set marker
sa.write("INIT:IMM;*OPC?")
sa.read()
sa.write('CALC:MARK1:ACT')
sa.write(f'CALC:MARK1:X {qubit_LO}')  # Leakage

# Optimize LO leakage
start_time = time.time()
fun_leakage = lambda x: get_leakage(x[0], x[1])
res_leakage = opti.minimize(fun_leakage, [0, 0], method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 3})
print(f"LO --- I0 = {res_leakage.x[0]:.4f}, Q0 = {res_leakage.x[1]:.4f} --- "
      f"{int(time.time() - start_time)} seconds --- {signal - int(res_leakage.fun)} dBc")

# Image optimize - change BW and start stop to be around image
sa.write('SENS:BAND 1e1')
sa.write(f"SENS:SWE:POIN 41")
sa.write(f"SENS:FREQ:START {qubit_LO-qubit_IF-100}")
sa.write(f"SENS:FREQ:STOP {qubit_LO-qubit_IF+100}")

# Set marker
sa.write("INIT:IMM;*OPC?")
sa.read()
sa.write('CALC:MARK1:ACT')
sa.write(f'CALC:MARK1:X {qubit_LO - qubit_IF}')  # Leakage

# Optimize LO leakage
start_time = time.time()
fun_image = lambda x: get_image(x[0], x[1])
res_image = opti.minimize(fun_image, [0, 0], method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 3})
print(f"Image --- g = {res_image.x[0]:.4f}, phi = {res_image.x[1]:.4f} --- "
      f"{int(time.time() - start_time)} seconds --- {signal - int(res_image.fun)} dBc")

# Set parameters back for a large sweep
sa.write('SENS:BAND:VID:AUTO 1')
sa.write('SENS:BAND:AUTO 0')
sa.write('SENS:BAND 1e3')
sa.write("SENS:SWE:POIN " + str(numPoints))
sa.write("SENS:FREQ:START " + str(startFreq))
sa.write("SENS:FREQ:STOP " + str(stopFreq))
sa.write("INIT:IMM;*OPC?")
sa.read()

# Do a large sweep
sa.write("TRACE:DATA?")
ff_SA_Trace_Data = sa.read()
ff_SA_Trace_Data_Array = ff_SA_Trace_Data.split(",")
amp = [float(i) for i in ff_SA_Trace_Data_Array]
plt.figure('Full Spectrum')
plt.plot(freq_vec, amp)

plt.legend(['Before', 'After'])
plt.show()

# Return the FieldFox back to free run trigger mode
sa.write("INIT:CONT ON")

# On exit clean a few items up.
sa.clear()
sa.close()

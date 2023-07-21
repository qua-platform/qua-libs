from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from configuration import *
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from qm.octave import *
from macros import assign_variables_to_element

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("quam_state.json", flat_data=False)
config = build_config(machine)

n_avg = 1000  # Number of averaging loops
depletion_time = 10000
nb_of_qubits = 9

###################
# The QUA program #
###################
with program() as raw_trace_prog:
    n = declare(int)
    adc_st = [declare_stream(adc_trace=True) for _ in range(nb_of_qubits)]

    with for_(n, 0, n < n_avg, n + 1):
        for i in range(9):
            reset_phase(machine.resonators[i].name)
            measure("readout", machine.resonators[i].name, adc_st[i])
        wait(depletion_time * u.ns)

    with stream_processing():
        # Will save average:
        adc_st[0].input1().average().save("adc1")
        adc_st[0].input2().average().save("adc2")
        # # Will save only last run:
        adc_st[0].input1().save("adc1_single_run")
        adc_st[0].input2().save("adc2_single_run")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)
qm = qmm.open_qm(config)
job = qm.execute(raw_trace_prog, flags=['auto-element-thread'])
res_handles = job.result_handles
res_handles.wait_for_all_values()
adc1 = u.raw2volts(res_handles.get("adc1").fetch_all())
adc2 = u.raw2volts(res_handles.get("adc2").fetch_all())
adc1_single_run = u.raw2volts(res_handles.get("adc1_single_run").fetch_all())
adc2_single_run = u.raw2volts(res_handles.get("adc2_single_run").fetch_all())

adc1_mean = np.mean(adc1)
adc2_mean = np.mean(adc2)
adc1_unbiased = adc1 - np.mean(adc1)
adc2_unbiased = adc2 - np.mean(adc2)
signal = savgol_filter(np.abs(adc1_unbiased + 1j * adc2_unbiased), 11, 3)
# detect arrival of readout signal
th = (np.mean(signal[:100]) + np.mean(signal[:-100])) / 2
delay = np.where(signal > th)[0][0]
delay = np.round(delay / 4) * 4
dc_offset_i = -adc1_mean
dc_offset_q = -adc2_mean
# Plot data
fig = plt.figure()
plt.subplot(121)
plt.title("Single run")
plt.plot(adc1_single_run, "b", label="Input 1")
plt.plot(adc2_single_run, "r", label="Input 2")
xl = plt.xlim()
yl = plt.ylim()
plt.plot(xl, adc1_mean * np.ones(2), "k--")
plt.plot(xl, adc2_mean * np.ones(2), "k--")
plt.plot(delay * np.ones(2), yl, "k--")
plt.axhline(y=0.5)
plt.axhline(y=-0.5)
plt.xlabel("Time [ns]")
plt.ylabel("Signal amplitude [V]")
plt.legend()
plt.subplot(122)
plt.title("Averaged run")
plt.plot(adc1, "b", label="Input 1")
plt.plot(adc2, "r", label="Input 2")
xl = plt.xlim()
yl = plt.ylim()
plt.plot(xl, adc1_mean * np.ones(2), "k--")
plt.plot(xl, adc2_mean * np.ones(2), "k--")
plt.plot(delay * np.ones(2), yl, "k--")
plt.xlabel("Time [ns]")
plt.legend()
plt.grid("all")
plt.tight_layout()
plt.show()

###### fft
plt.figure()
signal = np.fft.fft(adc1)
freqs = np.fft.fftfreq(len(adc1), d=1.0 / 1e9)
plt.plot(freqs, np.abs(signal))
plt.xlim(10e6, 400e6)
# plt.ylim(-1, 1)
plt.xlabel("Frequency [Hz]")
plt.ylabel("[a.u.]")
plt.show()

# Update the config
print(f"DC offset to add to I: {dc_offset_i:.6f} V")
print(f"DC offset to add to Q: {dc_offset_q:.6f} V")
print(f"TOF to add: {delay} ns")

machine.global_parameters.con1_downconversion_offset_I += dc_offset_i
machine.global_parameters.con1_downconversion_offset_Q += dc_offset_q
machine.global_parameters.time_of_flight += int(delay)
# machine._save("quam_state.json")
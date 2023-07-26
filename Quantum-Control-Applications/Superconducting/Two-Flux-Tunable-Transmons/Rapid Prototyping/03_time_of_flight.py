from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from quam import QuAM
from configuration import build_config, u

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("quam_bootstrap_state.json", flat_data=False)
config = build_config(machine)

###################
# The QUA program #
###################
n_avg = 100  # Number of averaging loops
depletion_time = 1000 // 4

with program() as raw_trace_prog:
    n = declare(int)
    adc_st = declare_stream(adc_trace=True)

    with for_(n, 0, n < n_avg, n + 1):
        reset_phase("rr0")
        measure("readout", "rr0", adc_st)
        wait(depletion_time, "rr0")

    with stream_processing():
        # Will save average:
        adc_st.input1().average().save("adc1")
        adc_st.input2().average().save("adc2")
        # # Will save only last run:
        adc_st.input1().save("adc1_single_run")
        adc_st.input2().save("adc2_single_run")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.qop_port)
qm = qmm.open_qm(config)
job = qm.execute(raw_trace_prog)
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
delay = int(np.round(delay / 4) * 4)
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

# Update the config
print(f"DC offset to add to I: {dc_offset_i:.6f} V")
print(f"DC offset to add to Q: {dc_offset_q:.6f} V")
print(f"TOF to add: {delay} ns")

machine.global_parameters.downconversion_offset_I += dc_offset_i
machine.global_parameters.downconversion_offset_Q += dc_offset_q
machine.global_parameters.time_of_flight += delay
# machine._save("quam_bootstrap_state.json")

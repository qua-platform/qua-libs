from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from configuration import *
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


###################
# The QUA program #
###################
n_avg = 100  # Number of averaging loops
cooldown_time = 2 * u.us // 4  # Resonator cooldown time in clock cycles (4ns)

with program() as raw_trace_prog:
    n = declare(int)
    adc_st = declare_stream(adc_trace=True)

    with for_(n, 0, n < n_avg, n + 1):
        reset_phase("resonator")
        measure("readout", "resonator", adc_st)
        wait(cooldown_time, "resonator")

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
qmm = QuantumMachinesManager(qop_ip, qop_port, octave=octave_config)

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

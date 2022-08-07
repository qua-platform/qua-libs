from state_and_config import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate.credentials import create_credentials
from qm.simulate import SimulationConfig
from qm.qua import *
from qualang_tools.units import unit
import matplotlib.pyplot as plt
import numpy as np


# build config
config = build_config(state)
qmm = QuantumMachinesManager(
    host="nord-quantique-d14d58b1.quantum-machines.co",
    port=443,
    credentials=create_credentials(),
)


##############################
# Program-specific variables #
##############################
qubits = [0, 1]
n_avg = 100  # Number of averaging loops
u = unit()
cooldown_time = 2 * u.us // 4  # Resonator cooldown time in clock cycles (4ns)

###################
# The QUA program #
###################

with program() as raw_adc_traces:

    n = declare(int)
    adc_st = declare_stream(adc_trace=True)

    with for_(n, 0, n < n_avg, n + 1):
        idx = 0
        for q in [0, 1]:
            # [0]\[1]: raw traces of rr0\rr1.
            # [0,1]: raw traces of multiplex singal from both rr0 and rr1
            reset_phase(f"rr{q}")
            measure("readout", f"rr{q}", adc_st)
            wait(cooldown_time, f"rr{q}")
            idx += 1

    with stream_processing():
        # Will save average:
        adc_st.input1().average().save("adc1")
        adc_st.input2().average().save("adc2")
        # Will save only last run:
        adc_st.input1().save("adc1_single_run")
        adc_st.input2().save("adc2_single_run")


job = qmm.simulate(build_config(state), raw_adc_traces, SimulationConfig(1500))
job.get_simulated_samples().con1.plot()
res_handles = job.result_handles
res_handles.wait_for_all_values()
adc1 = u.raw2volts(res_handles.get("adc1").fetch_all())
adc2 = u.raw2volts(res_handles.get("adc2").fetch_all())
adc1_single_run = u.raw2volts(res_handles.get("adc1_single_run").fetch_all())
adc2_single_run = u.raw2volts(res_handles.get("adc2_single_run").fetch_all())
plt.figure()
plt.subplot(121)
plt.title("Single run")
plt.plot(adc1_single_run, label="Input 1")
plt.plot(adc2_single_run, label="Input 2")
plt.xlabel("Time [ns]")
plt.ylabel("Signal amplitude [V]")
plt.legend()

plt.subplot(122)
plt.title("Averaged run")
plt.plot(adc1, label="Input 1")
plt.plot(adc2, label="Input 2")
plt.xlabel("Time [ns]")
plt.legend()
plt.tight_layout()

print(f"\nInput1 mean: {np.mean(adc1)} V\n" f"Input2 mean: {np.mean(adc2)} V")

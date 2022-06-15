from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig

###################
# The QUA program #
###################
with program() as tof_cal:
    n = declare(int)
    adc_st = declare_stream(adc_trace=True)

    with for_(n, 0, n < 50, n + 1):
        reset_phase("resonator")
        measure("short_readout", "resonator", adc_st)
        wait(1000, "resonator")

    with stream_processing():
        # Will save average:
        adc_st.input1().average().save("adc1")
        adc_st.input2().average().save("adc2")
        # Will save only last run:
        adc_st.input1().save("adc1_single_run")
        adc_st.input2().save("adc2_single_run")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

simulate = True

if simulate:
    simulation_config = SimulationConfig(duration=1000)  # in clock cycles
    job = qmm.simulate(config, tof_cal, simulation_config)
    job.get_simulated_samples().con1.plot()

else:

    qm = qmm.open_qm(config)
    job = qm.execute(tof_cal)
    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    adc1 = res_handles.get("adc1").fetch_all() / 2**12
    adc2 = res_handles.get("adc2").fetch_all() / 2**12
    adc1_single_run = res_handles.get("adc1_single_run").fetch_all() / 2**12
    adc2_single_run = res_handles.get("adc2_single_run").fetch_all() / 2**12

    plt.figure()
    plt.title("Single run (Check ADCs saturation)")
    plt.plot(adc1_single_run)
    plt.plot(adc2_single_run)
    plt.show()

    plt.figure()
    plt.title("Averaged run (Check ToF & DC Offset)")
    plt.plot(adc1)
    plt.plot(adc2)
    plt.show()
    print(f"Input1 mean: {np.mean(adc1)} V\n" f"Input2 mean: {np.mean(adc2)} V")

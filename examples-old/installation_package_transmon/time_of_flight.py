from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import config
import matplotlib.pyplot as plt

################################
# Open quantum machine manager #
################################

qmm = QuantumMachinesManager()

########################
# Open quantum machine #
########################

qm = qmm.open_qm(config)

###################
# The QUA program #
###################

with program() as tof_cal:

    n = declare(int)  # variable for averaging loop
    adc_st = declare_stream(adc_trace=True)  # stream to save ADC data

    with for_(n, 0, n < 50, n + 1):
        reset_phase("resonator")  # reset the phase of the next played pulse
        measure("readout", "resonator", adc_st)
        wait(250, "resonator")  # wait for photons in resonator to decay

    with stream_processing():
        adc_st.input1().average().save("adc1")
        adc_st.input2().average().save("adc2")

        # Will save only last run:
        adc_st.input1().save("adc1_single_run")
        adc_st.input2().save("adc2_single_run")

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    # simulation properties
    simulate_config = SimulationConfig(
        duration=1000,
        simulation_interface=LoopbackInterface(([("con1", 1, "con1", 1)])),
    )
    job = qmm.simulate(config, tof_cal, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    job = qm.execute(tof_cal)  # execute QUA program

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    adc1 = res_handles.get("adc1").fetch_all()
    adc2 = res_handles.get("adc2").fetch_all()
    adc1_single_run = res_handles.get("adc1_single_run").fetch_all()
    adc2_single_run = res_handles.get("adc2_single_run").fetch_all()

    plt.figure()
    plt.title("Single run (Check ADCs saturation)")
    plt.plot(adc1_single_run)
    plt.plot(adc2_single_run)

    plt.figure()
    plt.title("Averaged run (Check ToF & DC Offset)")
    plt.plot(adc1)
    plt.plot(adc2)

"""
A script to measure the analog signal when no drive is applied. Allows you to correct for offsets
"""
from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
from qm.simulate.credentials import create_credentials

###################
# The QUA program #
###################

with program() as input_cal:

    n = declare(int)

    adc_st = declare_stream(adc_trace=True)

    with for_(n, 0, n < 5000, n + 1):
        play("activate", "switch_receiver", duration=500)
        wait(150, "resonator")
        reset_phase("resonator")
        measure("readout", "resonator", adc_st)
        wait(250, "resonator")

    with stream_processing():
        adc_st.input1().average().save("adc1")
        adc_st.input2().average().save("adc2")

        adc_st.input1().save("adc1_single_run")
        adc_st.input2().save("adc2_single_run")

################################
# Open quantum machine manager #
################################

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

#######################
# Simulate or execute #
#######################

simulate = False

if simulate:
    # simulation properties
    simulate_config = SimulationConfig(
        duration=2000,
        simulation_interface=LoopbackInterface(([("con1", 3, "con1", 1), ("con1", 4, "con1", 2)]), latency=180),
    )
    job = qmm.simulate(config, input_cal, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    qm = qmm.open_qm(config)

    job = qm.execute(input_cal)  # execute QUA program

    res_hand = job.result_handles
    res_hand.wait_for_all_values()

    adc1 = res_hand.get("adc1").fetch_all()
    adc2 = res_hand.get("adc2").fetch_all()
    adc1_single_run = res_hand.get("adc1_single_run").fetch_all()
    adc2_single_run = res_hand.get("adc2_single_run").fetch_all()

    plt.figure()
    plt.title("Single run (Check ADCs saturation)")
    plt.plot(adc1_single_run / 2**12)
    plt.plot(adc2_single_run / 2**12)

    plt.figure()
    plt.title("Averaged run")
    plt.plot(adc1 / 2**12)
    plt.plot(adc2 / 2**12)

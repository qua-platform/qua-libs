from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from configuration import *
from qm import SimulationConfig
from qm import LoopbackInterface

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

with program() as cal_tt:

    # Declare QUA variables
    ###################
    times = declare(int, size=1000)  # 'size' defines the max number of photons to be counted
    counts = declare(int)  # variable to save the total number of photons
    counts_st = declare_stream()  # stream for 'counts'
    adc_st = declare_stream(adc_trace=True)  # stream to save ADC data

    # Pulse sequence
    ################
    play("PL", "laser_EX705nm", duration=int(meas_len // 4))  # Photoluminescence
    measure("photon_count", "SNSPD", adc_st, time_tagging.analog(times, meas_len, counts))  # photon count on SNSPD
    save(counts, counts_st)  # save QUA variable to stream

    # Stream processing
    ###################
    with stream_processing():
        counts_st.save("counts")  # save last variable in the stream to 'counts'
        adc_st.input1().save("adc1_data")  # save ADC data to 'adc2_data'

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    # simulation properties
    simulate_config = SimulationConfig(
        duration=int(2 * (meas_len // 4)),
        simulation_interface=LoopbackInterface(([("con1", 1, "con1", 1)])),
    )
    job = qmm.simulate(config, cal_tt, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    job = qm.execute(cal_tt)  # execute QUA program

    res_handle = job.result_handles  # get access to handles
    res_handle.wait_for_all_values()  # wait for all values before retrieving data

    counts = res_handle.get("counts").fetch_all()  # fetch all data related to counts_st
    adc1 = res_handle.get("adc1_data").fetch_all()  # fetch all data related to adc_st

    # plot raw ADC
    plt.figure()
    plt.plot(adc1)

    # print counts
    print(counts)

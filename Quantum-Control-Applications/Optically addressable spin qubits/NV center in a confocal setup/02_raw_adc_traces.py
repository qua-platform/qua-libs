"""
raw_adc_traces.py: a script used to look at the raw ADC data from inputs 1 and 2,
this allows checking that the ADC is not saturated, correct for DC offsets and define the time of flight and
threshold for time-tagging.
"""
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import matplotlib.pyplot as plt
from configuration import *
from qm import SimulationConfig

###################
# The QUA program #
###################
n_avg = 1_000

with program() as TimeTagging_calibration:
    n = declare(int)
    adc_st = declare_stream(adc_trace=True)
    with for_(n, 0, n < n_avg, n + 1):
        play("laser_ON", "AOM1")
        measure("long_readout", "SPCM1", adc_st)
        wait(1000, "SPCM1")

    with stream_processing():
        # Will save average:
        adc_st.input1().average().save("adc1")
        # Will save only last run:
        adc_st.input1().save("adc1_single_run")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, cluster_name=cluster_name)

simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, TimeTagging_calibration, simulation_config)
    plt.figure()
    job.get_simulated_samples().con1.plot()
else:
    # Open Quantum Machine
    qm = qmm.open_qm(config)
    # Execute program
    job = qm.execute(TimeTagging_calibration)
    # create a handle to get results
    res_handles = job.result_handles
    # Wait untill the program is done
    res_handles.wait_for_all_values()
    # Fetch results and convert traces to volts
    adc1 = u.raw2volts(res_handles.get("adc1").fetch_all())
    adc1_single_run = u.raw2volts(res_handles.get("adc1_single_run").fetch_all())
    # Plot data
    plt.figure()
    plt.subplot(121)
    plt.title("Single run")
    plt.plot(adc1_single_run, label="Input 1")
    plt.xlabel("Time [ns]")
    plt.ylabel("Signal amplitude [V]")
    plt.legend()

    plt.subplot(122)
    plt.title("Averaged run")
    plt.plot(adc1, label="Input 1")
    plt.xlabel("Time [ns]")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(f"\nInput1 mean: {np.mean(adc1)} V")

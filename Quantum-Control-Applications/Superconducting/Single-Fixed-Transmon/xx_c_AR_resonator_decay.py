# %%
"""
        RESONATOR DEPLETION TIME
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.simulate import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool, progress_counter
from macros import iq_blobs_analysis
import math
from qualang_tools.loops import from_array

###################
# The QUA program #
###################

n_avg = 1_000  # Number of runs
delay_min = 4
delay_max = 500
delay_step = 1
delays = np.arange(delay_min, delay_max, delay_step)

ramsey_idle_time = 200
# Time between populating the resonator and playing a Ramsey sequence in clock-cycles (4ns)

with program() as res_depletion_time:
    t = declare(int)
    n = declare(int)
    n_st  = declare_stream()
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        # Save the averaging iteration to get the progress bar
        save(n, n_st)
        with for_(*from_array(t, delays)):
            # Wait for the qubit to decay to the ground state
            wait(thermalization_time * u.ns)
            # Fill the resonator with photons
            measure(
                "midcircuit_readout",
                "resonator",
                None,
                dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
                dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q),
            )            
            align()
            play("zero", 'qubit', condition=I>-10)
            # Play a fixed duration Ramsey sequence after a varying time to estimate the effect of photons in the resonator
            wait(t)
            # Align the two elements to play the Ramsey sequence after having waited for a varying time "t".
            align()
            # Play the Ramsey sequence
            play("x90", 'qubit')
            wait(ramsey_idle_time // 4)  # fixed time ramsey
            play("x90", 'qubit')
            # Align the two elements to measure after playing the qubit pulse.
            align()
            # Measure the state of the resonator
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
                dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q),
            )
            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        n_st.save('iteration')
        # Save all streamed points for plotting the IQ blobs
        I_st.buffer(len(delays)).average().save("I")
        Q_st.buffer(len(delays)).average().save("Q")

# %%

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, res_depletion_time, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    try:
        # Open the quantum machine
        qm = qmm.open_qm(config, close_other_machines=False)
        print("Open QMs: ", qmm.list_open_quantum_machines())
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(res_depletion_time)

        fetch_names = ["iteration"]
        fetch_names.append("I")
        fetch_names.append("Q")
        results = fetching_tool(job, fetch_names, mode="live")

        while results.is_processing():
            # Fetch results
            res = results.fetch_all()
            # Progress bar
            progress_counter(res[0], n_avg, start_time=results.start_time)

            plt.plot(delays * 4, res[2])
            plt.ylabel('I-data')
            plt.xlabel('Delays [ns]')
            
            plt.tight_layout()
            plt.pause(1)
        
    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show(block=False)
# %%

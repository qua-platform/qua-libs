# %%
"""
    Repeated Readout

"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import fetching_tool, progress_counter
import matplotlib.pyplot as plt
from scipy import signal

###################
# The QUA program #
###################

shots = 5_000  # The number of averages

with program() as repeated_readout:

    n = declare(int)  # QUA variable for the averaging loop
    n_st = declare_stream()
    n1 = declare(int)  # QUA variable for the averaging loop
    adc_st = [declare_stream(adc_trace=True) for _ in range(2)]

    wait((lock_in_readout_length + 16) * u.ns, 'QDS_twin')

    with for_(n, 0, n < shots, n + 1):  # QUA for_ loop for averaging
        measure('readout', 'QDS', adc_st[0])
        save(n, n_st)
        wait(lock_in_readout_length * u.ns, 'QDS')

    with for_(n1, 0, n1 < shots, n1 + 1):  # QUA for_ loop for averaging
        measure('readout', 'QDS_twin', adc_st[1])
        wait(lock_in_readout_length * u.ns, 'QDS_twin')

    
    with stream_processing():
        n_st.save('iteration')
        for ind in range(2):
            adc_st[ind].input2().buffer(shots).save(f"adc_{ind}")

if True:

    from qm import generate_qua_script
    from pprint import pprint

    pprint(generate_qua_script(repeated_readout))

# %%
#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, repeated_readout, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show(block=False)

else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    print("Open QMs: ", qmm.list_open_quantum_machines())

    if lock_in_readout_length >= 1_000 and shots <= 5_000:

        job = qm.execute(repeated_readout)

        fetch_names = ['iteration']

        results = fetching_tool(job, fetch_names, mode="live")

        while results.is_processing():

            res = results.fetch_all()

            progress_counter(res[0], shots, start_time=results.start_time)

        for ind in range(2):
            fetch_names.append(f"adc_{ind}")

        results = fetching_tool(job, fetch_names)
        res = results.fetch_all()

        complete_adc = np.empty(((len(res[1]) + len(res[2])), lock_in_readout_length), dtype=res[1][0][0])

        complete_adc[0::2] = res[1]
        complete_adc[1::2] = res[2]

        f, pxx = signal.periodogram(complete_adc.flatten(), fs=1e9)

        plt.plot(f, pxx)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [a.u.]')

    else:
        print("Lock in readout length is less than 1 microsecond or shots > 10 million")

    qm.close()
    print("Experiment QM is now closed")
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up


# %%

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
import h5py
from scipy import signal
from qualang_tools.addons.variables import assign_variables_to_element

###################
# The QUA program #
###################

shots = 1_000_000  # The number of averages

with program() as repeated_readout:

    n = declare(int)  # QUA variable for the averaging loop
    n_st = declare_stream()
    n1 = declare(int)  # QUA variable for the averaging loop
    # Here we define one 'I', 'Q', 'I_st' & 'Q_st' for each resonator via a python list
    I = [declare(fixed) for _ in range(2)]
    Q = [declare(fixed) for _ in range(2)]
    I_st = [declare_stream() for _ in range(2)]
    Q_st = [declare_stream() for _ in range(2)]
    
    assign_variables_to_element('QDS', n, I[0], Q[0])
    assign_variables_to_element('QDS_twin', n1, I[1], Q[1])

    align()

    wait((lock_in_readout_length + 16) * u.ns, 'QDS_twin')  # needed to delay second for_loop

    with for_(n, 0, n < shots, n + 1):  # QUA for_ loop for averaging
        measure('readout', 'QDS', None, demod.full("cos", I[0], 'out2'), demod.full("sin", Q[0], 'out2'))
        save(I[0], I_st[0])
        save(Q[0], Q_st[0])
        save(n, n_st)
        wait(lock_in_readout_length * u.ns, 'QDS')

    with for_(n1, 0, n1 < shots, n1 + 1):  # QUA for_ loop for averaging
        measure('readout', 'QDS_twin', None, demod.full("cos", I[1], 'out2'), demod.full("sin", Q[1], 'out2'))
        save(I[1], I_st[1])
        save(Q[1], Q_st[1])
        wait(lock_in_readout_length * u.ns, 'QDS_twin')

    
    with stream_processing():
        n_st.save('iteration')
        for ind in range(2):
            I_st[ind].buffer(shots).save(f"I_{ind}")
            Q_st[ind].buffer(shots).save(f"Q_{ind}")


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

    if lock_in_readout_length >= 1_000 and shots <= 10_000_000:

        job = qm.execute(repeated_readout)

        fetch_names = ['iteration']

        results = fetching_tool(job, fetch_names, mode="live")

        while results.is_processing():

            res = results.fetch_all()

            progress_counter(res[0], shots, start_time=results.start_time)

        for ind in range(2):
            fetch_names.append(f"I_{ind}")
            fetch_names.append(f"Q_{ind}")

        results = fetching_tool(job, fetch_names)
        res = results.fetch_all()

        complete_I = np.empty((res[1].size + res[3].size), dtype=res[1][0])
        complete_Q = np.empty((res[2].size + res[4].size), dtype=res[2][0])

        complete_I[0::2] = res[1]
        complete_I[1::2] = res[3]

        complete_Q[0::2] = res[2]
        complete_Q[1::2] = res[4]

        complete_Z = complete_I + 1j*complete_Q

        phase = np.unwrap(np.angle(complete_Z))
        phase -= np.mean(phase)
        f, pxx = signal.welch(phase, nperseg=int(len(phase)/32), fs=1e9/lock_in_readout_length)

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

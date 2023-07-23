"""
storage_spec.py: Performs a 1D frequency sweep on the storage, measuring the resonator
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from qm import SimulationConfig
from quam import QuAM

machine = QuAM("quam_bootstrap_state.json", flat_data=False)
resonator = machine.resonators[0]
qubit = machine.qubits[0]
storage = machine.storage[0]

###################
# The QUA program #
###################

n_avg = 10

cooldown_time = int(5 * storage.T1 * 1e9 // 4)

f_min = storage.frequency - storage.lo - 20e6
f_max = storage.frequency - storage.lo + 20e6
df = 0.1e6
freqs = np.arange(f_min, f_max + 0.1, df)  # + 0.1 to add f_max to freqs

with program() as storage_spec:
    n = declare(int)
    n_st = declare_stream()
    f = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(f, f_min, f <= f_max, f + df):  # Notice it's <= to include f_max (This is only for integers!)
            # to the storage
            update_frequency(storage.name, f)
            play("saturation", storage.name)
            align(storage.name, qubit.name)
            # to the qubit
            play("saturation", qubit.name)
            align(qubit.name, resonator.name)
            measure(
                "readout",
                resonator.name,
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
            )
            save(I, I_st)
            save(Q, Q_st)
            wait(cooldown_time)
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(freqs)).average().save("I")
        Q_st.buffer(len(freqs)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.opx_ip, port=83)

###########################
# Run or Simulate Program #
###########################

simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=1000)
    job = qmm.simulate(build_config(machine), storage_spec, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(build_config(machine))

    job = qm.execute(storage_spec)
    # Get results from QUA program
    # res_handles = job.result_handles
    # I_handle = res_handles.get("I")
    # Q_handle = res_handles.get("Q")
    # iteration_handle = res_handles.get("iteration")
    # I_handle.wait_for_values(1)
    # Q_handle.wait_for_values(1)
    # iteration_handle.wait_for_values(1)
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        plt.cla()
        # I = I_handle.fetch_all()
        # Q = Q_handle.fetch_all()
        # iteration = iteration_handle.fetch_all()
        I, Q, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.subplot(211)
        plt.cla()
        plt.title("storage spectroscopy amplitude")
        plt.plot(freqs / u.MHz, np.sqrt(I**2 + Q**2), ".")
        plt.xlabel("frequency [MHz]")
        plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [a.u.]")
        plt.subplot(212)
        plt.cla()
        # detrend removes the linear increase of phase
        phase = signal.detrend(np.unwrap(np.angle(I + 1j * Q)))
        plt.title("storage spectroscopy phase")
        plt.plot(freqs / u.MHz, phase, ".")
        plt.xlabel("frequency [MHz]")
        plt.ylabel("Phase [rad]")
        plt.pause(0.1)
        plt.tight_layout()

# update parameters
#########################
ready = False

storage.frequency = storage.lo + 10e6
if ready:
    machine._save("quam_bootstrap_state.json", flat_data=False)

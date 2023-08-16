from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from qualang_tools.loops import from_array

##############################
# Program-specific variables #
##############################

n_avg = 3000  # Number of averaging loops

cooldown_time = 2 * u.us

# Frequency sweep in Hz
f_min = 55 * u.MHz
f_max = 65 * u.MHz
df = 50 * u.kHz
freqs = np.arange(f_min, f_max + df / 2, df)  # +df/2 to add f_max to the scan

###################
# The QUA program #
###################

with program() as qubit_spec:
    n = declare(int)  # Averaging index
    f = declare(int)  # Resonator frequency
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    n_st = declare_stream()

    # Adjust the flux line if needed
    # set_dc_offset("flux_line", "single", 0.0)
    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(f, freqs)):
            # Update the qubit frequency
            update_frequency("qubit", f)
            # Play a saturation pulse on the qubit
            play("cw", "qubit")
            align("qubit", "resonator")
            # Measure the resonator
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
            )
            # Wait for the resonator to cooldown
            wait(cooldown_time * u.ns, "resonator")
            # Save data to the stream processing
            save(I, I_st)
            save(Q, Q_st)
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(freqs)).average().save("I")
        Q_st.buffer(len(freqs)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, qop_port, octave=octave_config)

simulation = False
if simulation:
    simulation_config = SimulationConfig(
        duration=8000, simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])
    )
    job = qmm.simulate(config, qubit_spec, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(qubit_spec)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # 1D spectroscopy plot
        plt.subplot(211)
        plt.cla()
        plt.title("Qubit spectroscopy amplitude")
        plt.plot(freqs / u.MHz, np.sqrt(I**2 + Q**2), ".")
        plt.xlabel("qubit frequency [MHz]")
        plt.subplot(212)
        plt.cla()
        # detrend removes the linear increase of phase
        phase = signal.detrend(np.unwrap(np.angle(I + 1j * Q)))
        plt.title("Qubit spectroscopy phase")
        plt.plot(freqs / u.MHz, phase, ".")
        plt.xlabel("qubit frequency [MHz]")
        plt.pause(0.1)
        plt.tight_layout()
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from qm import SimulationConfig
from qualang_tools.loops import from_array

###################
# The QUA program #
###################

n_avg = 100

cooldown_time = 10 * u.us // 4

f_min = 30e6
f_max = 70e6
df = 0.5e6
freqs = np.arange(f_min, f_max + 0.1, df)  # + 0.1 to add f_max to freqs
# Readout amplitude sweep (as a pre-factor of the readout amplitude)
a_min = 0
a_max = 2
da = 0.01
amplitude = np.arange(a_min, a_max + da / 2, da)  # +da/2 to add a_max to the scan

with program() as resonator_spec_2D:
    n = declare(int)  # Averaging index
    f = declare(int)  # Resonator frequency
    a = declare(fixed)  # Readout amplitude pre-factor
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    n_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(f, freqs)):
            # Update the resonator frequency
            update_frequency("resonator", f)
            with for_(a, a_min, a < a_max + da / 2, a + da):  # Notice it's < a_max + da/2 to include a_max
                # Measure the resonator
                measure(
                    "readout" * amp(a),
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                # Wait for the resonator to cooldown
                wait(cooldown_time, "resonator")
                # Save data to the stream processing
                save(I, I_st)
                save(Q, Q_st)
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(amplitude)).buffer(len(freqs)).average().save("I")
        Q_st.buffer(len(amplitude)).buffer(len(freqs)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, qop_port, octave=octave_config)

#######################
# Simulate or execute #
#######################

simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=1000)
    job = qmm.simulate(config, resonator_spec_2D, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(resonator_spec_2D)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Normalize data
        s1 = u.demod2volts(I + 1j * Q, readout_len)
        A1 = np.abs(s1)
        row_sums = A1.sum(axis=0)
        A1 = A1 / row_sums[np.newaxis, :]
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # 2D spectroscopy plot
        plt.subplot(211)
        plt.cla()
        plt.title("resonator spectroscopy amplitude (normalized)")
        plt.pcolor(amplitude * readout_amp, freqs / u.MHz, A1)
        plt.ylabel("frequency [MHz]")
        plt.xlabel("readout amplitude [V]")
        plt.subplot(212)
        plt.cla()
        plt.title("resonator spectroscopy phase")
        plt.pcolor(amplitude * readout_amp, freqs / u.MHz, signal.detrend(np.unwrap(np.angle(I + 1j * Q))))
        plt.ylabel("frequency [MHz]")
        plt.xlabel("readout amplitude [V]")
        plt.pause(0.1)
        plt.tight_layout()

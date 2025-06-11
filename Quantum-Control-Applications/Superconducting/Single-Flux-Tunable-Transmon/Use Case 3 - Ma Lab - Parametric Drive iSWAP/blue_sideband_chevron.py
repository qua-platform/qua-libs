from qm import SimulationConfig
from qm.qua import *
from qm import QuantumMachinesManager
from configuration import *
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
import matplotlib.pyplot as plt
import numpy as np


###############
# QUA program #
###############

t_min = 16 // 4  # in units of clock cycles
t_max = 1200 // 4  # in units of clock cycles
dt = 8 // 4
times = np.arange(t_min, t_max + dt / 2, dt)

# sideband modulation freq
f_min = 95e6
f_max = 110e6
df = 0.1e6
freqs = np.arange(f_min, f_max + df / 2, df)  # + df/2 to add f_max to freqs

n_avg = 50

cooldown_time = 100000 // 4


with program() as blue_sideband:
    # Declare QUA variables
    ###################
    f = declare(int)  # variable for freqs sweep
    n = declare(int)  # variable for average loop
    tau = declare(int)  # Variable for the flux pulse duration sweep
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'
    a = declare(fixed, value=bs_a)  # Update the amplitude of the flux pulse directly in QUA

    # Pulse sequence
    ################
    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(tau, times)):
            with for_(*from_array(f, freqs)):
                # update the frequency of the flux line
                update_frequency("flux_line_bs", f)
                # Wait for the qubit to decay
                wait(cooldown_time, "qubit1", "flux_line_bs")
                # Play the blue sideband after the qubit pulse with varying duration
                align("qubit1", "flux_line_bs")
                play("const" * amp(a), "flux_line_bs", duration=tau)
                # Measure the state of the resonator after the flux pulse
                align("resonator1", "flux_line_bs")
                measure(
                    "readout",
                    "resonator1",
                    None,
                    dual_demod.full("cos", "sin", I),
                    dual_demod.full("minus_sin", "cos", Q),
                )
                # Save the 'I' & 'Q' quadratures to their respective streams
                save(I, I_st)
                save(Q, Q_st)
                # Save the averaging iteration to get the progress bar
            save(n, n_st)

    # Stream processing
    ###################
    with stream_processing():
        # Cast the data into a 2D matrix, average the 2D matrices together and store the results on the OPX processor
        I_st.buffer(len(freqs)).buffer(len(times)).average().save("I")
        Q_st.buffer(len(freqs)).buffer(len(times)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################

simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=100_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, blue_sideband, simulation_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(blue_sideband)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.cla()
        plt.title("Blue sideband chevron pattern")
        plt.pcolor(freqs, times * 4, I)
        plt.xlabel("Modulation frequency [Hz]")
        plt.ylabel("Variable pulse length [ns]")
        plt.pause(0.1)

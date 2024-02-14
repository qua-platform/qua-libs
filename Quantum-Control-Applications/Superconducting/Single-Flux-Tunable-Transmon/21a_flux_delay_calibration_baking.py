# %%
"""
        FLUX LINE DELAY CALIBRATION
The goal of this protocol is to measure the delay between pulses send through the flux line and
the qubit drive line. Because of the intrinsic wiring, delays in the order of nanoseconds
can exists in between the flux and qubit drive line.

The sequence consists of having a flux pulse with the same length as the x180 qubit pulse. Then the idea
is to fix the position of the x180 pulse in time, and scan de distance between the center of the pulses.
According to this protocl, one sets the flux pulse to start quite before the x180 pulse, in the middle
part of the sequence the flux pulse overlaps completely with the x180 pulse, and in the later
part of the sequence the flux pulse occurs after the x180 pulse.

The results are then post-processed to retrieve the delay between a pulse traveling through the flux line 
and the XY qubit drive line.

The protocol is inspired from https://web.physics.ucsb.edu/~martinisgroup/theses/Chen2018.pdf, page 109, 
which contains more details about the sequence.

This version sweeps the flux pulse position with respect to a pi-pulse using the baking tool for both the
flux and the qubit pulse. Doing so enables the ability to do 1 ns steps during a QUA program, but the total waveform
length of flux pulse and zeros before and after must be shorter than ~260ns. If you want to measure longer flux pulse, 
you can either reduce the resolution (do 2ns steps instead of 1ns) or use the other variant which is to
modify the config dictionary before the quantum machine is open (21b_flux_delay_calibration_analog_delay.py).

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit gates (x90 and y90) by running qubit spectroscopy, rabi_chevron, power_rabi, Ramsey and updated the configuration.
    - OPTIONAL: Having correceted the flux-pulse distortion with cryoscope

Next steps before going to the next node:
    - Update the analog delay parameter in the controllers dictionary in the configuration (config/controllers/con1/analog_outputs/"delay": measured_delay).
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.bakery import baking
from macros import ge_averaged_measurement
from scipy import signal, optimize
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

###################
# The QUA program #
###################
n_avg = 1_000  # Number of averages
# Flag to set to True if state discrimination is calibrated (where the qubit state is inferred from the 'I' quadrature).
# Otherwise, a preliminary sequence will be played to measure the averaged I and Q values when the qubit is in |g> and |e>.
state_discrimination = True
# FLux pulse waveform generation
# The zeros are just here to visualize the rising and falling times of the flux pulse. they need to be set to 0 before
# fitting the step response with an exponential.
zeros_before_pulse = x180_len + 20  # Beginning of the flux pulse (before we put zeros to see the rising time)
zeros_after_pulse = x180_len + 20  # End of the flux pulse (after we put zeros to see the falling time)
total_zeros = zeros_after_pulse + zeros_before_pulse
const_amp = 0.1
const_len = x180_len
flux_waveform = [const_amp] * x180_len

# %%


def baked_waveform(waveform):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    for i in range(0, zeros_before_pulse + zeros_after_pulse):
        with baking(config, padding_method="none") as b:
            wf = [0.0] * i + waveform + [0.0] * (zeros_after_pulse + zeros_before_pulse - i)
            I_wf = [0.0] * zeros_before_pulse + x180_I_wf.tolist() + [0.0] * zeros_after_pulse
            Q_wf = [0.0] * zeros_before_pulse + x180_Q_wf.tolist() + [0.0] * zeros_after_pulse
            b.add_op("flux_pulse", "flux_line", wf)
            b.add_op("x180", "qubit", [I_wf, Q_wf])
            b.play("flux_pulse", "flux_line")
            b.play("x180", "qubit")
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)
    return pulse_segments


delay_segments = baked_waveform(flux_waveform)
xplot = np.arange(-zeros_before_pulse, zeros_after_pulse, 1)  # x-axis for plotting - Must be in ns.
number_of_segments = zeros_after_pulse + zeros_before_pulse

# %%

with program() as delay_calibration:
    n = declare(int)  # QUA variable for the averaging loop
    segment = declare(int)  # QUA variable for the flux pulse segment index
    flag = declare(bool)  # QUA boolean to switch between x90 and y90
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    if state_discrimination:
        state = declare(bool)
        state_st = declare_stream()
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    if not state_discrimination:
        # Calibrate the ground and excited states readout for deriving the Bloch vector
        # The ge_averaged_measurement() function is defined in macros.py
        # Note that if you have calibrated the readout to perform state discrimination, then the QUA program below can
        # be modified to directly fetch the qubit state.
        Ig_st, Qg_st, Ie_st, Qe_st = ge_averaged_measurement(thermalization_time, n_avg)

    # Outer loop for averaging
    with for_(n, 0, n < n_avg, n + 1):
        # Loop over the truncated flux pulse
        with for_(segment, 0, segment < number_of_segments, segment + 1):
            # Alternate between X and I pulses
            with for_each_(flag, [True, False]):
                # Play second X or I
                with if_(flag):
                    play("x180", "qubit")
                with else_():
                    wait(x180_len, "qubit")
                align("qubit", "flux_line")
                # Play truncated flux pulse
                with switch_(segment):
                    for j in range(0, number_of_segments):
                        with case_(j):
                            delay_segments[j].run()
                align("resonator", "qubit", "flux_line")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q),
                )
                # State discrimination if the readout has been calibrated
                if state_discrimination:
                    assign(state, I > ge_threshold)
                    save(state, state_st)

                # Wait cooldown time and save the results
                wait(thermalization_time * u.ns)
                save(I, I_st)
                save(Q, Q_st)
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 2D matrix (x180/I, delay length), average the 2D matrices together and store the
        # results on the OPX processor
        I_st.buffer(2).buffer(number_of_segments).average().save("I")
        Q_st.buffer(2).buffer(number_of_segments).average().save("Q")
        if state_discrimination:
            # Also save the qubit state
            state_st.boolean_to_int().buffer(2).buffer(number_of_segments).average().save("state")
        else:
            # Also save the averaged I/Q values for the qubit in |g> and |e>
            Ig_st.average().save("Ig")
            Qg_st.average().save("Qg")
            Ie_st.average().save("Ie")
            Qe_st.average().save("Qe")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, delay_calibration, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(delay_calibration)
    # Get results from QUA program
    if state_discrimination:
        results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    else:
        results = fetching_tool(job, data_list=["I", "Q", "Ie", "Qe", "Ig", "Qg", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        if state_discrimination:
            I, Q, state, iteration = results.fetch_all()
            # Convert the results into Volts
            I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
            # Bloch vector Sx + iSy
            e_preparation = state[:, 0] * 2 - 1
            g_preparation = state[:, 1] * 2 - 1
        else:
            I, Q, Ie, Qe, Ig, Qg, iteration = results.fetch_all()
            # Phase of ground and excited states
            phase_g = np.angle(Ig + 1j * Qg)
            phase_e = np.angle(Ie + 1j * Qe)
            # Phase of delay_calibration measurement
            phase = np.unwrap(np.angle(I + 1j * Q))
            # Population in excited state
            state = (phase - phase_g) / (phase_e - phase_g)
            # Convert the results into Volts
            I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
            # Bloch vector Sx + iSy
            e_preparation = state[:, 0] * 2 - 1
            g_preparation = state[:, 1] * 2 - 1

        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

        # Plots
        plt.cla()
        plt.plot(xplot, e_preparation, label="|e>")
        plt.plot(xplot, g_preparation, label="|g>")
        plt.ylabel("State")
        plt.xlabel("Timing Delay[ns]")
        plt.legend()
        plt.pause(0.1)
        plt.tight_layout()

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

# %%

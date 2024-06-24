"""
SWAP_spectroscopy.py: program performing a SWAP spectroscopy used to calibrate the CZ gate.
"""

from qm.qua import *
from configuration import *
from qm import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.bakery import baking
from qualang_tools.loops import from_array


##############################
# Program-specific variables #
##############################
conditional_qubit = 1  # this line decides on which qubit to apply flux to
# Choose relevant elements
# The flux amplitude is chosen to reach the 02-11 avoided crossing found by performing a flux versus frequency spectroscopy
if conditional_qubit == 0:
    flux = f"flux_line{0}"
    res_name = f"resonator{1}"
    flux_amp = 0.2
else:
    flux = f"flux_line{1}"
    res_name = f"resonator{0}"
    flux_amp = 0.2

# FLux pulse waveform generation
flux_waveform = np.array([flux_amp] * const_flux_len)  # The variable const_flux_len is defined in the configuration


def baked_waveform(waveform, pulse_duration):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    for i in range(0, pulse_duration + 1):
        with baking(config, padding_method="right") as b:
            if i == 0:  # Otherwise, the baking will be empty and will not be created
                wf = [0.0] * 16
            else:
                wf = waveform[:i].tolist()
            b.add_op("flux_pulse", flux, wf)
            b.play("flux_pulse", flux)
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)
    return pulse_segments


# Baked flux pulse segments
square_pulse_segments = baked_waveform(flux_waveform, const_flux_len)

# FLux pulse amplitude pre-factor
a_min = 0.48
a_max = 0.60
da = 0.001
amps = np.arange(a_min, a_max + da / 2, da)

# Qubit cooldown time
cooldown_time = 5 * qubit_T1 // 4
# Number of averages
n_avg = 1e3 / 2


###################
# The QUA program #
###################
with program() as SWAP_spectroscopy:
    n = declare(int)  # Variable for averaging
    n_st = declare_stream()
    I = declare(fixed)  # I quadrature for state measurement
    Q = declare(fixed)  # Q quadrature for state measurement
    state = declare(bool)  # Qubit state
    state_st = declare_stream()
    a = declare(fixed)  # Flux pulse amplitude
    segment = declare(int)  # Flux pulse segment

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(a, amps)):
            # Notice it's <= to include t_max (This is only for integers!)
            with for_(segment, 0, segment <= const_flux_len, segment + 1):
                # Cooldown to have the qubit in the ground state
                wait(cooldown_time)
                # CZ 02-11 protocol
                # Play pi on both qubits
                play("x180", "qubit0")
                play("x180", "qubit1")
                # global align
                align()
                # Wait some additional time to be sure that the pulses don't overlap, this can be calibrated
                wait(20)
                # Play flux pulse with 1ns resolution
                with switch_(segment):
                    for j in range(0, const_flux_len + 1):
                        with case_(j):
                            square_pulse_segments[j].run(amp_array=[("flux_line1", a)])
                # global align
                align()
                # Wait some additional time to be sure that the pulses don't overlap, this can be calibrated
                wait(20)
                # q0 state readout
                measure(
                    "readout",
                    res_name,
                    None,
                    dual_demod.full("rotated_cos", "rotated_sin", I),
                    dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
                )
                # State discrimination
                assign(state, I > ge_threshold)
                save(state, state_st)

        save(n, n_st)

    with stream_processing():
        state_st.boolean_to_int().buffer(const_flux_len + 1).buffer(len(amps)).average().save("state")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host="192.168.88.10", port="80")
# Open quantum machine
qm = qmm.open_qm(config)
# Execute QUA program
job = qm.execute(SWAP_spectroscopy)
# Get results from QUA program
results = fetching_tool(job, data_list=["state", "iteration"], mode="live")
# Live plotting
fig = plt.figure()
interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
xplot = np.arange(0, const_flux_len + 0.1, 1)
while results.is_processing():
    # Fetch results
    state, iteration = results.fetch_all()
    # Progress bar
    progress_counter(iteration, n_avg, start_time=results.get_start_time())
    # Plot results
    plt.cla()
    plt.pcolor(xplot, amps * flux_amp, state, cmap="magma")
    plt.xlabel("Flux pulse time [ns]")
    plt.ylabel("Flux voltage wrt sweet spot [V]")
    plt.title("02-11 conditional on q1, measurement on q0")

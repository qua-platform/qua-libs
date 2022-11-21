"""
SWAP_spectroscopy.py: program performing a SWAP spectroscopy used to calibrate the CZ gate.
"""

from qm.qua import *
from quam import QuAM
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from qualang_tools.bakery import baking
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array


##################
# State and QuAM #
##################
experiment = "readout_opt"
debug = True
simulate = True
qubit_list = [0, 1]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"
now = datetime.now()
now = now.strftime("%m%d%Y_%H%M%S")

config = machine.build_config(digital, qubit_list, gate_shape)

##############################
# Program-specific variables #
##############################
# FLux pulse amplitude pre-factor
span = 0.2
da = 0.001
# Number of averages
n_avg = 1e3

cz = machine.two_qubit_gates.CZ[0]
print(f"SWAP spectroscopy with target qubit {cz.target_qubit} and conditional qubit {cz.conditional_qubit}")

# The flux amplitude is chosen to reach the 02-11 avoided crossing found by performing a flux versus frequency spectroscopy
# FLux pulse waveform generation
flux_len = cz.flux_pulse.constant.length
flux_amp = cz.flux_pulse.constant.amplitude
flux_waveform = np.array([flux_amp] * flux_len)  # The variable flux_len is defined in the configuration


def baked_waveform(waveform, pulse_duration):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    for i in range(0, pulse_duration + 1):
        with baking(config, padding_method="right") as b:
            if i == 0:  # Otherwise, the baking will be empty and will not be created
                wf = [0.0] * 16
            else:
                wf = waveform[:i].tolist()
            b.add_op(cz.flux_pulse.constant.name, machine.qubits[cz.conditional_qubit].name + "_flux", wf)
            b.play(cz.flux_pulse.constant.name, machine.qubits[cz.conditional_qubit].name + "_flux")
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)
    return pulse_segments


# Baked flux pulse segments
square_pulse_segments = baked_waveform(flux_waveform, flux_len)
# Amplitude scan
amps = np.arange(flux_amp - span, flux_amp + span + da / 2, da)
# Qubit cooldown time
cooldown_time = int(5 * machine.qubits[cz.target_qubit].t1 * 1e9 // 4)


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
            with for_(segment, 0, segment <= flux_len, segment + 1):
                # Cooldown to have the qubit in the ground state
                # wait(cooldown_time)
                # CZ 02-11 protocol
                # Play pi on both qubits
                play("x180", machine.qubits[cz.conditional_qubit].name)
                play("x90", machine.qubits[cz.target_qubit].name)
                # global align
                align()
                # Wait some additional time to be sure that the pulses don't overlap, this can be calibrated
                wait(20)
                # Play flux pulse with 1ns resolution
                with switch_(segment):
                    for j in range(0, flux_len + 1):
                        with case_(j):
                            square_pulse_segments[j].run(
                                amp_array=[(machine.qubits[cz.conditional_qubit].name + "_flux", a)]
                            )
                # global align
                align()
                # Wait some additional time to be sure that the pulses don't overlap, this can be calibrated
                wait(20)
                # q0 state readout
                measure(
                    "readout",
                    machine.readout_resonators[cz.target_qubit].name,
                    None,
                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q),
                )
                # State discrimination
                assign(state, I > machine.readout_resonators[cz.target_qubit].ge_threshold)
                save(state, state_st)

        save(n, n_st)

    with stream_processing():
        state_st.boolean_to_int().buffer(flux_len + 1).buffer(len(amps)).average().save("state")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)

if simulate:
    simulation_config = SimulationConfig(duration=10000)
    job = qmm.simulate(config, SWAP_spectroscopy, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open quantum machine
    qm = qmm.open_qm(config)
    # Execute QUA program
    job = qm.execute(SWAP_spectroscopy)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["state", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    xplot = np.arange(0, flux_len + 0.1, 1)
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

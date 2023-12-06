"""
A simple sandbox to showcase different QUA functionalities during the installation.
"""
import matplotlib.pyplot as plt
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from qualang_tools.loops import from_array
from configuration import *
from scipy.optimize import minimize
from macros import RF_reflectometry_macro, DC_current_sensing_macro
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.bakery import baking
from typing import List, Any, Dict
from qm.qua._dsl import _ResultSource, _Variable, _Expression
from qm import generate_qua_script


###################
# The QUA program #
###################
durations = np.arange(1, 500, 200)
pi_levels = np.arange(0.21, 0.3, 0.01)

seq = OPX_background_sequence(config, ["P1_sticky", "P2_sticky"])
seq.add_points("initialization", level_init, duration_init)
seq.add_points("manipulation", level_manip, duration_manip)
seq.add_points("readout", level_readout, duration_readout)

# Bake the Rabi pulses
pi_list = []
pi_list_4ns = []
for t in range(16):  # Create the different baked sequences
    t = int(t)
    with baking(config, padding_method="left") as b:  # don't use padding to assure error if timing is incorrect
        if t == 0:
            wf1 = [0.0] * 16
            wf2 = [0.0] * 16
        else:
            wf1 = [0.25] * t
            wf2 = [0.25] * t

        # Add the baked operation to the config
        b.add_op("pi_baked", "P1", wf1)
        b.add_op("pi_baked", "P2", wf2)

        # Baked sequence
        b.wait(16 - t, "P1")
        b.wait(16 - t, "P2")
        b.play("pi_baked", "P1")  # Play the qubit pulse
        b.play("pi_baked", "P2")  # Play the qubit pulse
    if t < 4:
        with baking(config, padding_method="left") as b4ns:  # don't use padding to assure error if timing is incorrect
            wf1 = [0.25] * t
            wf2 = [0.25] * t

            # Add the baked operation to the config
            b4ns.add_op("pi_baked2", "P1", wf1)
            b4ns.add_op("pi_baked2", "P2", wf2)

            # Baked sequence
            b4ns.wait(32 - t, "P1")
            b4ns.wait(32 - t, "P2")
            b4ns.play("pi_baked2", "P1")  # Play the qubit pulse
            b4ns.play("pi_baked2", "P2")  # Play the qubit pulse

    # Append the baking object in the list to call it from the QUA program
    pi_list.append(b)
    if t < 4:
        pi_list_4ns.append(b4ns)


with program() as hello_qua:
    t = declare(int)
    t_cycles = declare(int)
    t_left_ns = declare(int)
    a = declare(fixed)
    # seq.add_step(voltage_point_name="readout")
    with for_(*from_array(a, pi_levels)):
        with for_(*from_array(t, durations)):
            with strict_timing_():
                # Navigate through the charge stability map
                seq.add_step(voltage_point_name="initialization", ramp_duration=None)
                seq.add_step(voltage_point_name="manipulation")
                seq.add_step(voltage_point_name="readout")
                seq.add_compensation_pulse(duration=duration_compensation_pulse)

            # Short qubit pulse: baking only
            with if_(t<=16):
                # switch case to select the baked waveform corresponding to the burst duration
                with switch_(t, unsafe=True):
                    for ii in range(16):
                        with case_(ii):
                            # Drive the singlet-triplet qubit using an exchange pulse at the end of the manipulation step
                            wait((duration_init + duration_manip) * u.ns - 4 - 9, "P1", "P2")
                            pi_list[ii].run(amp_array=[("P1", (a-level_manip[0]) * 4), ("P2", (-a-level_manip[1]) * 4)])
                            
            # Long qubit pulse: baking and play combined
            with else_():
                assign(t_cycles, t >> 2)  # Right shift by 2 is a quick way to divide by 4
                assign(t_left_ns, t - (t_cycles << 2))  # left shift by 2 is a quick way to multiply by 4
                # switch case to select the baked waveform corresponding to the burst duration
                with switch_(t_left_ns, unsafe=True):
                    for ii in range(4):
                        with case_(ii):
                            # Drive the singlet-triplet qubit using an exchange pulse at the end of the manipulation step
                            wait((duration_init + duration_manip) * u.ns - t_cycles - 4 - 30, "P1", "P2")
                            pi_list_4ns[ii].run(
                                amp_array=[("P1", (a - level_manip[0]) * 4), ("P2", (-a - level_manip[1]) * 4)])
                            play("step" * amp((a - level_manip[0]) * 4), "P1", duration=t_cycles)
                            play("step" * amp((-a - level_manip[1]) * 4), "P2", duration=t_cycles)

            # Measure the dot right after the qubit manipulation
            wait((duration_init + duration_manip) * u.ns, "tank_circuit", "TIA")
            I, Q, I_st, Q_st = RF_reflectometry_macro()
            dc_signal, dc_signal_st = DC_current_sensing_macro()

            # Ramp the background voltage to zero to avoid propagating floating point errors
            ramp_to_zero("P1_sticky")
            ramp_to_zero("P2_sticky")

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################

simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, hello_qua, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()
    plt.axhline(0.1, color="k", linestyle="--")
    plt.axhline(0.3, color="k", linestyle="--")
    plt.axhline(0.2, color="k", linestyle="--")
    plt.axhline(-0.1, color="k", linestyle="--")
    plt.axhline(-0.3, color="k", linestyle="--")
    plt.axhline(-0.2, color="k", linestyle="--")
    plt.yticks([-0.2, -0.3, -0.1, 0.0, 0.1, 0.3, 0.2], ["readout", "manip", "init", "0", "init", "manip", "readout"])
    plt.legend("")
    samples = job.get_simulated_samples()
    report = job.get_simulated_waveform_report()
    report.create_plot(samples, plot=True)
else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(hello_qua)

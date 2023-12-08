"""
A simple sandbox to showcase different QUA functionalities during the installation.
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qm import generate_qua_script

###################
# The QUA program #
###################
level_init = [0.1, -0.1]
level_manip = [0.2, -0.2]
level_readout = [0.1, -0.1]
duration_init = 200
duration_manip = 800
duration_readout = 500

# Add the relevant voltage points describing the "slow" sequence (no qubit pulse)
seq = OPX_background_sequence(config, ["P1_sticky", "P2_sticky"])
seq.add_points("initialization", level_init, duration_init)
seq.add_points("idle", level_manip, duration_manip)
seq.add_points("readout", level_readout, duration_readout)

with program() as hello_qua:
    t = declare(int, value=16)
    a = declare(fixed, value=0.2)
    i = declare(int)
    k = 16
    with for_(i, 0, i<200, i+1):
        assign(t, t+100)
        with strict_timing_():
        # for k in np.arange(16,240,4):

            seq.add_step(voltage_point_name="initialization")
            seq.add_step(voltage_point_name="idle", ramp_duration=k, duration=t)
            seq.add_step(voltage_point_name="readout", ramp_duration=k)
            seq.add_compensation_pulse(duration=2_000)
            seq.ramp_to_zero()


#####################################
#  Open Communication with the QOP  #
#####################################
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
    job.get_simulated_samples().con1.plot()
    plt.axhline(level_init[0], color="k", linestyle="--")
    plt.axhline(level_manip[0], color="k", linestyle="--")
    plt.axhline(level_readout[0], color="k", linestyle="--")
    plt.axhline(level_init[1], color="k", linestyle="--")
    plt.axhline(level_manip[1], color="k", linestyle="--")
    plt.axhline(level_readout[1], color="k", linestyle="--")
    plt.axhline(pi_half_amps[0], color="k", linestyle="--")
    plt.axhline(pi_half_amps[1], color="k", linestyle="--")
    plt.yticks(
        [level_readout[1], level_manip[1], level_init[1], 0.0, level_init[0], level_manip[0], level_readout[0]],
        ["readout", "manip", "init", "0", "init", "manip", "readout"],
    )
    plt.legend("")
    from macros import get_filtered_voltage
    get_filtered_voltage(job.get_simulated_samples().con1.analog["1"], 1e-9, 1e4, True)
else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(hello_qua)

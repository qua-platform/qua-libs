from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.bakery import baking
from scipy import signal, optimize
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
import warnings

warnings.filterwarnings("ignore")


####################
# Helper functions #
####################
def baked_waveform(waveform, pulse_duration):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    for i in range(0, pulse_duration + 1):
        with baking(config, padding_method="right") as b:
            if i == 0:  # Otherwise, the baking will be empty and will not be created
                wf = [0.0] * 16
            else:
                wf = waveform[:i].tolist()
            b.add_op("flux_pulse", "flux_line", wf)
            b.play("flux_pulse", "flux_line")
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)
    return pulse_segments


def baked_pi_half(fid, pi_amp, pi_len, element1, element2):
    baked_list_right = []  # Stores the baking objects
    baked_list_left = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    for i in range(32):
        with baking(config, padding_method=None) as bright:
            wf = [fid] * i + [pi_amp] * pi_len + [0] * (4 - (pi_len + i) % 4)
            bright.add_op("pi", element1, wf)
            bright.play("pi", element1)
        baked_list_right.append(bright)
        with baking(config, padding_method=None) as bleft:
            wf = [fid] * (4 - i % 4) + [0] * (4 - pi_len % 4) + [pi_amp] * pi_len + [0] * i
            bleft.add_op("pi", element2, wf)
            bleft.play("pi", element2)
        baked_list_left.append(bleft)

    return baked_list_left, baked_list_right


def baked_1st_pi_half(fid, pi_amp, pi_len, element):
    # Create the different baked sequences, each one corresponding to a different truncated duration
    with baking(config, padding_method=None) as bleft:
        if pi_len >= 16:
            wf = [fid] * (4 - pi_len % 4) + [pi_amp] * pi_len
        else:
            wf = [fid] * (16 - pi_len % 16) + [pi_amp] * pi_len
        bleft.add_op("pi", element, wf)
        bleft.play("pi", element)
    return bleft


###################
# The QUA program #
###################
n_avg = 1  # Number of averages
N = 20
pi_half_len = 15  # ns
pi_len = pi_half_len
idle_times = np.arange(10, 100, 11)  # ns

pi_half_left, pi_half_right = baked_pi_half(-0, 0.1, pi_len, "gate_2", "gate_1")
pi_left, pi_right = baked_pi_half(-0, 0.1, pi_len, "gate_2", "gate_1")

pi_half_1 = baked_1st_pi_half(-0, 0.1, pi_half_len, "gate_1")

with program() as cpmg:
    n = declare(int)  # QUA variable for the averaging loop
    segment = declare(int)  # QUA variable for the flux pulse segment index
    t = declare(int)  # QUA variable for the flux pulse segment index
    t_cycles = declare(int)  # QUA variable for the flux pulse segment index
    t_left_ns = declare(int)  # QUA variable for the flux pulse segment index
    wait_time = declare(int)  # QUA variable for the flux pulse segment index
    qua_delay = declare(int)
    gap = 0

    # Outer loop for averaging
    with for_(n, 0, n < n_avg, n + 1):
        # Loop over the truncated flux pulse
        # with for_(segment, 0, segment <= len(idle_times), segment + 1):
        with for_(*from_array(t, idle_times)):
            set_dc_offset("gate_2", "single", 0.3)

            with if_(t < 32):
                # align()
                # deterministic_run(short_ramsey_baking_list, t, unsafe=True)
                with switch_(t):
                    for j in range(32):
                        gap = 0
                        with case_(j):
                            set_dc_offset("gate_2", "single", 0.1)
                            pi_half_1.run()
                            with strict_timing_():
                                wait(4, "gate_2")
                                for k in range(N):
                                    pi_right[max(0, j - gap)].run()
                                    gap = 4 - (j - gap + pi_half_len) % 4
                                pi_half_right[j - gap].run()
                            align()
                            set_dc_offset("gate_2", "single", 0.3)

            with else_():
                assign(t_cycles, t >> 2)  # Right shift by 2 is a quick way to divide by 4
                assign(t_left_ns, 16 + t - (t_cycles << 2))  # left shift by 2 is a quick way to multiply by 4
                assign(t_cycles, t_cycles - 32 // 4)
                assign(wait_time, 4 + t_cycles)
                with switch_(t_left_ns, unsafe=True):
                    for j in range(16, 21):
                        gap = 0
                        with case_(j):
                            pi_half_1.run()
                            with strict_timing_():
                                for k in range(N):
                                    if k == 0:
                                        wait(wait_time + 4, "gate_2")
                                    else:
                                        wait(wait_time, "gate_2")
                                    pi_right[max(0, j - gap)].run()
                                    gap = 4 - (j - gap + pi_half_len) % 4
                                wait(wait_time, "gate_2")
                                pi_half_right[j - gap].run()
            align()
            wait(25)
            set_dc_offset("gate_1", "single", -0)


with program() as cpmg_python_for:
    n = declare(int)  # QUA variable for the averaging loop
    segment = declare(int)  # QUA variable for the flux pulse segment index
    t = declare(int)  # QUA variable for the flux pulse segment index
    t_cycles = declare(int)  # QUA variable for the flux pulse segment index
    t_left_ns = declare(int)  # QUA variable for the flux pulse segment index
    wait_time = declare(int)  # QUA variable for the flux pulse segment index
    qua_delay = declare(int)
    gap = 0

    # Outer loop for averaging
    with for_(n, 0, n < n_avg, n + 1):
        # Loop over the truncated flux pulse
        # with for_(segment, 0, segment <= len(idle_times), segment + 1):
        with for_(*from_array(t, idle_times)):
            with if_(t < 32):
                # deterministic_run(short_ramsey_baking_list, t, unsafe=True)
                with switch_(t):
                    for j in range(32):
                        gap = 0
                        with case_(j):
                            pi_half_1.run()
                            with strict_timing_():
                                wait(4, "gate_2")
                                for k in range(N):
                                    pi_right[max(0, j - gap)].run()
                                    gap = 4 - (j - gap + pi_half_len) % 4
                                pi_half_right[j - gap].run()
            with else_():
                assign(t_cycles, t >> 2)  # Right shift by 2 is a quick way to divide by 4
                assign(t_left_ns, 16 + t - (t_cycles << 2))  # left shift by 2 is a quick way to multiply by 4
                assign(t_cycles, t_cycles - 32 // 4)
                assign(wait_time, 4 + t_cycles)
                with switch_(t_left_ns, unsafe=True):
                    for j in range(32):
                        gap = 0
                        with case_(j):
                            pi_half_1.run()
                            with strict_timing_():
                                for k in range(N):
                                    if k == 0:
                                        wait(wait_time + 4, "gate_2")
                                    else:
                                        wait(wait_time, "gate_2")
                                    pi_right[max(0, j - gap)].run()
                                    gap = 4 - (j - gap + pi_half_len) % 4
                                wait(wait_time, "gate_2")
                                pi_half_right[j - gap].run()


with program() as ramsey:
    n = declare(int)  # QUA variable for the averaging loop
    segment = declare(int)  # QUA variable for the flux pulse segment index
    t = declare(int)  # QUA variable for the flux pulse segment index
    t_cycles = declare(int)  # QUA variable for the flux pulse segment index
    t_left_ns = declare(int)  # QUA variable for the flux pulse segment index
    qua_delay = declare(int)

    # Outer loop for averaging
    with for_(n, 0, n < n_avg, n + 1):
        # Loop over the truncated flux pulse
        # with for_(segment, 0, segment <= len(idle_times), segment + 1):
        with for_(*from_array(t, idle_times)):
            with if_(t < 16):
                # deterministic_run(short_ramsey_baking_list, t, unsafe=True)
                with switch_(t):
                    for j in range(16):
                        with case_(j):
                            # align()
                            pi_half_1.run()
                            wait(4, "gate_2")
                            pi_half_right[j].run()
            with else_():
                assign(t_cycles, t >> 2)  # Right shift by 2 is a quick way to divide by 4
                assign(t_left_ns, t - (t_cycles << 2))  # left shift by 2 is a quick way to multiply by 4
                assign(t_cycles, t_cycles - 16 // 4)
                with switch_(t_left_ns, unsafe=True):
                    for j in range(4):
                        with case_(j):
                            pi_half_1.run()
                            # long_ramsey_1st_pulse_baking_list[j].run()
                            wait(8 + t_cycles, "gate_2")
                            pi_half_right[j].run()

            # with switch_(segment):
            #     for j in range(0, len(idle_times) + 1):
            #         with case_(j):
            #             # align()
            #             pi_half_1.run()
            #             wait(4, "gate_2")
            #             pi_half_right[j].run()
            # Wait cooldown time and save the results

#####################################
#  Open Communication with the QOP  #
#####################################

qmm = QuantumMachinesManager(host="172.16.33.101", cluster_name="Cluster_83")

###########################
# Run or Simulate Program #
###########################
# simulate = True
#
# if simulate:
#     # Simulates the QUA program for the specified duration
simulation_config = SimulationConfig(duration=1_000)  # In clock cycles = 4ns
job = qmm.simulate(config, cpmg, simulation_config)
job.get_simulated_samples().con1.plot()
# else:
#     # Open the quantum machine
#     qm = qmm.open_qm(config)
#     # Send the QUA program to the OPX, which compiles and executes it
#     job = qm.execute(cryoscope)
#     # Get results from QUA program
#     # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
#     qm.close()

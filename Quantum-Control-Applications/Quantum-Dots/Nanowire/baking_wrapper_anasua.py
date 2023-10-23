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
def baked_pi_half(fid, pi_amp, zero_pad, pi_len, element1, element2):
    baked_list_right = []  # Stores the baking objects
    baked_list_left = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    for i in range(48):
        with baking(config, padding_method=None) as bright:
            if (pi_len + i) < 16:
                wf = [fid] * i + [pi_amp] * pi_len + [zero_pad] * (16 - (pi_len + i) % 16)
            else:
                wf = [fid] * i + [pi_amp] * pi_len + [zero_pad] * (4 - (pi_len + i) % 4)
            bright.add_op("pi", element1, wf)
            bright.play("pi", element1)
        baked_list_right.append(bright)
        with baking(config, padding_method=None) as bleft:
            wf = [zero_pad] * (4 - i % 4) + [fid] * (4 - pi_len % 4) + [pi_amp] * pi_len + [fid] * i
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
pi_half_len = 8  # ns
pi_len = pi_half_len
idle_times = np.arange(16, 100, 8)  # ns multiples of 8ns if > 32ns
##################################
# ns multiples of 8ns if > 32ns
##################################

_, pi_right = baked_pi_half(-0.2, -0.1, -0.2, pi_len, "gate_2", "gate_1")
_, pi_right_last = baked_pi_half(-0.2, -0, -0, pi_len, "gate_2", "gate_1")


with program() as cpmg:
    n = declare(int)  # QUA variable for the averaging loop
    segment = declare(int)  # QUA variable for the flux pulse segment index
    t = declare(int)  # QUA variable for the flux pulse segment index
    t_cycles = declare(int)  # QUA variable for the flux pulse segment index
    t_left_ns = declare(int)  # QUA variable for the flux pulse segment index
    wait_time = declare(int)  # QUA variable for the flux pulse segment index
    wait_time_first = declare(int)  # QUA variable for the flux pulse segment index
    qua_delay = declare(int)
    gap = 0

    # Outer loop for averaging
    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(t, idle_times)):
            set_dc_offset("gate_2", "single", 0.3)
            with if_(t < 48):
                # deterministic_run(short_ramsey_baking_list, t, unsafe=True)
                with switch_(t):
                    for j in range(48):
                        gap = 0
                        with case_(j):
                            with strict_timing_():
                                pi_right[max(0, j // 2 - gap)].run()
                                if (j // 2 + pi_half_len) < 16:
                                    gap = 16 - (j // 2 - gap + pi_half_len) % 16
                                else:
                                    gap = 4 - (j // 2 - gap + pi_half_len) % 4
                                for k in range(N):
                                    pi_right[max(0, j - gap)].run()
                                    if (j + pi_half_len) < 16:
                                        gap = 4 - (j - gap + pi_half_len) % 4
                                    else:
                                        gap = 4 - (j - gap + pi_half_len) % 4
                                pi_right_last[max(0, j // 2 - gap)].run()
                            align()

            with else_():
                assign(t_cycles, t >> 2)  # Right shift by 2 is a quick way to divide by 4
                assign(t_left_ns, 16 + t - (t_cycles << 2))  # left shift by 2 is a quick way to multiply by 4
                assign(t_cycles, t_cycles - 32 // 4)
                assign(wait_time, 4 + t_cycles)
                assign(wait_time_first, (4 + t_cycles) >> 1)
                with switch_(t_left_ns, unsafe=True):
                    for j in range(16, 21):
                        gap = 0
                        with case_(j):
                            with strict_timing_():
                                # This could be replaced by just baked seg to avoid having tau multiple of 8
                                play("fid", "gate_2", duration=wait_time_first)
                                pi_right[max(0, j // 2 - gap)].run()
                                if (j // 2 + pi_half_len) < 16:
                                    gap = 16 - (j // 2 - gap + pi_half_len) % 16
                                else:
                                    gap = 4 - (j // 2 - gap + pi_half_len) % 4
                                for k in range(N):
                                    play("fid", "gate_2", duration=wait_time)
                                    pi_right[max(0, j - gap)].run()
                                    if (j + pi_half_len) < 16:
                                        gap = 4 - (j - gap + pi_half_len) % 4
                                    else:
                                        gap = 4 - (j - gap + pi_half_len) % 4
                                play("fid", "gate_2", duration=wait_time_first)
                                pi_right_last[max(0, j // 2 - gap)].run()
            align()
            wait(25)
            measure("readout", "TIA", None)
            align()
            set_dc_offset("gate_1", "single", -0)


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
simulation_config = SimulationConfig(duration=2_000)  # In clock cycles = 4ns
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

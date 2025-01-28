from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig, generate_qua_script
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.results.data_handler import DataHandler
from configuration import *
from TwoQ_RB_Sequence_Generation_CNOT_CR import *
from macros import qua_declaration, multiplexed_readout
import time


root_directory = "C:\\Users\\name\\Desktop\\QM_RB\\data_q1q2"  # change to your local directory
data_handler = DataHandler(root_data_folder=root_directory)
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

##############################
# Program-specific variables #
##############################

# Qubits and resonators
qc = 1  # index of control qubit
qt = 2  # index of target qubit

# Derived parameters
qc_xy = f"q{qc}_xy"
qt_xy = f"q{qt}_xy"
cr_drive = f"cr_drive_c{qc}t{qt}"
cr_cancel = f"cr_cancel_c{qc}t{qt}"
qubits = [f"q{i}_xy" for i in [qc, qt]]
resonators = [f"rr{i}" for i in [qc, qt]]
seed = 0

cr_drive_amp = 0.95  # scaling factor
cr_drive_phase = 0.1925  # in units of 2pi
cr_cancel_amp = 0.164  # scaling factor
cr_cancel_phase = -0.008  # in units of 2pi

threshold_1 = ge_threshold_q1  # set the threshold from the configuration
threshold_2 = ge_threshold_q2  # set the threshold from the configuration

# Random circuit generation
num_of_sequences = 20  # Number of random sequences
n_avg = 200  # Number of averaging loops for each random sequence
depth_list = [0, 1, 5, 7, 14, 20, 26, 32]
np.random.seed(seed=int(time.time() % 1 * 1e8))
# np.random.seed(seed = 0) # can set the seed for consistent comparison
sequence_list, len_list = pre_generate_sequence(num_of_sequences, depth_list)  # standard RB
# sequence_list, len_list = pre_generate_sequence_interleaved(num_of_sequences, depth_list) # interleaved RB

# ###### TEST EXAMPLE
# num_of_sequences = 1
# depth_list = [1]
# sequence_list = [49, 49]
# len_list = [2]
# n_avg = 100
# ###### TEST EXAMPLE

print("The sequence was successfully generated")


def play_sequence(sequence, start, length):
    align(qc_xy, qt_xy, cr_drive, cr_cancel)
    i = declare(int)
    align(qc_xy, qt_xy, cr_drive, cr_cancel)
    with for_(i, start, i < start + length, i + 1):
        align(qc_xy, qt_xy, cr_drive, cr_cancel)
        with switch_(sequence[i], unsafe=False):
            with case_(0):
                # identity
                align(qc_xy, qt_xy, cr_drive, cr_cancel)
            with case_(1):
                play("x90", qc_xy)
            with case_(2):
                play("-x90", qc_xy)
            with case_(3):
                play("x180", qc_xy)
            with case_(4):
                play("y90", qc_xy)
            with case_(5):
                play("-y90", qc_xy)
            with case_(6):
                play("y180", qc_xy)
            with case_(7):
                play("x90", qt_xy)
            with case_(8):
                play("x90", qc_xy)
                play("x90", qt_xy)
            with case_(9):
                play("-x90", qc_xy)
                play("x90", qt_xy)
            with case_(10):
                play("x180", qc_xy)
                play("x90", qt_xy)
            with case_(11):
                play("y90", qc_xy)
                play("x90", qt_xy)
            with case_(12):
                play("-y90", qc_xy)
                play("x90", qt_xy)
            with case_(13):
                play("y180", qc_xy)
                play("x90", qt_xy)
            with case_(14):
                play("-x90", qt_xy)
            with case_(15):
                play("x90", qc_xy)
                play("-x90", qt_xy)
            with case_(16):
                play("-x90", qc_xy)
                play("-x90", qt_xy)
            with case_(17):
                play("x180", qc_xy)
                play("-x90", qt_xy)
            with case_(18):
                play("y90", qc_xy)
                play("-x90", qt_xy)
            with case_(19):
                play("-y90", qc_xy)
                play("-x90", qt_xy)
            with case_(20):
                play("y180", qc_xy)
                play("-x90", qt_xy)
            with case_(21):
                play("x180", qt_xy)
            with case_(22):
                play("x90", qc_xy)
                play("x180", qt_xy)
            with case_(23):
                play("-x90", qc_xy)
                play("x180", qt_xy)
            with case_(24):
                play("x180", qc_xy)
                play("x180", qt_xy)
            with case_(25):
                play("y90", qc_xy)
                play("x180", qt_xy)
            with case_(26):
                play("-y90", qc_xy)
                play("x180", qt_xy)
            with case_(27):
                play("y180", qc_xy)
                play("x180", qt_xy)
            with case_(28):
                play("y90", qt_xy)
            with case_(29):
                play("x90", qc_xy)
                play("y90", qt_xy)
            with case_(30):
                play("-x90", qc_xy)
                play("y90", qt_xy)
            with case_(31):
                play("x180", qc_xy)
                play("y90", qt_xy)
            with case_(32):
                play("y90", qc_xy)
                play("y90", qt_xy)
            with case_(33):
                play("-y90", qc_xy)
                play("y90", qt_xy)
            with case_(34):
                play("y180", qc_xy)
                play("y90", qt_xy)
            with case_(35):
                play("-y90", qt_xy)
            with case_(36):
                play("x90", qc_xy)
                play("-y90", qt_xy)
            with case_(37):
                play("-x90", qc_xy)
                play("-y90", qt_xy)
            with case_(38):
                play("x180", qc_xy)
                play("-y90", qt_xy)
            with case_(39):
                play("y90", qc_xy)
                play("-y90", qt_xy)
            with case_(40):
                play("-y90", qc_xy)
                play("-y90", qt_xy)
            with case_(41):
                play("y180", qc_xy)
                play("-y90", qt_xy)
            with case_(42):
                play("y180", qt_xy)
            with case_(43):
                play("x90", qc_xy)
                play("y180", qt_xy)
            with case_(44):
                play("-x90", qc_xy)
                play("y180", qt_xy)
            with case_(45):
                play("x180", qc_xy)
                play("y180", qt_xy)
            with case_(46):
                play("y90", qc_xy)
                play("y180", qt_xy)
            with case_(47):
                play("-y90", qc_xy)
                play("y180", qt_xy)
            with case_(48):
                play("y180", qc_xy)
                play("y180", qt_xy)
            with case_(49):
                # We use direct CR
                align(qt_xy, qc_xy, cr_drive, cr_cancel)
                play("square_positive" * amp(cr_drive_amp), cr_drive, duration=120)
                play("square_positive" * amp(cr_cancel_amp), cr_cancel, duration=120)

                # One can also use flattop envelope, which is smoother than a square envelope
                # play("flattop_blackman" * amp(cr_drive_amp), cr_drive, duration=120)
                # play("flattop_blackman" * amp(cr_cancel_amp), cr_cancel, duration=120)

                # align for the next step and clear the phase shift
                align(qt_xy, qc_xy, cr_drive, cr_cancel)
                frame_rotation_2pi(+0.25, qc_xy)
                # check the sign before 0.25 -- could be either way depending on cr_drive_phase
                play("y90" * amp(0.00001), qc_xy, duration=1)
                # this is a tricky step. the goal is to avoid face-to-face frame rotations
                # qm only recognizes the last frame rotation before a real pulse if multiple frame rotations are stacked
                # e.g., two CNOTs can be adjacent in a random gate sequence, and two 0.25 rotations are intended
                # but qm only implements the last 0.25 rotation

                # if you want to avoid this subtlety, you can use real XY gates to replace the virtual Z/2
                # Z/2: play('-x90', qc_xy); play('y90', qc_xy); play('x90', qc_xy);
                # -Z/2: play('-x90', qc_xy); play('-y90', qc_xy); play('x90', qc_xy);

                align(qt_xy, qc_xy, cr_drive, cr_cancel)


###################
# The QUA program #
###################

with program() as rb:
    depth = declare(int)  # QUA variable for the varying depth_list index
    depth_target = declare(int)  # QUA variable for the current depth (changes in steps of delta_clifford)
    depth_len = declare(int)
    assign(depth_len, len(depth_list))
    n_avg_ = declare(int, value=n_avg)
    m = declare(int)  # QUA variable for the loop over random sequences
    # n = declare(int)  # QUA variable for the averaging loop
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    state1 = declare(bool)  # QUA variable for state discrimination
    state2 = declare(bool)
    # The relevant streams
    m_st = declare_stream()

    state1_st = declare_stream()
    state2_st = declare_stream()

    sequence_qua = declare(int, value=sequence_list)
    len_list_qua = declare(int, value=len_list)
    num_of_sequences_ = declare(int, num_of_sequences)

    start = declare(int, value=0)
    run = declare(int, value=0)

    frame_rotation_2pi(cr_drive_phase, cr_drive)
    frame_rotation_2pi(cr_cancel_phase, cr_cancel)
    # Globally setting the frame rotations for cr_drive and cr_cancel

    with for_(m, 0, m < num_of_sequences, m + 1):  # QUA for_ loop over the random sequences

        # No flux element in this experiment
        align()
        with for_(depth, 0, depth < len(depth_list), depth + 1):  # Loop over the depths

            # Insert the last gate in the sequence by the sequence's inverse gate
            # Only played the depth corresponding to target_depth

            with for_(n, 0, n < n_avg_, n + 1):
                align()
                wait(thermalization_time * u.ns, f"rr{1}")
                wait(thermalization_time * u.ns, f"rr{2}")
                align()

                # play("cw_rip_pulse", "c_12_xy")
                # wait(10*u.us, qc_xy,qt_xy,cr_drive, cr_cancel)
                # The above is for SQMS use case, where we also drive a center coupler resonator (Phys. Rev. Applied 22, 034007)

                play_sequence(sequence_qua, start, len_list_qua[run])
                align(qc_xy, qt_xy, "rr1", "rr2", cr_drive, cr_cancel)
                # Align the two elements to measure after playing the circuit.
                # Make sure you updated the ge_threshold and angle if you want to use state discrimination
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")

                assign(state1, (I[0] < ge_threshold_q1) & (I[1] < ge_threshold_q2))
                save(state1, state1_st)
                assign(state2, I[1] > ge_threshold_q2)
                save(state2, state2_st)

                assign(start, start + len_list_qua[run])
                assign(run, run + 1)
                # Go to the next depth
                align()
        save(m, m_st)

    with stream_processing():
        m_st.save("iteration")
        # saves a 2D array of depth and random pulse sequences in order to get error bars along the random sequences
        state2_st.boolean_to_int().buffer(n_avg).map(FUNCTIONS.average()).buffer(len(depth_list)).buffer(
            num_of_sequences
        ).save("state2")
        # returns a 1D array of averaged random pulse sequences vs depth of circuit for live plotting
        state2_st.boolean_to_int().buffer(n_avg).map(FUNCTIONS.average()).buffer(len(depth_list)).average().save(
            "state2_avg"
        )
        # saves a 2D array of depth and random pulse sequences in order to get error bars along the random sequences
        state1_st.boolean_to_int().buffer(n_avg).map(FUNCTIONS.average()).buffer(len(depth_list)).buffer(
            num_of_sequences
        ).save("state1")
        # returns a 1D array of averaged random pulse sequences vs depth of circuit for live plotting
        state1_st.boolean_to_int().buffer(n_avg).map(FUNCTIONS.average()).buffer(len(depth_list)).average().save(
            "state1_avg"
        )


###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, rb, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(rb)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["state1_avg", "state2_avg", "iteration", "state1"], mode="live")

    state1_avg, state2_avg, iteration, state = results.fetch_all()

    print(state)
    print(state1_avg)
    ydata = state1_avg

    qm.close()

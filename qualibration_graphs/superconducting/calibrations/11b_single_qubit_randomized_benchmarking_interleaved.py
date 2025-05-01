# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.single_qubit_randomized_benchmarking_interleaved import (
    Parameters,
    get_interleaved_gate_index,
)
from calibration_utils.single_qubit_randomized_benchmarking import (
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Initialisation}
description = """
        SINGLE QUBIT RANDOMIZED BENCHMARKING - INTERLEAVED
The program consists in playing random sequences of Clifford gates and measuring the
state of the resonator afterward. Each random sequence is derived on the FPGA for the
maximum depth (specified as an input) and played for each depth asked by the user (the
sequence is truncated to the desired depth). Each truncated sequence ends with the
recovery gate, found at each step thanks to a preloaded lookup table (Cayley table),
that will bring the qubit back to its ground state.
In this version, a Clifford gate chosen by the user is interleaved between each random gate in the sequence. This allows
to characterize the fidelity of a specific gate.

If the readout has been calibrated and is good enough, then state discrimination can be
applied to only return the state of the qubit. Otherwise, the 'I' and 'Q' quadratures
are returned. Each sequence is played n_avg times for averaging. A second averaging is
performed by playing different random sequences.

The data is then post-processed to extract the single-qubit gate fidelity and error per
gate.

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the qubit parameters precisely (nodes 04b_power_rabi.py and 06a_ramsey.py).
    - (optional) Having optimized the readout parameters (nodes 08a, 08b and 08c).
    - (optional) Having calibrated the DRAG parameters (nodes 10a and 10b or 10c).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - The averaged single qubit gate fidelity: qubit.gate_fidelity["interleaved_gate"].
"""

node = QualibrationNode[Parameters, Quam](
    name="11b_single_qubit_randomized_benchmarking_interleaved",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    num_of_sequences = node.parameters.num_random_sequences  # Number of random sequences
    # Number of averaging loops for each random sequence
    strict_timing = node.parameters.use_strict_timing
    n_avg = node.parameters.num_shots
    max_circuit_depth = node.parameters.max_circuit_depth
    delta_clifford = node.parameters.delta_clifford
    assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
    num_depths = max_circuit_depth // delta_clifford
    seed = node.parameters.seed  # Pseudo-random number generator seed
    interleaved_gate_index = get_interleaved_gate_index(node.parameters.interleaved_gate_operation)
    # List of recovery gates from the lookup table
    inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]

    def generate_sequence(interleaved_gate_index):
        cayley = declare(int, value=c1_table.flatten().tolist())
        inv_list = declare(int, value=inv_gates)
        current_state = declare(int)
        step = declare(int)
        sequence = declare(int, size=2 * max_circuit_depth + 1)
        inv_gate = declare(int, size=2 * max_circuit_depth + 1)
        i = declare(int)
        rand = Random(seed=seed)

        assign(current_state, 0)
        with for_(i, 0, i < 2 * max_circuit_depth, i + 2):
            assign(step, rand.rand_int(24))
            assign(current_state, cayley[current_state * 24 + step])
            assign(sequence[i], step)
            assign(inv_gate[i], inv_list[current_state])
            # interleaved gate
            assign(step, interleaved_gate_index)
            assign(current_state, cayley[current_state * 24 + step])
            assign(sequence[i + 1], step)
            assign(inv_gate[i + 1], inv_list[current_state])

        return sequence, inv_gate

    def play_sequence(sequence_list, depth, qubit):
        i = declare(int)
        with for_(i, 0, i <= depth, i + 1):
            with switch_(sequence_list[i], unsafe=True):
                with case_(0):
                    qubit.xy.wait(qubit.xy.operations["x180"].length // 4)
                with case_(1):  # x180
                    qubit.xy.play("x180")
                with case_(2):  # y180
                    qubit.xy.play("y180")
                with case_(3):  # Z180
                    qubit.xy.play("y180")
                    qubit.xy.play("x180")
                with case_(4):  # Z90 X180 Z-180
                    qubit.xy.play("x90")
                    qubit.xy.play("y90")
                with case_(5):  # Z-90 Y-90 Z-90
                    qubit.xy.play("x90")
                    qubit.xy.play("-y90")
                with case_(6):  # Z-90 X180 Z-180
                    qubit.xy.play("-x90")
                    qubit.xy.play("y90")
                with case_(7):  # Z-90 Y90 Z-90
                    qubit.xy.play("-x90")
                    qubit.xy.play("-y90")
                with case_(8):  # X90 Z90
                    qubit.xy.play("y90")
                    qubit.xy.play("x90")
                with case_(9):  # X-90 Z-90
                    qubit.xy.play("y90")
                    qubit.xy.play("-x90")
                with case_(10):  # z90 X90 Z90
                    qubit.xy.play("-y90")
                    qubit.xy.play("x90")
                with case_(11):  # z90 X-90 Z90
                    qubit.xy.play("-y90")
                    qubit.xy.play("-x90")
                with case_(12):  # x90
                    qubit.xy.play("x90")
                with case_(13):  # -x90
                    qubit.xy.play("-x90")
                with case_(14):  # y90
                    qubit.xy.play("y90")
                with case_(15):  # -y90
                    qubit.xy.play("-y90")
                with case_(16):  # Z90
                    qubit.xy.play("-x90")
                    qubit.xy.play("y90")
                    qubit.xy.play("x90")
                with case_(17):  # -Z90
                    qubit.xy.play("-x90")
                    qubit.xy.play("-y90")
                    qubit.xy.play("x90")
                with case_(18):  # Y-90 Z-90
                    qubit.xy.play("x180")
                    qubit.xy.play("y90")
                with case_(19):  # Y90 Z90
                    qubit.xy.play("x180")
                    qubit.xy.play("-y90")
                with case_(20):  # Y90 Z-90
                    qubit.xy.play("y180")
                    qubit.xy.play("x90")
                with case_(21):  # Y-90 Z90
                    qubit.xy.play("y180")
                    qubit.xy.play("-x90")
                with case_(22):  # x90 Z-90
                    qubit.xy.play("x90")
                    qubit.xy.play("y90")
                    qubit.xy.play("x90")
                with case_(23):  # -x90 Z90
                    qubit.xy.play("-x90")
                    qubit.xy.play("y90")
                    qubit.xy.play("-x90")

    # Register the sweep axes to be added to the dataset when fetching data
    depths = np.arange(1, max_circuit_depth + 0.1, delta_clifford)
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "nb_of_sequences": xr.DataArray(np.arange(num_of_sequences), attrs={"long_name": "Number of sequences"}),
        "depths": xr.DataArray(depths, attrs={"long_name": "Number of Clifford gates"}),
    }
    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        depth = declare(int)  # QUA variable for the varying depth
        # QUA variable for the current depth (changes in steps of delta_clifford)
        depth_target = declare(int)
        # QUA variable to store the last Clifford gate of the current sequence which is replaced by the recovery gate
        saved_gate = declare(int)
        # QUA variable for the loop over random sequences
        m = declare(int)
        m_st = declare_stream()

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            # QUA for_ loop over the random sequences
            with for_(m, 0, m < num_of_sequences, m + 1):
                # Save the counter for the progress bar
                save(m, m_st)
                # Generate the random sequence of length max_circuit_depth
                sequence_list, inv_gate_list = generate_sequence(interleaved_gate_index=interleaved_gate_index)
                assign(depth_target, 2)  # Initialize the current depth to 1

                with for_(depth, 1, depth <= 2 * max_circuit_depth, depth + 1):
                    # Replacing the last gate in the sequence with the sequence's inverse gate
                    # The original gate is saved in 'saved_gate' and is being restored at the end
                    assign(saved_gate, sequence_list[depth])
                    assign(sequence_list[depth], inv_gate_list[depth - 1])
                    # Only played the depth corresponding to target_depth
                    with if_(depth == depth_target):
                        with for_(n, 0, n < n_avg, n + 1):
                            # Initialize the qubits
                            for i, qubit in multiplexed_qubits.items():
                                qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                                # Align the two elements to play the sequence after qubit initialization
                            align()
                            # Manipulate the qubits
                            for i, qubit in multiplexed_qubits.items():
                                # The strict_timing ensures that the sequence will be played without gaps
                                if strict_timing:
                                    with strict_timing_():
                                        # Play the random sequence of desired depth
                                        play_sequence(sequence_list, depth, qubit)
                                else:
                                    play_sequence(sequence_list, depth, qubit)
                            align()
                            # Readout the qubits
                            for i, qubit in multiplexed_qubits.items():
                                if node.parameters.use_state_discrimination:
                                    qubit.readout_state(state[i])
                                    save(state[i], state_st[i])
                                else:
                                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                    save(I[i], I_st[i])
                                    save(Q[i], Q_st[i])
                            align()
                        # Go to the next depth
                        assign(depth_target, depth_target + 2 * delta_clifford)
                    # Reset the last gate of the sequence back to the original Clifford gate
                    # (that was replaced by the recovery gate at the beginning)
                    assign(sequence_list[depth], saved_gate)

        with stream_processing():
            m_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).buffer(num_of_sequences).save(
                        f"state{i + 1}"
                    )
                else:
                    I_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).buffer(num_of_sequences).save(
                        f"I{i + 1}"
                    )
                    Q_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).buffer(num_of_sequences).save(
                        f"Q{i + 1}"
                    )


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_random_sequences,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    fig_raw_fit.suptitle(
        f"Single qubit randomized benchmarking interleaved with {node.parameters.interleaved_gate_operation}"
    )
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "amplitude": fig_raw_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            q.gate_fidelity[node.parameters.interleaved_gate_operation] = float(
                1 - node.results["fit_results"][q.name]["error_per_gate"]
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

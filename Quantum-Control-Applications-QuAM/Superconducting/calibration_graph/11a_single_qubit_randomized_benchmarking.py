# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from qualibrate.utils.logger_m import logger
from quam_config import QuAM
from quam_experiments.experiments.single_qubit_randomized_benchmarking import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from quam_experiments.parameters.qubits_experiment import get_qubits
from quam_experiments.workflow import simulate_and_plot
from qualibration_libs.xarray_data_fetcher import XarrayDataFetcher


# %% {Initialisation}
description = """
        SINGLE QUBIT RANDOMIZED BENCHMARKING
The program consists in playing random sequences of Clifford gates and measuring the
state of the resonator afterward. Each random sequence is derived on the FPGA for the
maximum depth (specified as an input) and played for each depth asked by the user
(the sequence is truncated to the desired depth). Each truncated sequence ends with the
recovery gate, found at each step thanks to a preloaded lookup table (Cayley table),
that will bring the qubit back to its ground state.

If the readout has been calibrated and is good enough, then state discrimination can be
applied to only return the state of the qubit. Otherwise, the 'I' and 'Q' quadratures
are returned. Each sequence is played n_avg times for averaging. A second averaging is
performed by playing different random sequences.

The data is then post-processed to extract the single-qubit gate fidelity and error per
gate.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under
      study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit spectroscopy,
      rabi_chevron, power_rabi and updated the state.
    - Having the qubit frequency perfectly calibrated (ramsey).
    - (optional) Having calibrated the readout (readout_frequency, amplitude,
      duration_optimization IQ_blobs) for better SNR and state discrimination.
    - Set the desired flux bias.
"""


node = QualibrationNode[Parameters, QuAM](
    name="11a_single_qubit_randomized_benchmarking",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, QuAM]):
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["q1", "q3"]
    pass


# Instantiate the QuAM class from the state file
node.machine = QuAM.load()

# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, QuAM]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    num_of_sequences = node.parameters.num_random_sequences  # Number of random sequences
    # Number of averaging loops for each random sequence
    n_avg = node.parameters.num_averages
    max_circuit_depth = node.parameters.max_circuit_depth  # Maximum circuit depth
    if node.parameters.delta_clifford < 1:
        raise NotImplementedError("Delta clifford < 2 is not supported.")
    #  Play each sequence with a depth step equals to 'delta_clifford - Must be > 1
    delta_clifford = node.parameters.delta_clifford
    assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
    num_depths = max_circuit_depth // delta_clifford + 1
    seed = node.parameters.seed  # Pseudo-random number generator seed
    # Flag to enable state discrimination if the readout has been calibrated (rotated blobs and threshold)
    state_discrimination = node.parameters.use_state_discrimination
    strict_timing = node.parameters.use_strict_timing
    # List of recovery gates from the lookup table
    from qualang_tools.bakery.randomized_benchmark_c1 import c1_table

    inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]

    # %% {Utility functions}
    def power_law(power, a, b, p):
        return a * (p**power) + b

    def generate_sequence():
        cayley = declare(int, value=c1_table.flatten().tolist())
        inv_list = declare(int, value=inv_gates)
        current_state = declare(int)
        step = declare(int)
        sequence = declare(int, size=max_circuit_depth + 1)
        inv_gate = declare(int, size=max_circuit_depth + 1)
        i = declare(int)
        rand = Random(seed=seed)

        assign(current_state, 0)
        with for_(i, 0, i < max_circuit_depth, i + 1):
            assign(step, rand.rand_int(24))
            assign(current_state, cayley[current_state * 24 + step])
            assign(sequence[i], step)
            assign(inv_gate[i], inv_list[current_state])

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
    depths = np.arange(0, max_circuit_depth + 0.1, delta_clifford)
    depths[0] = 1
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "nb_of_sequences": xr.DataArray(np.arange(num_of_sequences), attrs={"long_name": "number of sequences"}),
        "depths": xr.DataArray(depths, attrs={"long_name": "depths", "units": "Hz"}),
    }
    with program() as node.namespace["qua_program"]:
        depth = declare(int)  # QUA variable for the varying depth
        # QUA variable for the current depth (changes in steps of delta_clifford)
        depth_target = declare(int)
        # QUA variable to store the last Clifford gate of the current sequence which is replaced by the recovery gate
        saved_gate = declare(int)
        m = declare(int)  # QUA variable for the loop over random sequences
        I, I_st, Q, Q_st, n, n_st = node.machine.qua_declaration()
        state = [declare(int) for _ in range(num_qubits)]
        # The relevant streams
        m_st = declare_stream()
        # state_st = declare_stream()
        state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            # QUA for_ loop over the random sequences
            with for_(m, 0, m < num_of_sequences, m + 1):
                # Generate the random sequence of length max_circuit_depth
                sequence_list, inv_gate_list = generate_sequence()
                assign(depth_target, 0)  # Initialize the current depth to 0

                with for_(depth, 1, depth <= max_circuit_depth, depth + 1):
                    # Replacing the last gate in the sequence with the sequence's inverse gate
                    # The original gate is saved in 'saved_gate' and is being restored at the end
                    assign(saved_gate, sequence_list[depth])
                    assign(sequence_list[depth], inv_gate_list[depth - 1])
                    for i, qubit in multiplexed_qubits.items():
                        # Only played the depth corresponding to target_depth
                        with if_((depth == 1) | (depth == depth_target)):
                            with for_(n, 0, n < n_avg, n + 1):
                                # Initialize the qubits
                                qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                                # Align the two elements to play the sequence after qubit initialization
                                qubit.align()
                                # The strict_timing ensures that the sequence will be played without gaps
                                if strict_timing:
                                    with strict_timing_():
                                        # Play the random sequence of desired depth
                                        play_sequence(sequence_list, depth, qubit)
                                else:
                                    play_sequence(sequence_list, depth, qubit)
                                # Align the two elements to measure after playing the circuit.
                                qubit.align()
                                if node.parameters.use_state_discrimination:
                                    qubit.readout_state(state[i])
                                    save(state[i], state_st[i])
                                else:
                                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                    save(I[i], I_st[i])
                                    save(Q[i], Q_st[i])

                            # Go to the next depth
                            assign(depth_target, depth_target + delta_clifford)
                        # Reset the last gate of the sequence back to the original Clifford gate
                        # (that was replaced by the recovery gate at the beginning)
                        assign(sequence_list[depth], saved_gate)
                # Save the counter for the progress bar
                save(m, m_st)

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

    # with program() as randomized_benchmarking_multiplexed:
    #     depth = declare(int)  # QUA variable for the varying depth
    #     # QUA variable for the current depth (changes in steps of delta_clifford)
    #     depth_target = declare(int)
    #     # QUA variable to store the last Clifford gate of the current sequence which is replaced by the recovery gate
    #     saved_gate = declare(int)
    #     m = declare(int)  # QUA variable for the loop over random sequences
    #     I, I_st, Q, Q_st, n, n_st = node.machine.qua_declaration()
    #     state = [declare(int) for _ in range(num_qubits)]
    #     # The relevant streams
    #     m_st = declare_stream()
    #     # state_st = declare_stream()
    #     state_st = [declare_stream() for _ in range(num_qubits)]
    #
    #     for i, qubit in enumerate(qubits):
    #
    #         # Bring the active qubits to the desired frequency point
    #         node.machine.set_all_fluxes(flux_point=flux_point, target=qubit)
    #
    #     # QUA for_ loop over the random sequences
    #     with for_(m, 0, m < num_of_sequences, m + 1):
    #         # Generate the random sequence of length max_circuit_depth
    #         sequence_list, inv_gate_list = generate_sequence()
    #         assign(depth_target, 0)  # Initialize the current depth to 0
    #
    #         with for_(depth, 1, depth <= max_circuit_depth, depth + 1):
    #             # Replacing the last gate in the sequence with the sequence's inverse gate
    #             # The original gate is saved in 'saved_gate' and is being restored at the end
    #             assign(saved_gate, sequence_list[depth])
    #             assign(sequence_list[depth], inv_gate_list[depth - 1])
    #             # Only played the depth corresponding to target_depth
    #             with if_((depth == 1) | (depth == depth_target)):
    #
    #                 with for_(n, 0, n < n_avg, n + 1):
    #
    #                     for i, qubit in enumerate(qubits):
    #
    #                         # Initialize the qubits
    #                         if reset_type == "active":
    #                             qubit.reset_qubit_active()
    #                         else:
    #                             qubit.resonator.wait(qubit.thermalization_time * u.ns)
    #                         # Align the two elements to play the sequence after qubit initialization
    #
    #                     align()
    #
    #                     for i, qubit in enumerate(qubits):
    #
    #                         # The strict_timing ensures that the sequence will be played without gaps
    #                         if strict_timing:
    #                             with strict_timing_():
    #                                 # Play the random sequence of desired depth
    #                                 play_sequence(sequence_list, depth, qubit)
    #                         else:
    #                             play_sequence(sequence_list, depth, qubit)
    #
    #                     align()
    #
    #                     # Align the two elements to measure after playing the circuit.
    #                     for i, qubit in enumerate(qubits):
    #
    #                         qubit.readout_state(state[i])
    #
    #                         save(state[i], state_st[i])
    #
    #                 # Go to the next depth
    #                 assign(depth_target, depth_target + delta_clifford)
    #             # Reset the last gate of the sequence back to the original Clifford gate
    #             # (that was replaced by the recovery gate at the beginning)
    #             assign(sequence_list[depth], saved_gate)
    #         # Save the counter for the progress bar
    #         save(m, m_st)
    #
    #     with stream_processing():
    #         m_st.save("iteration")
    #         for i in range(num_qubits):
    #             state_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(
    #                 num_depths
    #             ).buffer(num_of_sequences).save(f"state{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, QuAM]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, QuAM]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        # TODO: sort out multiplexed with RB
        # if not node.parameters.multiplexed:
        #     job = qm.execute(randomized_benchmarking_individual)
        # else:
        #     job = qm.execute(randomized_benchmarking_multiplexed)
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            # print_progress_bar(job, iteration_variable="n", total_number_of_iterations=node.parameters.num_averages)
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_random_sequences,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        print(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, QuAM]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node = node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, QuAM]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], logger)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, QuAM]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figure_amplitude"] = fig_raw_fit


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, QuAM]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            q.gate_fidelities["averaged"] = float(1 - node.results["fit_results"][q.name]["error_per_gate"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, QuAM]):
    node.save()

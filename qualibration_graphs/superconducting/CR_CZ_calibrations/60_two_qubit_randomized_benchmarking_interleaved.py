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
from quam_config import Quam
from calibration_utils.two_qubit_randomized_benchmarking.sequence_tools import (
    pre_generate_sequence_interleaved,
)
from calibration_utils.two_qubit_randomized_benchmarking import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.data_process_utils import *
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Initialisation}
description = """
        TWO-QUBIT INTERLEAVED RANDOMIZED BENCHMARKING
"""

node = QualibrationNode[Parameters, Quam](
    name="60a_two_qubit_randomized_benchmarking",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.multiplexed = True
    # node.parameters.qubit_pairs = ["q1-2", "q3-4"]
    # node.parameters.use_state_discrimination = True
    # node.parameters.num_random_sequences = 5
    # node.parameters.delta_clifford = 1
    # node.parameters.max_circuit_depth = 10
    # node.parameters.simulate = False
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
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    num_of_sequences = (
        node.parameters.num_random_sequences
    )  # Number of random sequences
    # Number of averaging loops for each random sequence
    n_avg = node.parameters.num_shots
    state_discrimination = node.parameters.use_state_discrimination
    max_circuit_depth = node.parameters.max_circuit_depth
    delta_clifford = node.parameters.delta_clifford
    assert (max_circuit_depth / delta_clifford).is_integer(), (
        "max_circuit_depth / delta_clifford must be an integer."
    )
    num_depths = max_circuit_depth // delta_clifford
    seed = node.parameters.seed  # Pseudo-random number generator seed
    strict_timing = node.parameters.use_strict_timing

    def play_sequence(sequence, start, length, qp):
        qc = qp.qubit_control
        qt = qp.qubit_target
        zz = qp.zz_drive
        zz_elems = [zz.name, qc.xy.name, qt.xy.name, qt.xy_detuned.name]

        i = declare(int)
        with for_(i, start, i < start + length, i + 1):
            align(*zz_elems)
            with switch_(sequence[i], unsafe=False):
                with case_(0):
                    # identity
                    align(*zz_elems)
                with case_(1):
                    qc.xy.play("x90")
                with case_(2):
                    qc.xy.play("-x90")
                with case_(3):
                    qc.xy.play("x180")
                with case_(4):
                    qc.xy.play("y90")
                with case_(5):
                    qc.xy.play("-y90")
                with case_(6):
                    qc.xy.play("y180")
                with case_(7):
                    qt.xy.play("x90")
                with case_(8):
                    qc.xy.play("x90")
                    qt.xy.play("x90")
                with case_(9):
                    qc.xy.play("-x90")
                    qt.xy.play("x90")
                with case_(10):
                    qc.xy.play("x180")
                    qt.xy.play("x90")
                with case_(11):
                    qc.xy.play("y90")
                    qt.xy.play("x90")
                with case_(12):
                    qc.xy.play("-y90")
                    qt.xy.play("x90")
                with case_(13):
                    qc.xy.play("y180")
                    qt.xy.play("x90")
                with case_(14):
                    qt.xy.play("-x90")
                with case_(15):
                    qc.xy.play("x90")
                    qt.xy.play("-x90")
                with case_(16):
                    qc.xy.play("-x90")
                    qt.xy.play("-x90")
                with case_(17):
                    qc.xy.play("x180")
                    qt.xy.play("-x90")
                with case_(18):
                    qc.xy.play("y90")
                    qt.xy.play("-x90")
                with case_(19):
                    qc.xy.play("-y90")
                    qt.xy.play("-x90")
                with case_(20):
                    qc.xy.play("y180")
                    qt.xy.play("-x90")
                with case_(21):
                    qt.xy.play("x180")
                with case_(22):
                    qc.xy.play("x90")
                    qt.xy.play("x180")
                with case_(23):
                    qc.xy.play("-x90")
                    qt.xy.play("x180")
                with case_(24):
                    qc.xy.play("x180")
                    qt.xy.play("x180")
                with case_(25):
                    qc.xy.play("y90")
                    qt.xy.play("x180")
                with case_(26):
                    qc.xy.play("-y90")
                    qt.xy.play("x180")
                with case_(27):
                    qc.xy.play("y180")
                    qt.xy.play("x180")
                with case_(28):
                    qt.xy.play("y90")
                with case_(29):
                    qc.xy.play("x90")
                    qt.xy.play("y90")
                with case_(30):
                    qc.xy.play("-x90")
                    qt.xy.play("y90")
                with case_(31):
                    qc.xy.play("x180")
                    qt.xy.play("y90")
                with case_(32):
                    qc.xy.play("y90")
                    qt.xy.play("y90")
                with case_(33):
                    qc.xy.play("-y90")
                    qt.xy.play("y90")
                with case_(34):
                    qc.xy.play("y180")
                    qt.xy.play("y90")
                with case_(35):
                    qt.xy.play("-y90")
                with case_(36):
                    qc.xy.play("x90")
                    qt.xy.play("-y90")
                with case_(37):
                    qc.xy.play("-x90")
                    qt.xy.play("-y90")
                with case_(38):
                    qc.xy.play("x180")
                    qt.xy.play("-y90")
                with case_(39):
                    qc.xy.play("y90")
                    qt.xy.play("-y90")
                with case_(40):
                    qc.xy.play("-y90")
                    qt.xy.play("-y90")
                with case_(41):
                    qc.xy.play("y180")
                    qt.xy.play("-y90")
                with case_(42):
                    qt.xy.play("y180")
                with case_(43):
                    qc.xy.play("x90")
                    qt.xy.play("y180")
                with case_(44):
                    qc.xy.play("-x90")
                    qt.xy.play("y180")
                with case_(45):
                    qc.xy.play("x180")
                    qt.xy.play("y180")
                with case_(46):
                    qc.xy.play("y90")
                    qt.xy.play("y180")
                with case_(47):
                    qc.xy.play("-y90")
                    qt.xy.play("y180")
                with case_(48):
                    qc.xy.play("y180")
                    qt.xy.play("y180")
                with case_(49):
                    # CNOT decomposition into IH CZ IH gate
                    qt.xy.play("x180")
                    qt.xy.play("y90")
                    align(*zz_elems)
                    qp.apply("stark_cz")
                    align(*zz_elems)
                    qt.xy.play("x180")
                    qt.xy.play("y90")

    def qubit_pair_reset(qp):
        qc = qp.qubit_control
        qt = qp.qubit_target
        # Reset the qubits to the ground state
        qc.reset(
            node.parameters.reset_type,
            node.parameters.simulate,
            log_callable=node.log,
        )
        qt.reset(
            node.parameters.reset_type,
            node.parameters.simulate,
            log_callable=node.log,
        )

    # Register the sweep axes to be added to the dataset when fetching data
    depths = np.arange(1, max_circuit_depth + 0.1, delta_clifford, dtype=int)

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "nb_of_sequences": xr.DataArray(
            np.arange(num_of_sequences), attrs={"long_name": "Number of sequences"}
        ),
        "depths": xr.DataArray(depths, attrs={"long_name": "Number of Clifford gates"}),
    }

    np.random.seed(seed=seed)
    if node.parameters.interleaved_CNOT:
        sequence_list, len_list = pre_generate_sequence_interleaved(
            num_of_sequences, depths, [("CNOT", "01")]
        )
    else:
        sequence_list, len_list = pre_generate_sequence_interleaved(
            num_of_sequences, depths
        )

    with program() as node.namespace["qua_program"]:
        I_c, I_c_st, Q_c, Q_c_st, n, n_st = node.machine.declare_qua_variables(
            num_IQ_pairs=num_qubit_pairs
        )
        I_t, I_t_st, Q_t, Q_t_st, _, _ = node.machine.declare_qua_variables(
            num_IQ_pairs=num_qubit_pairs
        )
        if state_discrimination:
            state = [declare(int) for _ in range(num_qubit_pairs)]
            state_c = [declare(int) for _ in range(num_qubit_pairs)]
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
            state_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]

        sequence_qua = declare(int, value=sequence_list)
        len_list_qua = declare(int, value=len_list)
        start = declare(int, value=0)

        seq_idx = declare(int, value=0)  # Index for the sequence
        idx_st = declare_stream()

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            # QUA for_ loop over the random sequences
            with for_(seq_idx, 0, seq_idx < len(len_list), seq_idx + 1):

                with for_(n, 0, n < n_avg, n + 1):

                    for i, qp in multiplexed_qubit_pairs.items():
                        qc = qp.qubit_control
                        qt = qp.qubit_target
                        zz = qp.zz_drive
                        zz_elems = [zz.name, qc.xy.name, qt.xy.name, qt.xy_detuned.name]

                        # Initialize the qubits
                        qubit_pair_reset(qp)
                        align(*zz_elems)

                        # Manipulate the qubits
                        # The strict_timing ensures that the sequence will be played without gaps
                        if strict_timing:
                            with strict_timing_():
                                # Play the random sequence of desired depth
                                play_sequence(sequence_qua, start, len_list_qua[seq_idx], qp)
                        else:
                            play_sequence(sequence_qua, start, len_list_qua[seq_idx], qp)
                        align(qc.resonator.name, qt.resonator.name, *zz_elems)        

                        # Readout the qubits
                        if state_discrimination:
                            qc.readout_state(state_c[i])
                            qt.readout_state(state_t[i])
                            # state is 0 when |00>
                            # P(|00>) =/= P(|0X>)P(|X0>)
                            assign(
                                state[i],
                                Cast.to_int(~((state_c[i] == 0) & (state_t[i] == 0))),
                            )
                            save(state[i], state_st[i])
                            save(state_c[i], state_c_st[i])
                            save(state_t[i], state_t_st[i])
                        else:
                            qc.resonator.measure("readout", qua_vars=(I_c[i], Q_c[i]))
                            qt.resonator.measure("readout", qua_vars=(I_t[i], Q_t[i]))
                            # save data
                            save(I_c[i], I_c_st[i])
                            save(Q_c[i], Q_c_st[i])
                            save(I_t[i], I_t_st[i])
                            save(Q_t[i], Q_t_st[i])

                        align(qc.resonator.name, qt.resonator.name, *zz_elems)

                        # Reset the frame of the qubits in order not to accumulate rotations
                        reset_frame(zz.name)
                        reset_frame(qt.xy_detuned.name)
                        reset_frame(qt.xy.name)
                        reset_frame(qt.xy.name)

                # Save the counter for the progress bar
                save(seq_idx, idx_st)

        with stream_processing():
            idx_st.save("iteration")
            for i in range(num_qubit_pairs):
                if state_discrimination:
                    state_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(
                        num_depths
                    ).buffer(num_of_sequences).save(f"state{i + 1}")
                    state_c_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(
                        num_depths
                    ).buffer(num_of_sequences).save(f"state_c{i + 1}")
                    state_t_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(
                        num_depths
                    ).buffer(num_of_sequences).save(f"state_t{i + 1}")
                else:
                    I_c_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(
                        num_depths
                    ).buffer(num_of_sequences).save(f"I_c{i + 1}")
                    Q_c_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(
                        num_depths
                    ).buffer(num_of_sequences).save(f"Q_c{i + 1}")
                    I_t_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(
                        num_depths
                    ).buffer(num_of_sequences).save(f"I_t{i + 1}")
                    Q_t_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(
                        num_depths
                    ).buffer(num_of_sequences).save(f"Q_t{i + 1}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()

    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
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
                data_fetcher["iteration"],
                node.parameters.num_random_sequences
                * (node.parameters.max_circuit_depth // node.parameters.delta_clifford),
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
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


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
    fig_raw_fit = plot_raw_data_with_fit(
        node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["ds_fit"]
    )
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "2QRB": fig_raw_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for i, qp in enumerate(node.namespace["qubit_pairs"]):
            if node.outcomes[qp.name] == "failed":
                continue
            if node.parameters.interleaved_CNOT:
                qp.extras["RB_decay_interleaved_CNOT"] = float(
                    node.results["ds_fit"]
                    .fit_data.sel(qubit_pair=qp.name)
                    .sel(fit_vals="decay")
                )
            else:
                qp.extras["RB_decay"] = float(
                    node.results["ds_fit"]
                    .fit_data.sel(qubit_pair=qp.name)
                    .sel(fit_vals="decay")
                )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

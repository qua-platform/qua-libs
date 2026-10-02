"""
Gate Set Tomography (Advance Input Stream)
==========================================
Variant of gate set tomography that streams germ sequences to the OPX via
``advance_input_stream``, avoiding the OPX1000 static gate-table limit (~16000 ints).

Germ sequences are pushed from Python with ``job.push_to_input_stream`` once per
circuit while the QUA program repeats each circuit ``num_shots`` times before
advancing to the next. Only ``max_germs_depth + 1`` ints live on the OPX at compile
time (length prefix plus the longest germ) instead of the full static table.

Prerequisites:
    - Calibrated readout with state discrimination (nodes 14-16 or GEF chain).
    - Calibrated single-qubit gates (x90, y90).

Note:
    pyGSTi StandardGST is single-qubit; run one qubit at a time via ``qubits``.
"""

# %% {Imports}
import numpy as np
from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualibrate import QualibrationNode
from quam_config import Quam

from calibration_utils.gate_set_tomography_ais import (
    Parameters,
    GERM_TOKENS_STREAM_NAME,
    analyse_gst_data,
    build_raw_dataset,
    recompute_counts_from_state,
    log_gst_design_summary,
    play_tokenized_gst_circuits,
    setup_gst_experiment,
    start_push_gst_germs_in_background,
    write_gst_html_report,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

#%%
# %% {Node initialisation}
description = """
        GATE SET TOMOGRAPHY (AIS)
Gate set tomography characterizes state preparation, measurement, and native gates
(x90, y90, I) using pyGSTi. Germ sequences are streamed via advance_input_stream to
avoid OPX1000 static gate-table limits.

Prerequisites:
    - Calibrated readout with state discrimination.
    - Calibrated x90 / y90 pulses.

State update:
    - None (analysis-only characterization).
"""

node = QualibrationNode[Parameters, Quam](
    name="70a_gate_set_tomography_ais",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    # node.parameters.qubits = ["q1"]
    # node.parameters.max_circuit_depth_in_power = 9
    # node.parameters.num_shots = 100
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the GST experiment design and compile the QUA program."""
    if not node.parameters.use_state_discrimination:
        raise ValueError("Gate set tomography requires use_state_discrimination=True.")

    node.namespace["qubits"] = qubits = get_qubits(node)
    if len(qubits) != 1:
        raise ValueError(
            "GST currently supports exactly one qubit per run. "
            f"Got {len(qubits)} qubits: {qubits.get_names()}."
        )

    qubit = qubits[0]
    n_runs = node.parameters.num_shots
    design = setup_gst_experiment(node.parameters.max_circuit_depth_in_power)
    node.namespace["gst_design"] = design
    log_gst_design_summary(design, n_runs, log_callable=node.log)

    tokenized_germs = design.all_germs_to_qua_tokenized_labels
    max_germs_depth = design.max_germs_depth
    total_germs_num = design.total_germs_num
    row_len = max_germs_depth + 1

    if node.parameters.simulate:
        with program() as node.namespace["qua_program"]:
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
            state = [declare(int)]
            state_st = [declare_stream()]

            single_germ_order = declare(int)
            native_gate_order = declare(int)
            tokenized_germs_list = declare(
                int, value=np.array(tokenized_germs).flatten()
            )
            single_germ_list = declare(int, size=row_len)
            germ_idx = declare(int)

            node.machine.initialize_qpu(target=qubit)
            align()

            with for_(germ_idx, 0, germ_idx < total_germs_num, germ_idx + 1):
                save(germ_idx, n_st)
                assign(single_germ_order, germ_idx * row_len)
                with for_(
                    native_gate_order,
                    0,
                    native_gate_order < row_len,
                    native_gate_order + 1,
                ):
                    assign(
                        single_germ_list[native_gate_order],
                        tokenized_germs_list[single_germ_order + native_gate_order],
                    )
                with for_(n, 0, n < n_runs, n + 1):
                    qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()
                    play_tokenized_gst_circuits(single_germ_list, qubit=qubit)
                    align()
                    qubit.readout_state(state[0])
                    save(state[0], state_st[0])

            if not node.parameters.multiplexed:
                align()

            with stream_processing():
                n_st.save("n")
                state_st[0].buffer(n_runs).buffer(total_germs_num).save("state1")
    else:
        with program() as node.namespace["qua_program"]:
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
            state = [declare(int)]
            state_st = [declare_stream()]

            germ_idx = declare(int)
            germ_tokens_is = declare_input_stream(
                int, name=GERM_TOKENS_STREAM_NAME, size=row_len
            )

            node.machine.initialize_qpu(target=qubit)
            align()

            with for_(germ_idx, 0, germ_idx < total_germs_num, germ_idx + 1):
                save(germ_idx, n_st)
                advance_input_stream(germ_tokens_is)
                with for_(n, 0, n < n_runs, n + 1):
                    qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()
                    play_tokenized_gst_circuits(germ_tokens_is, qubit=qubit)
                    align()
                    qubit.readout_state(state[0])
                    save(state[0], state_st[0])

            if not node.parameters.multiplexed:
                align()

            with stream_processing():
                n_st.save("n")
                state_st[0].buffer(n_runs).buffer(total_germs_num).save("state1")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Simulate the QUA program (uses static gate table; input streams are unreliable in sim)."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program, stream germ tokens, and fetch averaged state counts."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    design = node.namespace["gst_design"]
    tokenized_germs = design.all_germs_to_qua_tokenized_labels
    qubits = node.namespace["qubits"]

    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        push_thread = start_push_gst_germs_in_background(job, tokenized_germs)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            germ_idx = results.fetch_all()[0]
            progress_counter(
                germ_idx,
                design.total_germs_num,
                start_time=results.start_time,
            )
        push_thread.join()
        job.result_handles.wait_for_all_values()

    node.results["ds_raw"] = build_raw_dataset(
        job.result_handles,
        qubits,
        design,
        node.parameters.num_shots,
    )
    node.log(job.execution_report())


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset and rebuild the GST experiment design."""
    node.load_from_id(node.parameters.load_data_id)
    node.namespace["qubits"] = get_qubits(node)
    node.namespace["gst_design"] = setup_gst_experiment(
        node.parameters.max_circuit_depth_in_power
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Run pyGSTi StandardGST on the fetched counts."""
    if node.parameters.load_data_id is None:
        ds = node.results["ds_raw"]
    else:
        ds = node.results.get("ds_raw") or node.results.get("ds")
        if ds is None:
            raise KeyError("Loaded node has no dataset ('ds_raw' or 'ds').")

    # Stored counts from before the integer-sum fix are not trusted: rebuild from state.
    ds = recompute_counts_from_state(ds, node.parameters.num_shots)
    node.results["ds_raw"] = ds
    design = node.namespace["gst_design"]
    gst_result_objects = analyse_gst_data(node, ds, design)
    node.namespace["gst_result_objects"] = gst_result_objects
    node.outcomes = {q.name: "successful" for q in node.namespace["qubits"]}


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save node results and write pyGSTi HTML reports when analysis ran."""
    if not node.parameters.simulate:
        node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()

    if not node.parameters.simulate:
        gst_result_objects = node.namespace.get("gst_result_objects", {})
        for qubit in node.namespace.get("qubits", []):
            gst_results = gst_result_objects.get(qubit.name)
            if gst_results is None:
                continue
            report_dirname, report_main = write_gst_html_report(
                gst_results,
                qubit.name,
                node.snapshot_idx,
            )
            node.results["gst_report_dir"] = report_dirname
            node.results["gst_report_main"] = report_main
        node.save()

# %%

"""
Two-Qubit Gate Set Tomography (Advance Input Stream)
====================================================
2Q variant of ``11c_gate_set_tomography`` using pyGSTi ``smq2Q_XYICPHASE``
(fallback ``smq2Q_XYCPHASE``) as in ``CQT_2Q_GST.ipynb``.

Native gates: I, x90, y90 on both qubits, plus the calibrated CZ macro.
Each circuit is a length-prefixed row ``[L, op_1, ..., op_L, 0-pad]``.
The QUA loop plays only the first ``L`` opcodes, so the padding never
executes. Rows are streamed via ``advance_input_stream``; only
``max_germs_depth + 1`` ints live on the OPX at compile time.

x90 and y90 must have the same duration on both qubits (a multiple of 4 ns
and at least 16 ns). The idle layer waits exactly that long on both XY lines.

The unsafe switch has no ``align()``. CZ is ``apply(align_elements=False)``
plus a 16 ns wait on both XY elements, so the virtual-Z corrections do not
open a gap before the next pulse. That keyword has to be a real parameter of
``CZGate.apply`` (quam-builder PR #154, ``hotfix/cz-align-elements-opt-in``).
An older builder swallows it in ``**kwargs``, the macro still aligns, and
every I/x90/y90 layer pays that overhead. The node refuses to compile unless
the parameter is present.

Prerequisites:
    - Calibrated readout with state discrimination on both qubits.
    - Calibrated single-qubit gates (x90, y90).
    - Calibrated CZ macro on the selected qubit pair.

Note:
    Two-qubit GST circuit counts grow quickly. Default max length is 4
    (``max_circuit_depth_in_power=2``). Run one pair at a time.
"""

# %% {Imports}
import numpy as np
from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualibrate import QualibrationNode
from quam_config import Quam

from calibration_utils.gate_set_tomography_2q import (
    Parameters,
    GERM_TOKENS_STREAM_NAME,
    analyse_gst_data_2q,
    build_raw_dataset_2q,
    log_gst_design_summary,
    play_tokenized_gst_circuits_2q,
    require_cz_align_elements,
    setup_gst_experiment_2q,
    start_push_gst_germs_in_background,
    write_gst_html_report,
)
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot

# %%
# %% {Node initialisation}
description = """
        TWO-QUBIT GATE SET TOMOGRAPHY (AIS)
Two-qubit GST of I, x90, y90 (both qubits) and CZ using pyGSTi StandardGST.
Germ / fiducial circuits are streamed via advance_input_stream.

Prerequisites:
    - Calibrated readout with state discrimination on both qubits.
    - Calibrated x90 / y90 pulses.
    - Calibrated CZ macro on the qubit pair.

State update:
    - None (analysis-only characterization).
"""

node = QualibrationNode[Parameters, Quam](
    name="40_two_qubit_gate_set_tomography",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    # node.parameters.qubit_pairs = ["q19-20"]
    # node.parameters.operation = "cz_unipolar"
    # node.parameters.max_circuit_depth_in_power = 2
    # node.parameters.num_shots = 100
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the 2Q GST experiment design and compile the QUA program."""
    if not node.parameters.use_state_discrimination:
        raise ValueError("Gate set tomography requires use_state_discrimination=True.")

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    if len(qubit_pairs) != 1:
        raise ValueError(
            "2Q GST currently supports exactly one qubit pair per run. "
            f"Got {len(qubit_pairs)} pairs: {qubit_pairs.get_names()}."
        )

    qp = qubit_pairs[0]
    cz_operation = node.parameters.operation
    if cz_operation not in qp.macros:
        available = sorted(qp.macros.keys())
        raise ValueError(f"Qubit pair {qp.name!r} has no macro {cz_operation!r}. Available macros: {available}")
    require_cz_align_elements(qp, cz_operation)

    n_runs = node.parameters.num_shots
    design = setup_gst_experiment_2q(
        node.parameters.max_circuit_depth_in_power,
        use_fiducial_pair_reduction=node.parameters.use_fiducial_pair_reduction,
    )
    node.namespace["gst_design"] = design
    log_gst_design_summary(design, n_runs, log_callable=node.log)

    tokenized_germs = design.all_germs_to_qua_tokenized_labels
    max_germs_depth = design.max_germs_depth
    total_germs_num = design.total_germs_num
    row_len = max_germs_depth + 1

    if node.parameters.simulate:
        with program() as node.namespace["qua_program"]:
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
            state_control = declare(int)
            state_target = declare(int)
            state = declare(int)
            state_st = declare_stream()

            single_germ_order = declare(int)
            native_gate_order = declare(int)
            tokenized_germs_list = declare(int, value=np.array(tokenized_germs).flatten())
            single_germ_list = declare(int, size=row_len)
            germ_idx = declare(int)

            node.machine.initialize_qpu(target=qp.qubit_control)
            node.machine.initialize_qpu(target=qp.qubit_target)
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
                    qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                    qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()
                    play_tokenized_gst_circuits_2q(
                        single_germ_list,
                        qubit_pair=qp,
                        cz_operation=cz_operation,
                    )
                    align()
                    qp.qubit_control.readout_state(state_control)
                    qp.qubit_target.readout_state(state_target)
                    assign(state, state_control * 2 + state_target)
                    save(state, state_st)

            if not node.parameters.multiplexed:
                align()

            with stream_processing():
                n_st.save("n")
                state_st.buffer(n_runs).buffer(total_germs_num).save("state2q")
    else:
        with program() as node.namespace["qua_program"]:
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
            state_control = declare(int)
            state_target = declare(int)
            state = declare(int)
            state_st = declare_stream()

            germ_idx = declare(int)
            germ_tokens_is = declare_input_stream(int, name=GERM_TOKENS_STREAM_NAME, size=row_len)

            node.machine.initialize_qpu(target=qp.qubit_control)
            node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(germ_idx, 0, germ_idx < total_germs_num, germ_idx + 1):
                save(germ_idx, n_st)
                advance_input_stream(germ_tokens_is)
                with for_(n, 0, n < n_runs, n + 1):
                    qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                    qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()
                    play_tokenized_gst_circuits_2q(
                        germ_tokens_is,
                        qubit_pair=qp,
                        cz_operation=cz_operation,
                    )
                    align()
                    qp.qubit_control.readout_state(state_control)
                    qp.qubit_target.readout_state(state_target)
                    assign(state, state_control * 2 + state_target)
                    save(state, state_st)

            if not node.parameters.multiplexed:
                align()

            with stream_processing():
                n_st.save("n")
                state_st.buffer(n_runs).buffer(total_germs_num).save("state2q")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Simulate the QUA program (uses static gate table; input streams are unreliable in sim)."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program, stream layer tokens, and fetch 2Q state counts."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    design = node.namespace["gst_design"]
    tokenized_germs = design.all_germs_to_qua_tokenized_labels
    qp = node.namespace["qubit_pairs"][0]

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

    node.results["ds_raw"] = build_raw_dataset_2q(
        job.result_handles,
        design,
        node.parameters.num_shots,
        qp.name,
    )
    node.log(job.execution_report())


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset and rebuild the 2Q GST experiment design."""
    node.load_from_id(node.parameters.load_data_id)
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)
    node.namespace["gst_design"] = setup_gst_experiment_2q(
        node.parameters.max_circuit_depth_in_power,
        use_fiducial_pair_reduction=node.parameters.use_fiducial_pair_reduction,
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Run pyGSTi StandardGST on the fetched 2Q counts."""
    if node.parameters.load_data_id is None:
        ds = node.results["ds_raw"]
    else:
        ds = node.results.get("ds_raw") or node.results.get("ds")
        if ds is None:
            raise KeyError("Loaded node has no dataset ('ds_raw' or 'ds').")

    design = node.namespace["gst_design"]
    gst_result_objects = analyse_gst_data_2q(node, ds, design)
    node.namespace["gst_result_objects"] = gst_result_objects
    node.outcomes = {qp.name: "successful" for qp in node.namespace["qubit_pairs"]}


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save node results and write the pyGSTi HTML report when analysis ran."""
    if not node.parameters.simulate:
        node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()

    if not node.parameters.simulate:
        gst_result_objects = node.namespace.get("gst_result_objects", {})
        for qp in node.namespace.get("qubit_pairs", []):
            gst_results = gst_result_objects.get(qp.name)
            if gst_results is None:
                continue
            report_dirname, report_main = write_gst_html_report(
                gst_results,
                qp.name,
                node.snapshot_idx,
            )
            node.results["gst_report_dir"] = report_dirname
            node.results["gst_report_main"] = report_main
        node.save()


# %%

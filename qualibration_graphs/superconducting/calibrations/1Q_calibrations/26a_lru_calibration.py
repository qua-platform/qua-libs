"""f-LRU calibration: |f0> -> |g1> pump amplitude, frequency and duration for microwave leakage removal."""

# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.common_utils import (
    check_if_reach,
    ensure_gef_readout_pulse,
    reset_for_gef,
    set_gef_readout_frequency,
    upconverter_frequency,
    warn_if_reach,
)
from calibration_utils.lru_calibration import (
    Parameters,
    add_lru_element,
    assign_lru_coords,
    classify_iq_shots,
    fit_raw_data,
    get_amp_factors,
    get_detunings,
    log_fitted_results,
    log_iq_results,
    plot_iq_blobs,
    plot_lru_maps,
    plot_pf_along_line,
    prediction_outside_window,
    process_raw_dataset,
    spurious_lines,
)
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Description}
description = """
        f-LRU CALIBRATION (|f0> -> |g1> PUMP)
Microwave leakage removal: a pump at f_LRU = f_02 - f_r = 2 * f_01 - |anharmonicity| - f_r drives
|f0> -> |g1> (transmon level, resonator photon number); the photon leaks out of the resonator and
leaves the qubit in |g0>. |f0> and |g1> have opposite parity, so a charge drive on the qubit XY line
couples them at first order through the transmon anharmonicity.

The pump is played by the qubit.lru element, on the second upconverter of the qubit XY port.
Qubits without one get it for this run: the XY port's LO moves to upconverter 1 and upconverter 2
is set 100 MHz below the expected pump. It is kept in the QUAM state only for the qubits whose fit
succeeds, and only when add_lru_to_state is set. The qubit XY drive must not share its core with
another element that changes frequency (resonator or lru): on QOP 3.7.3 that cancels the
x180/EF_x180 pulses. An existing lru element far from the f0-g1 prediction gets a warning.

Sequence, for each LRU duration, amplitude and drive frequency:
    1. Prepare |f> (x180, then EF_x180 at the e-f frequency).
    2. Play the LRU pulse at the swept amplitude and frequency (IF of qubit.lru) and duration.
    3. Wait for the resonator to empty (wait_after_lru_ns, by default its depletion time).
    4. Measure g/e/f with the readout_GEF pulse.

The |f0> -> |g1> resonance appears as a dip in P(f) and a ridge in P(g) whose frequency moves
with amplitude A by the AC Stark shift. The node fits the line f0g1 + c2 * A^2 through the P(f)
minima, reads P(f) along it, and selects the duration and amplitude reaching the lowest P(f).
Other lines can give a P(f) dip in the same window: f_r/2 (second harmonic filling the resonator),
f_02/3 (three-photon g-f, also with a P(g) rise), and f_12 or f_01 (direct e-f or g-e drive). The
node warns when the swept window contains f_r/2 or f_02/3, or comes within 150 MHz of f_12 or f_01.

Debug switches: skip_lru_pulse replaces the LRU pulse by a wait; return_iq_blobs saves every shot's raw
readout_GEF I/Q instead of its g/e/f state, classifies the shots offline with centers refit on the run,
and runs the usual analysis on the resulting populations, plus an IQ-blob figure.

Prerequisites:
    - Calibrated x180 and EF_x180 (nodes 04b and 13) and anharmonicity (node 12).
    - Calibrated qubit frequency (nodes 03a and 06a) and resonator frequency (node 02a), since both
      enter the pump prediction.
    - Calibrated g/e/f readout: readout_GEF, GEF_frequency_shift and gef_centers (nodes 14a/14b/15).

State update (successful qubits with an LRU element in the state, or add_lru_to_state set):
    - qubit.lru.RF_frequency: the Stark-shifted resonance at the selected amplitude.
    - qubit.lru.operations[operation].amplitude and .length: the selected amplitude and duration.
    An element added by this run (add_lru_to_state) is written with these values and its upconverter on
    the XY port directly, not as proposals; a qubit from an older state also gets the LRU-capable class.
    The stored frequency is absolute: re-run this node after recalibrating f_01, the anharmonicity or f_r.
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="26a_lru_calibration",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    # node.parameters.skip_lru_pulse = True
    # node.parameters.return_iq_blobs = True
    pass


def _check_qubits(node: QualibrationNode[Parameters, Quam], qubits, amp_factors, detunings, added):
    """Fail early on a missing operation or calibration, and warn about LRU IFs beyond the MW-FEM
    specification, a stale lru element (not in ``added``, the qubits given one for this run) and
    spurious lines in the swept window."""
    operation = node.parameters.operation
    for qubit in qubits:
        if operation not in qubit.lru.operations:
            raise ValueError(f"{qubit.name}.lru has no '{operation}' operation.")
        if "EF_x180" not in qubit.xy.operations:
            raise ValueError(f"{qubit.name} has no EF_x180 operation: calibrate it with node 13 first.")
        if qubit.resonator.gef_centers is None:
            raise ValueError(f"{qubit.name} has no gef_centers: calibrate the g/e/f readout with node 15 first.")
        max_amplitude = abs(qubit.lru.operations[operation].amplitude) * np.max(np.abs(amp_factors))
        if max_amplitude >= 1:
            raise ValueError(
                f"{qubit.name}: the LRU amplitude reaches {max_amplitude:.3f} of full scale; "
                f"lower max_amp_factor below {1 / abs(qubit.lru.operations[operation].amplitude):.3f}."
            )
        lo = upconverter_frequency(qubit.lru)
        f_min = qubit.lru.RF_frequency + detunings.min()
        f_max = qubit.lru.RF_frequency + detunings.max()
        for frequency in (f_min, f_max):
            message = check_if_reach(frequency, lo, f"{qubit.name}.lru")
            if message is not None:
                node.log(message, level="warning")
        if qubit.name not in added:
            message = prediction_outside_window(qubit, f_min, f_max)
            if message is not None:
                node.log(message, level="warning")
        for name, frequency in spurious_lines(qubit, f_min, f_max):
            node.log(
                f"{qubit.name}: {name} at {frequency / 1e9:.4f} GHz is near the swept window "
                f"{f_min / 1e9:.4f}-{f_max / 1e9:.4f} GHz; a P(f) dip there is not the f0-g1 line.",
                level="warning",
            )


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    if node.parameters.reset_type == "active":
        raise ValueError("'active' reset is not GEF-aware; use 'thermal' or 'active_gef'")
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    operation = node.parameters.operation
    readout_operation = node.parameters.readout_operation
    amp_factors = get_amp_factors(node.parameters)
    detunings = get_detunings(node.parameters)
    durations_ns = np.array(node.parameters.pulse_durations_ns, dtype=int)
    durations_cc = (durations_ns // 4).tolist()

    # Qubits without an LRU element get one for this run; keep_or_remove_lru_elements decides whether it stays
    node.namespace["lru_additions"] = {}
    for qubit in qubits:
        addition = add_lru_element(qubit, operation)
        if addition is not None:
            node.namespace["lru_additions"][qubit.name] = addition
            node.log(
                f"{qubit.name}: added an lru element for this run at {qubit.lru.RF_frequency / 1e9:.4f} GHz, "
                f"upconverter {qubit.lru.upconverter} at {upconverter_frequency(qubit.lru) / 1e9:.4f} GHz."
            )
            if addition.band_note is not None:
                node.log(addition.band_note, level="warning")
    _check_qubits(node, qubits, amp_factors, detunings, set(node.namespace["lru_additions"]))
    ensure_gef_readout_pulse(qubits, log_callable=node.log)

    return_iq = node.parameters.return_iq_blobs
    n_points = len(durations_cc) * len(amp_factors) * len(detunings)
    if return_iq and n_avg * n_points > 1e6:
        node.log(
            f"return_iq_blobs saves {n_avg * n_points:.2e} shots per qubit and quadrature; shrink the sweep "
            "or num_shots if fetching is slow.",
            level="warning",
        )

    # Register the sweep axes to be added to the dataset when fetching data (outer to inner loop)
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        **({"shot": xr.DataArray(np.arange(n_avg), attrs={"long_name": "shot"})} if return_iq else {}),
        "duration": xr.DataArray(durations_ns, attrs={"long_name": "LRU pulse duration", "units": "ns"}),
        "amp_prefactor": xr.DataArray(amp_factors, attrs={"long_name": "LRU amplitude pre-factor"}),
        "detuning": xr.DataArray(detunings, attrs={"long_name": "LRU drive detuning", "units": "Hz"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        state = [declare(int) for _ in range(num_qubits)]
        is_e = [declare(bool) for _ in range(num_qubits)]
        is_f = [declare(bool) for _ in range(num_qubits)]
        e_st = [declare_stream() for _ in range(num_qubits)]
        f_st = [declare_stream() for _ in range(num_qubits)]
        t = declare(int)  # LRU pulse duration in clock cycles
        a = declare(fixed)  # LRU amplitude pre-factor
        df = declare(int)  # LRU drive detuning in Hz

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_each_(t, durations_cc):
                    with for_(*from_array(a, amp_factors)):
                        with for_(*from_array(df, detunings)):
                            # Qubit initialization, leaving the resonator at the GEF readout frequency
                            for qubit in multiplexed_qubits.values():
                                reset_for_gef(qubit, node.parameters.reset_type, node.parameters.simulate, u)
                            align()

                            # Prepare |f> exactly as node 15 does, then pump |f0> -> |g1>.
                            # On hardware, a frequency update right before x180 cancels it, so x180 relies on
                            # the drive being left at the g-e frequency (config, reset, or the end of this block).
                            for qubit in multiplexed_qubits.values():
                                qubit.xy.play("x180")
                                qubit.xy.update_frequency(
                                    qubit.xy.intermediate_frequency - abs(qubit.anharmonicity), keep_phase=True
                                )
                                qubit.xy.play("EF_x180")
                                qubit.xy.update_frequency(qubit.xy.intermediate_frequency, keep_phase=True)
                                align(qubit.xy.name, qubit.lru.name)
                                # Retune the pump only once the XY pulses on the same port are over
                                qubit.lru.update_frequency(df + int(qubit.lru.intermediate_frequency))
                                if node.parameters.skip_lru_pulse:
                                    qubit.lru.wait(t)  # clock cycles
                                else:
                                    qubit.lru.play(operation, amplitude_scale=a, duration=t)
                                wait_ns = node.parameters.wait_after_lru_ns
                                if wait_ns is None:
                                    wait_ns = qubit.resonator.depletion_time
                                if wait_ns >= 16:
                                    qubit.lru.wait(int(wait_ns) // 4)  # clock cycles
                                align(qubit.xy.name, qubit.lru.name)
                            align()

                            # g/e/f readout
                            for i, qubit in multiplexed_qubits.items():
                                if return_iq:
                                    # Same measurement as readout_state_gef, without the classification
                                    rr = qubit.resonator
                                    set_gef_readout_frequency(qubit)
                                    rr.measure(readout_operation, qua_vars=(I[i], Q[i]))
                                    rr.update_frequency(rr.intermediate_frequency)
                                    rr.wait(rr.depletion_time // 4)
                                    save(I[i], I_st[i])
                                    save(Q[i], Q_st[i])
                                else:
                                    qubit.readout_state_gef(state[i], pulse_name=readout_operation)
                                    assign(is_e[i], state[i] == 1)
                                    assign(is_f[i], state[i] == 2)
                                    save(is_e[i], e_st[i])
                                    save(is_f[i], f_st[i])
                            align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if return_iq:
                    for stream, name in ((I_st[i], "I"), (Q_st[i], "Q")):
                        stream.buffer(len(detunings)).buffer(len(amp_factors)).buffer(len(durations_cc)).buffer(
                            n_avg
                        ).save(f"{name}{i + 1}")
                    continue
                for stream, name in ((e_st[i], "pe"), (f_st[i], "pf")):
                    stream.boolean_to_int().buffer(len(detunings)).buffer(len(amp_factors)).buffer(
                        len(durations_cc)
                    ).average().save(f"{name}{i + 1}")


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
    """Connect, execute QUA program, fetch raw data and store as dataset 'ds_raw'."""
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
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset, with the LRU amplitudes and frequencies needed to reload it
    node.results["ds_raw"] = assign_lru_coords(dataset, node.namespace["qubits"], node.parameters.operation)


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
    """Analyse raw data -> store fitted dataset 'ds_fit' and dict 'fit_results'."""
    if node.parameters.return_iq_blobs:
        # Classify the raw shots offline into pe and pf, then analyse them like the real-time populations
        node.results["ds_raw"], node.results["iq_classification"] = classify_iq_shots(
            node.results["ds_raw"], node.namespace["qubits"]
        )
        log_iq_results(node.results["iq_classification"], log_callable=node.log)
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
    """Plot the P(g), P(e) and P(f) maps with the fitted resonance line, and P(f) along that line (IQ blobs in debug mode)."""
    ds, ds_fit, qubits = node.results["ds_raw"], node.results["ds_fit"], node.namespace["qubits"]
    figures = {f"map_{duration:.0f}ns": plot_lru_maps(ds, qubits, ds_fit, duration) for duration in ds.duration.values}
    # P(e) and P(f) maps, for display only: the analysis is unchanged
    for population in ("pe", "pf"):
        for duration in ds.duration.values:
            figures[f"map_{population}_{duration:.0f}ns"] = plot_lru_maps(ds, qubits, ds_fit, duration, population)
    figures["pf_along_line"] = plot_pf_along_line(ds, qubits, ds_fit, node.results["fit_results"])
    if node.parameters.return_iq_blobs:
        figures["iq_blobs"] = plot_iq_blobs(ds, qubits, node.results["iq_classification"])
    plt.show()
    # Store the generated figures
    node.results["figures"] = figures


# %% {Keep_or_remove_LRU_elements}
def _write_lru_values(qubit, fit_result: dict, operation: str) -> None:
    """Set the calibrated LRU frequency, amplitude and duration on ``qubit.lru``."""
    qubit.lru.RF_frequency = fit_result["frequency"]
    qubit.lru.operations[operation].amplitude = fit_result["amplitude"]
    qubit.lru.operations[operation].length = int(fit_result["duration_ns"])


@node.run_action()
def keep_or_remove_lru_elements(node: QualibrationNode[Parameters, Quam]):
    """Remove the LRU elements added for this run, except for successful qubits when add_lru_to_state is set.

    node.save() writes the machine as it is in memory, and a new element is not a simple value that
    record_state_updates could offer as a proposal, so this choice is made here, before saving. A kept
    element gets its calibrated values right away, so the state never holds one with placeholder values,
    even when the proposals of update_state are rejected.
    """
    for name, addition in node.namespace.get("lru_additions", {}).items():
        if node.parameters.add_lru_to_state and node.outcomes.get(name) == "successful":
            _write_lru_values(addition.qubit, node.results["fit_results"][name], node.parameters.operation)
            node.log(f"{name}: keeping the calibrated lru element in the QUAM state (add_lru_to_state).")
            warn_if_reach(node, addition.qubit.lru, f"{name}.lru")
        else:
            addition.remove()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Propose the selected LRU frequency, amplitude and duration for every successful qubit whose LRU element
    existed before this run (elements kept from this run already hold them)."""
    added = node.namespace.get("lru_additions", {})
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed" or q.name in added:
                continue
            if getattr(q, "lru", None) is None:
                node.log(
                    f"{q.name}: the lru element was for this run only, so the calibrated values are not saved. "
                    "Re-run with add_lru_to_state=True to keep it."
                )
                continue
            _write_lru_values(q, node.results["fit_results"][q.name], node.parameters.operation)
            warn_if_reach(node, q.lru, f"{q.name}.lru")


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save all node results and state updates."""
    node.save()


# %%

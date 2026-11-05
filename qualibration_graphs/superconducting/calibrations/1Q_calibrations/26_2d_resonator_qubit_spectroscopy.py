# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from config.quam_builder import Quam
from calibration_utils.resonator_qubit_2d_spectroscopy import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    plot_stark_shift_with_fit,
    plot_photon_number_with_fit,
    get_qubit_probe_amplitude_factor,
    log_runtime_estimate,
)
from qualibration_libs.core import tracked_updates
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialisation}
description = """
        2D RESONATOR-QUBIT SPECTROSCOPY (chi, kappa, power)
This sequence implements the protocol of Fig. 2 of "System characterization of dispersive readout in
superconducting qubits", https://link.aps.org/doi/10.1103/PhysRevApplied.23.024055, which uses the ac Stark
shifted qubit frequency as a sensor of the energy stored in the readout resonator.

For every point of a 2D grid, the qubit is prepared in |0> or |1>, the resonator is driven at a frequency
wd,r for much longer than 1/kappa so that it reaches its steady-state photon occupation n_bar, and the qubit is
then probed at a frequency wd,q while the resonator is still being driven. A standard dispersive readout, taken
once the resonator has depleted, reports whether the qubit state was flipped by the probe. The probe is
resonant with the qubit when wd,q matches the ac Stark shifted qubit frequency of Eq. (7),

    w*_d,q = w_q,0 + 2 * chi * kappa * |A|^2 / ((w_d,r - (w_r,m -/+ chi))^2 + (kappa / 2)^2),

the upper sign being for |0> and the lower for |1>. Both states are acquired in the same run, interleaved
within each shot so that slow drifts move both curves alike and cancel in their separation.

Eq. (7) is a pair of Lorentzians with means w_r,m -/+ chi, depths (2 * chi)(4 |A|^2 / kappa) and widths kappa.
Fitting both therefore gives all three parameters at once:
    - 2 * chi as the horizontal distance between the two means,
    - kappa as their common width,
    - |A|^2 from the depths, which are 2 * chi * n_bar_max with n_bar_max = 4 |A|^2 / kappa, once chi is known.

The resonator drive amplitude should be raised, through `resonator_drive_amplitude_factor`, until the ac Stark
shift on resonance is several times the qubit linewidth; -20 MHz is used in the paper. The qubit probe detuning
sweep is deliberately asymmetric, since the Stark shift follows the sign of chi and is negative for a transmon
below its resonator.

Cores:
    The qubit probe has to be played while the resonator is still being driven. Elements that share a core are
    serialised by the pulse processor, so if the xy drive and the resonator sit on the same core the probe is
    silently pushed to after the drive ends, the qubit never sees any photons and the measured ac Stark shift is
    flat at zero. The node therefore gives the xy drive a core of its own for the duration of the run, which is
    reverted afterwards; see 'separate_drive_and_probe_cores'. Confirm the overlap in the waveform report when
    running with 'simulate' = True.

Runtime:
    This is a 2D sweep, and every one of its points carries a full qubit reset, so the node is long. With a
    thermal reset the reset alone is ~98% of each shot, and replaying the sequence for each qubit in turn
    multiplies that by the number of qubits. Set 'multiplexed' to True so that all qubits are driven and measured
    together, and prefer 'reset_type' = "active" once the readout threshold is calibrated; either change takes
    the default sweep from tens of minutes to tens of seconds on a full chip. The estimated runtime is logged
    before the job is submitted, which matters on backends that cap the execution time.

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c).
    - Having found the qubit frequency (node 03a) and roughly calibrated the x180 pulse (node 04b), which is
      used both to prepare 'e' and to set the area of the qubit probe pulse.
    - Having calibrated the readout discrimination threshold (node 07) when `use_state_discrimination` is True.
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - (optional) The resonator linewidth and the dressed resonator frequency, stored per prepared state under the
      keys already used by the state: qubit.resonator.extras["kappa_hz"] and ["f_01_ground"] when preparing 'g',
      ["kappa_e_hz"] and ["f_01_excited"] when preparing 'e'.
    - (optional) The dispersive shift qubit.chi, half the separation between the two dressed resonator
      frequencies, which is therefore only updated once both runs are present in the state.
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="2d_resonator_qubit_spectroscopy",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = [
        "qA4",
    ]
    node.parameters.reset_type = "active"
    node.parameters.resonator_drive_amplitude_factor = 0.03
    for q in node.parameters.qubits:
        node.machine.qubits[q].resonator.operations["readout"] = "#./readout_square"
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    # Temporary changes to the state, reverted by the revert_tracked_updates run_action at the end of the node.
    node.namespace["tracked_qubits"] = []
    for q in qubits:
        # Resolve the name before entering the context, since the tracked object proxies attribute access
        xy_core = f"{q.name}_xy"
        with tracked_updates(q, auto_revert=False) as q:
            # The qubit probe must be played while the resonator is still being driven, but elements sharing a
            # core are serialised by the pulse processor, so a shared core silently pushes the probe to after the
            # drive and no ac Stark shift is seen at all. Give the xy drive a core of its own.
            if node.parameters.separate_drive_and_probe_cores:
                q.xy.thread = None  # 'thread' is the deprecated spelling of 'core', and 'core' wins if both are set
                q.xy.core = xy_core
            # Raise the resonator drive amplitude beyond what the [-2; 2) amplitude_scale range allows
            if node.parameters.resonator_drive_amplitude_in_v is not None:
                q.resonator.operations[node.parameters.resonator_drive_operation].amplitude = (
                    node.parameters.resonator_drive_amplitude_in_v
                )
        node.namespace["tracked_qubits"].append(q)

    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots
    state_discrimination = node.parameters.use_state_discrimination
    initial_states = list(node.parameters.qubit_initial_states)
    if not initial_states:
        raise ValueError("'qubit_initial_states' must contain at least one of 'g' and 'e'.")

    # The resonator drive frequency sweep, centred on the readout frequency
    span_r = node.parameters.resonator_frequency_span_in_mhz * u.MHz
    step_r = node.parameters.resonator_frequency_step_in_mhz * u.MHz
    dfs_r = np.arange(-span_r / 2, +span_r / 2, step_r)
    # The qubit probe detuning sweep, asymmetric since the ac Stark shift follows the sign of chi
    dfs_q = np.arange(
        node.parameters.qubit_detuning_min_in_mhz * u.MHz,
        node.parameters.qubit_detuning_max_in_mhz * u.MHz,
        node.parameters.qubit_detuning_step_in_mhz * u.MHz,
    )

    # The pulse durations must be expressible in clock cycles
    probe_len = node.parameters.qubit_probe_len_in_ns
    settle_time = node.parameters.resonator_drive_settle_time_in_ns
    for name, duration in [
        ("qubit_probe_len_in_ns", probe_len),
        ("resonator_drive_settle_time_in_ns", settle_time),
    ]:
        if duration < 16 or duration % 4 != 0:
            raise ValueError(f"'{name}' must be a multiple of 4 ns and at least 16 ns, got {duration} ns.")
    # The resonator is driven for the whole time the qubit is probed, so that it stays in its steady state
    stark_drive_len = settle_time + probe_len
    node.namespace["stark_drive_len"] = stark_drive_len

    # Give the probe pulse an area of `qubit_probe_pi_fraction` times a pi pulse, unless it was set explicitly
    probe_operation = node.parameters.qubit_probe_operation
    probe_amplitudes = {
        q.name: (
            node.parameters.qubit_probe_amplitude_factor
            if node.parameters.qubit_probe_amplitude_factor is not None
            else get_qubit_probe_amplitude_factor(
                qubit=q,
                probe_operation=probe_operation,
                probe_length_in_ns=probe_len,
                pi_fraction=node.parameters.qubit_probe_pi_fraction,
                log_callable=node.log,
            )
        )
        for q in qubits
    }
    node.namespace["qubit_probe_amplitude_factors"] = probe_amplitudes

    # Every point of the 2D grid carries a full qubit reset, so this node is long and is easily rejected by a
    # backend that caps the execution time. Report the expected runtime before the job is submitted.
    node.namespace["estimated_runtime_in_seconds"] = log_runtime_estimate(
        qubits=list(qubits),
        parameters=node.parameters,
        num_points=len(dfs_r) * len(dfs_q),
        stark_drive_len=stark_drive_len,
        log_callable=node.log,
    )

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "initial_state": xr.DataArray(initial_states, attrs={"long_name": "prepared qubit state"}),
        "resonator_detuning": xr.DataArray(dfs_r, attrs={"long_name": "resonator drive detuning", "units": "Hz"}),
        "qubit_detuning": xr.DataArray(dfs_q, attrs={"long_name": "qubit pulse detuning", "units": "Hz"}),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        df_r = declare(int)  # QUA variable for the resonator drive frequency
        df_q = declare(int)  # QUA variable for the qubit probe frequency

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                # The prepared states are interleaved within each shot rather than acquired as separate runs, so
                # that a slow drift moves both Lorentzians alike and cancels in the 2 * chi that separates them
                for initial_state in initial_states:
                    with for_(*from_array(df_r, dfs_r)):
                        with for_(*from_array(df_q, dfs_q)):
                            # Qubit initialization, with both lines back at their nominal frequencies so that the
                            # active reset and the state preparation are on resonance
                            for i, qubit in multiplexed_qubits.items():
                                qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                                qubit.resonator.update_frequency(qubit.resonator.intermediate_frequency)
                                qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                            align()

                            # Prepare the qubit in 'g' or 'e'
                            if initial_state == "e":
                                for i, qubit in multiplexed_qubits.items():
                                    qubit.xy.play("x180")
                                align()

                            # Drive the resonator into its steady state and probe the ac Stark shifted qubit
                            for i, qubit in multiplexed_qubits.items():
                                qubit.resonator.update_frequency(df_r + qubit.resonator.intermediate_frequency)
                                qubit.xy.update_frequency(df_q + qubit.xy.intermediate_frequency)
                            align()
                            for i, qubit in multiplexed_qubits.items():
                                qubit.resonator.play(
                                    node.parameters.resonator_drive_operation,
                                    amplitude_scale=node.parameters.resonator_drive_amplitude_factor,
                                    duration=stark_drive_len // 4,
                                )
                                # Only probe the qubit once the resonator has reached its steady state, and keep
                                # driving the resonator until the probe is over
                                qubit.xy.wait(settle_time // 4)
                                qubit.xy.play(
                                    probe_operation,
                                    amplitude_scale=probe_amplitudes[qubit.name],
                                    duration=probe_len // 4,
                                )
                            align()

                            # Qubit readout, back at the nominal readout frequency and once the resonator has
                            # depleted, so that the ac Stark drive does not corrupt the measurement
                            for i, qubit in multiplexed_qubits.items():
                                qubit.resonator.update_frequency(qubit.resonator.intermediate_frequency)
                                qubit.resonator.wait(qubit.resonator.depletion_time // 4)
                            align()
                            for i, qubit in multiplexed_qubits.items():
                                if state_discrimination:
                                    qubit.readout_state(state[i])
                                    save(state[i], state_st[i])
                                else:
                                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                    save(I[i], I_st[i])
                                    save(Q[i], Q_st[i])
                            align()

        with stream_processing():
            n_st.save("n")
            # Buffers are applied innermost first, matching the save order within one shot:
            # the qubit pulse sweep, then the resonator drive sweep, then the interleaved prepared states
            for i in range(num_qubits):
                if state_discrimination:
                    (
                        state_st[i]
                        .buffer(len(dfs_q))
                        .buffer(len(dfs_r))
                        .buffer(len(initial_states))
                        .average()
                        .save(f"state{i + 1}")
                    )
                else:
                    for stream, label in ((I_st[i], "I"), (Q_st[i], "Q")):
                        (
                            stream.buffer(len(dfs_q))
                            .buffer(len(dfs_r))
                            .buffer(len(initial_states))
                            .average()
                            .save(f"{label}{i + 1}")
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
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
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

    # Compute and save the resonator drive amplitude (in V) to results
    resonator_drive_amplitudes = {
        q.name: float(
            (
                node.parameters.resonator_drive_amplitude_in_v
                if node.parameters.resonator_drive_amplitude_in_v is not None
                else q.resonator.operations[node.parameters.resonator_drive_operation].amplitude
            )
            * node.parameters.resonator_drive_amplitude_factor
        )
        for q in node.namespace["qubits"]
    }
    node.results["resonator_drive_amplitude"] = resonator_drive_amplitudes
    for q_name, amp in resonator_drive_amplitudes.items():
        if q_name in node.results["fit_results"]:
            node.results["fit_results"][q_name]["resonator_drive_amplitude"] = amp

    if "ds_fit" in node.results and node.results["ds_fit"] is not None:
        ds_qubits = [str(q) for q in node.results["ds_fit"].qubit.values]
        amps = [resonator_drive_amplitudes.get(name, np.nan) for name in ds_qubits]
        node.results["ds_fit"] = node.results["ds_fit"].assign(
            {"resonator_drive_amplitude": (["qubit"], amps)}
        )
        node.results["ds_fit"].resonator_drive_amplitude.attrs = {
            "long_name": "resonator drive amplitude",
            "units": "V",
        }

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
    # One 2D map per prepared state, as in Fig. 2(b) and 2(c), plus both Lorentzians together as in Fig. 2(d)
    node.results["figures"] = {
        f"spectroscopy_{initial_state}": plot_raw_data_with_fit(
            node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"], initial_state
        )
        for initial_state in node.parameters.qubit_initial_states
    }
    node.results["figures"]["stark_shift"] = plot_stark_shift_with_fit(
        node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"]
    )
    if "chi" in node.results["ds_fit"] and np.isfinite(node.results["ds_fit"].chi).any():
        node.results["figures"]["photon_number"] = plot_photon_number_with_fit(
            node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"]
        )
    plt.show()


# %% {Revert_tracked_updates}
@node.run_action()
def revert_tracked_updates(node: QualibrationNode[Parameters, Quam]):
    """Revert the temporary changes made to the state at the beginning of the node.

    This runs unconditionally, and before update_state, so that the temporary core and amplitude overrides are
    undone even when the state update is skipped and never leak into the recorded state updates.
    """
    for tracked_qubit in node.namespace.get("tracked_qubits", []):
        tracked_qubit.revert_changes()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate or not node.parameters.update_state)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            fit_result = node.results["fit_results"][q.name]
            # The width and the mean of each Lorentzian are stored per prepared state, under the keys already
            # used by the state
            extras_keys = {"g": ("kappa_hz", "f_01_ground"), "e": ("kappa_e_hz", "f_01_excited")}
            for initial_state, state_fit in fit_result["states"].items():
                if not state_fit["success"]:
                    continue
                kappa_key, frequency_key = extras_keys[initial_state]
                q.resonator.extras[kappa_key] = state_fit["kappa"]
                q.resonator.extras[frequency_key] = state_fit["resonator_frequency"]

            # chi is half the horizontal distance between the two means, so it needs both states
            if fit_result["chi"] is not None:
                q.chi = fit_result["chi"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

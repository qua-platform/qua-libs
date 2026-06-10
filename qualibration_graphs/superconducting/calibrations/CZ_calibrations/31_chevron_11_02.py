# %% {Imports}
from dataclasses import asdict
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.chevron_cz import (
    Parameters,
    baked_waveform,
    estimate_cz_flux_amplitude,
    fit_raw_data,
    log_fitted_results,
    get_moving_qubit,
    get_stationary_qubit,
    verify_moving_qubit,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Node_parameters}
description = """
CZ |11⟩↔|02⟩ or |11⟩↔|20⟩ Flux Chevron Calibration

Measures the time and amplitude required for the CZ gate by sweeping the moving-qubit flux
pulse amplitude (around the estimated |11⟩↔|02⟩ or |11⟩↔|20⟩ operating point) and duration (1 ns
granularity via baking). The resulting 2D Chevron pattern is fitted to extract the optimal gate amplitude and duration.

For tunable-coupler architectures the coupler is held at its CZ bias point
(``macros[operation].coupler_flux_pulse.amplitude``) throughout each flux pulse. For fixed-coupler
architectures no coupler element is needed.

Method
------
1. Prepare |11⟩ by applying x180 to both qubits.
2. Apply the moving-qubit flux pulse at scaled amplitude and variable duration, bringing it
   to the |11⟩↔|02⟩ or |11⟩↔|20⟩ avoided crossing. For tunable couplers, the coupler is simultaneously
   held at the CZ bias.
3. Measure both qubits (state discrimination or raw IQ).
4. Fit the 2D population map to a Rabi-Chevron model to extract the resonance amplitude and gate time.

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair.
- Calibrated readout for both qubits.
- Initial estimate of the coupler flux amplitude in case of tunable couplers.

Outcomes:
- Optimal flux pulse amplitude and duration for the CZ gate.
- Fitted Chevron pattern for visualization and verification.

State update:
- Updates the amplitude and duration of the flux pulse in ``macros[operation]``
  to the fitted CZ values (duration rounded up to the next multiple of 4 ns).
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="31_chevron_1102",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under calibration_utils/chevron_cz/parameters.py
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # node.parameters.qubit_pairs = ["q1-q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    # Get the qubit pairs to be calibrated
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    for qp in qubit_pairs:
        verify_moving_qubit(qp, log_callable=node.log)

    # define the amplitudes for the flux pulses
    pulse_amplitudes = {}
    for qp in qubit_pairs:
        pulse_amplitudes[qp.name] = estimate_cz_flux_amplitude(node.parameters, qp, log_callable=node.log)

    node.namespace["pulse_amplitudes"] = pulse_amplitudes

    # The number of averages
    n_avg = node.parameters.num_shots

    # Loop parameters
    amplitudes = np.arange(1 - node.parameters.amp_range, 1 + node.parameters.amp_range, node.parameters.amp_step)
    times_cycles = np.arange(1, node.parameters.max_time_in_ns)

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "amplitude": xr.DataArray(amplitudes, attrs={"long_name": "amplitudes of the flux pulse"}),
        "time": xr.DataArray(times_cycles, attrs={"long_name": "pulse duration", "units": "ns"}),
    }

    baked_config = node.machine.generate_config()

    # Detect tunable coupler presence for each pair (Python-time, not QUA-time).
    operation = node.parameters.operation
    has_coupler = {}
    coupler_amplitudes = {}
    for qp in qubit_pairs:
        macro = qp.macros[operation]
        has_coupler[qp.name] = macro.coupler_flux_pulse is not None
        if has_coupler[qp.name]:
            coupler_amplitudes[qp.name] = macro.coupler_flux_pulse.amplitude
            node.namespace["has_coupler"] = has_coupler
            node.namespace["coupler_amplitudes"] = coupler_amplitudes

    # Pre-compute baked short segments (1..16 ns) for each moving qubit (qubit only —
    # the coupler is played separately with 4 ns granularity to avoid strict_timing gaps).
    baked_signals = {
        qp.name: baked_waveform(
            get_moving_qubit(qp), baked_config, base_level=pulse_amplitudes[qp.name], max_samples=16
        )
        for qp in qubit_pairs
    }

    node.namespace["baked_config"] = baked_config

    with program() as node.namespace["qua_program"]:
        t = declare(int)  # QUA variable for the flux pulse segment index
        a = declare(fixed)
        t_left_ns = declare(int)  # QUA variable for the flux pulse segment index
        t_cycles = declare(int)  # QUA variable for the flux pulse segment index
        I_m, I_m_st, Q_m, Q_m_st, n, n_st = node.machine.declare_qua_variables()
        I_s, I_s_st, Q_s, Q_s_st, _, _ = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state_mq = [declare(int) for _ in range(num_qubit_pairs)]
            state_sq = [declare(int) for _ in range(num_qubit_pairs)]
            state_mq_st = [declare_output_stream() for _ in range(num_qubit_pairs)]
            state_sq_st = [declare_output_stream() for _ in range(num_qubit_pairs)]

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qp in multiplexed_qubit_pairs.values():
                mq = get_moving_qubit(qp)
                sq = get_stationary_qubit(qp)
                node.machine.initialize_qpu(target=mq)
                node.machine.initialize_qpu(target=sq)
            align()
            # Averaging loop
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                # Pulse amplitude loop
                with for_(*from_array(a, amplitudes)):
                    ################################################################################################
                    # The duration argument in the play command can only produce pulses with duration multiple of  #
                    # 4ns. To overcome this limitation we use the baking tool from the qualang-tools package to    #
                    # generate pulses with 1ns granularity. To avoid creating custom waveforms for each iteration  #
                    # we combine baked pulses with dynamically stretched (multiple of 4ns) pulses.                 #
                    ################################################################################################
                    with for_(*from_array(t, times_cycles)):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            mq = get_moving_qubit(qp)
                            sq = get_stationary_qubit(qp)
                            # Qubit initialization
                            mq.reset(node.parameters.reset_type, node.parameters.simulate)
                            sq.reset(node.parameters.reset_type, node.parameters.simulate)
                            align()
                            # set both qubits to the excited state to prepare |11⟩
                            mq.xy.play("x180")
                            sq.xy.play("x180")

                            align()

                            if has_coupler[qp.name]:
                                coupler_scale = coupler_amplitudes[qp.name] / qp.coupler.operations["const"].amplitude

                            # For the first 16ns we play baked pulses exclusively. Loop the time index until 16.
                            with if_(t <= 16):
                                with switch_(t):
                                    # Switch case to select the baked pulse with duration t ns
                                    for j in range(1, 17):
                                        with case_(j):
                                            baked_signals[qp.name][j - 1].run(
                                                amp_array=[(mq.z.name, a)]
                                            )
                                            if has_coupler[qp.name]:
                                                # Coupler only needs to hold the CZ bias level — 4 ns granularity is sufficient.
                                                # ceil(j/4) cycles ensures the hold covers the full j ns qubit baked pulse.
                                                qp.coupler.play(
                                                    "const", duration=(j + 3) // 4, amplitude_scale=coupler_scale
                                                )

                            # For pulse durations above 16ns we combine baking with regular play statements.
                            with else_():
                                # We calculate the closest lower multiple of 4 of the time index
                                assign(t_cycles, t >> 2)  # Right shift by 2 is a quick way to divide by 4
                                # Calculate the duration to add to pulse multiple of 4.
                                assign(t_left_ns, t - (t_cycles << 2))  # left shift by 2 to multiply by 4
                                # Switch case with the 4 possible sequences:
                                with switch_(t_left_ns):
                                    # Play only the pulse multiple of 4
                                    with case_(0):
                                        align()
                                        p = pulse_amplitudes[qp.name]
                                        denom = mq.z.operations["const"].amplitude
                                        scale = (p / denom) * a
                                        mq.z.play(
                                            "const",
                                            duration=t_cycles,
                                            amplitude_scale=scale,
                                        )
                                        if has_coupler[qp.name]:
                                            qp.coupler.play("const", duration=t_cycles, amplitude_scale=coupler_scale)
                                    # Play the pulse multiple of 4 followed by the baked pulse of the missing duration
                                    for j in range(1, 4):
                                        with case_(j):
                                            align()
                                            p = pulse_amplitudes[qp.name]
                                            denom = mq.z.operations["const"].amplitude
                                            scale = (p / denom) * a
                                            if has_coupler[qp.name]:
                                                qp.coupler.play(
                                                    "const", duration=t_cycles + 1, amplitude_scale=coupler_scale
                                                )
                                            with strict_timing_():
                                                mq.z.play(
                                                    "const",
                                                    duration=t_cycles,
                                                    amplitude_scale=scale,
                                                )
                                                baked_signals[qp.name][j - 1].run(
                                                    amp_array=[(mq.z.name, a)]
                                                )
                            align()

                            if node.parameters.use_state_discrimination:
                                mq.readout_state_gef(state_mq[ii])
                                sq.readout_state_gef(state_sq[ii])
                                save(state_mq[ii], state_mq_st[ii])
                                save(state_sq[ii], state_sq_st[ii])
                            else:
                                mq.resonator.measure("readout", qua_vars=(I_m[ii], Q_m[ii]))
                                sq.resonator.measure("readout", qua_vars=(I_s[ii], Q_s[ii]))
                                save(I_m[ii], I_m_st[ii])
                                save(Q_m[ii], Q_m_st[ii])
                                save(I_s[ii], I_s_st[ii])
                                save(Q_s[ii], Q_s_st[ii])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_mq_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"state_moving{i}")
                    state_sq_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"state_stationary{i}")
                else:
                    I_m_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"I_moving{i}")
                    Q_m_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"Q_moving{i}")
                    I_s_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"I_stationary{i}")
                    Q_s_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"Q_stationary{i}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.namespace["baked_config"]
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data.

    The raw data is stored in a xarray dataset called "ds_raw".
    """
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.namespace["baked_config"]
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


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    qubit_pairs = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]
    # define the amplitudes for the flux pulses
    pulse_amplitudes = {}
    for qp in qubit_pairs:
        pulse_amplitudes[qp.name] = estimate_cz_flux_amplitude(node.parameters, qp, log_callable=node.log)
    node.namespace["pulse_amplitudes"] = pulse_amplitudes
    node.namespace["qubits"] = [get_moving_qubit(qp) for qp in qubit_pairs] + [get_stationary_qubit(qp) for qp in qubit_pairs]
    node.namespace["qubit_pairs"] = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data.

    Stores the fitted data in another xarray dataset "ds_fit" and the fitted results in the
    "fit_results" dictionary.
    """
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_pair_name: ("successful" if fit_result["success"] else "failed")
        for qubit_pair_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_stationary = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["ds_fit"], qubit_role="stationary")
    fig_moving = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["ds_fit"], qubit_role="moving")
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "stationary_qubit": fig_stationary,
        "moving_qubit": fig_moving,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""

    operation = node.parameters.operation
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes[qp.name] == "failed":
                node.log(f"Skipping state update for {qp.name}: fit flagged unsuccessful.")
                continue
            qp.macros[operation].flux_pulse_qubit.amplitude = node.results["fit_results"][qp.name]["cz_amp"]
            # Round up to the upper 4 ns to be compatible with the hardware time resolution
            qp.macros[operation].flux_pulse_qubit.length = int(
                np.ceil(node.results["fit_results"][qp.name]["cz_len"] / 4) * 4
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

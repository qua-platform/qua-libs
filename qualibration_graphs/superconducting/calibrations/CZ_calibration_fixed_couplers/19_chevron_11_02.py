# %% {Imports}
import warnings
from dataclasses import asdict, dataclass
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.chevron_cz import (
    Parameters,
    baked_waveform,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from qm import SimulationConfig
from qm.qua import *
from qualang_tools.bakery import baking
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit
from qualibrate import NodeParameters, QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs, get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_builder.architecture.superconducting.custom_gates.flux_tunable_transmon_pair.two_qubit_gates import CZGate
from quam_config import Quam
from scipy.optimize import curve_fit

from quam.components.pulses import FlatTopGaussianPulse, SquarePulse

# %% {Node_parameters}
description = """
Unipolar CPhase Gate Calibration
This sequence measures the time and detuning required for a unipolar CPhase gate. The process involves:

1. Preparing both qubits in their excited states.
2. Applying a flux pulse with varying amplitude and duration.
3. Measuring the resulting state populations as a function of these parameters.
4. Fitting the results to a Ramsey-Chevron pattern.

From this pattern, we extract:
- The coupling strength (J2) between the qubits.
- The optimal gate parameters (amplitude and duration) for the CPhase gate.

The Ramsey-Chevron pattern emerges due to the interplay between the qubit-qubit coupling and the flux-induced detuning,
allowing us to precisely calibrate the CPhase gate.

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair.
- Calibrated readout for both qubits.
- Initial estimate of the flux pulse amplitude range.

Outcomes:
- Extracted J2 coupling strength.
- Optimal flux pulse amplitude and duration for the CPhase gate.
- Fitted Ramsey-Chevron pattern for visualization and verification.

State update:
        - (If missing) adds unipolar and flattop CZ gate macros to each calibrated qubit pair
            (node.machine.qubit_pairs[<pair_name>].macros["cz_unipolar"]).
        - Updates their flux pulse amplitude
            (qp.macros["cz_unipolar"].flux_pulse_control.amplitude) to the fitted CZ amplitude.
        - Updates their flux pulse duration (qp.macros["cz_unipolar"].flux_pulse_control.length) to the
            fitted CZ length rounded up to the next multiple of 4 ns.
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="19_chevron_1102",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
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

    u = unit(coerce_to_integer=True)

    # Get the qubit pairs to be calibrated
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # define the amplitudes for the flux pulses
    pulse_amplitudes = {}
    for qp in qubit_pairs:
        detuning = qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency - qp.qubit_target.anharmonicity
        pulse_amplitudes[qp.name] = float(np.sqrt(-detuning / qp.qubit_control.freq_vs_flux_01_quad_term))

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

    # Pre-compute the baked short segments (1..16 samples) for each control qubit in the pairs
    baked_signals = {
        qp.qubit_control.name: baked_waveform(qp.qubit_control, baked_config, base_level=0.5, max_samples=16)
        for qp in qubit_pairs
    }

    node.namespace["baked_config"] = baked_config

    with program() as node.namespace["qua_program"]:
        t = declare(int)  # QUA variable for the flux pulse segment index
        a = declare(fixed)
        n = declare(int)
        n_st = declare_stream()
        t_left_ns = declare(int)  # QUA variable for the flux pulse segment index
        t_cycles = declare(int)  # QUA variable for the flux pulse segment index
        I_c, I_c_st, Q_c, Q_c_st, n, n_st = node.machine.declare_qua_variables()
        I_t, I_t_st, Q_t, Q_t_st, _, _ = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state_c = [declare(int) for _ in range(num_qubit_pairs)]
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
            state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
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
                            # Qubit initialization
                            qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                            align()
                            # set both qubits to the excited state
                            qp.qubit_control.xy.play("x180")
                            qp.qubit_target.xy.play("x180")

                            align()

                            # For the first 16ns we play baked pulses exclusively. Loop the time index until 16.
                            with if_(t <= 16):
                                with switch_(t):
                                    # Switch case to select the baked pulse with duration t ns
                                    for j in range(1, 17):
                                        with case_(j):
                                            baked_signals[qp.qubit_control.name][j - 1].run(
                                                amp_array=[
                                                    (qp.qubit_control.z.name, pulse_amplitudes[qp.name] / 0.5 * a)
                                                ]
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
                                        denom = qp.qubit_control.z.operations["const"].amplitude
                                        scale = (p / denom) * a
                                        qp.qubit_control.z.play(
                                            "const",
                                            duration=t_cycles,
                                            amplitude_scale=scale,
                                        )
                                    # Play the pulse multiple of 4 followed by the baked pulse of the missing duration
                                    for j in range(1, 4):
                                        with case_(j):
                                            align()
                                            p = pulse_amplitudes[qp.name]
                                            denom = qp.qubit_control.z.operations["const"].amplitude
                                            scale = (p / denom) * a
                                            qp.qubit_control.z.play(
                                                "const",
                                                duration=t_cycles,
                                                amplitude_scale=scale,
                                            )
                                            baked_signals[qp.qubit_control.name][j - 1].run(
                                                amp_array=[
                                                    (qp.qubit_control.z.name, pulse_amplitudes[qp.name] / 0.5 * a)
                                                ]
                                            )
                            align()

                            if node.parameters.use_state_discrimination:
                                qp.qubit_control.readout_state_gef(state_c[ii])
                                qp.qubit_target.readout_state(state_t[ii])
                                save(state_c[ii], state_c_st[ii])
                                save(state_t[ii], state_t_st[ii])
                            else:
                                qp.qubit_control.resonator.measure("readout", qua_vars=(I_c[ii], Q_c[ii]))
                                qp.qubit_target.resonator.measure("readout", qua_vars=(I_t[ii], Q_t[ii]))
                                save(I_c[ii], I_c_st[ii])
                                save(Q_c[ii], Q_c_st[ii])
                                save(I_t[ii], I_t_st[ii])
                                save(Q_t[ii], Q_t_st[ii])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_c_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"state_control{i}")
                    state_t_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"state_target{i}")
                else:
                    I_c_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"I_control{i}")
                    Q_c_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"Q_control{i}")
                    I_t_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"I_target{i}")
                    Q_t_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"Q_target{i}")


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
                data_fetcher["n"],
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
        detuning = qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency - qp.qubit_target.anharmonicity
        pulse_amplitudes[qp.name] = float(np.sqrt(-detuning / qp.qubit_control.freq_vs_flux_01_quad_term))
    node.namespace["pulse_amplitudes"] = pulse_amplitudes
    node.namespace["qubits"] = [qp.qubit_control for qp in qubit_pairs] + [qp.qubit_target for qp in qubit_pairs]
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
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "amplitude": fig_raw_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""

    for qp in node.namespace["qubit_pairs"]:
        if hasattr(qp.macros, "cz_unipolar"):
            continue
        else:
            print(f"Creating CZ Unipolar gate macro for {qp.name}")
            cz_pulse = SquarePulse(length=100, amplitude=0.5, id="cz_unipolar_pulse")
            cz = CZGate(flux_pulse_control=cz_pulse)
            node.machine.qubit_pairs[qp.name].macros["cz_unipolar"] = cz
            pulse_length = (
                node.machine.qubit_pairs[qp.name].macros["cz_unipolar"].flux_pulse_control.get_reference() + "/length"
            )
            pulse_amp = (
                node.machine.qubit_pairs[qp.name].macros["cz_unipolar"].flux_pulse_control.get_reference()
                + "/amplitude"
            )
            pulse_name = node.machine.qubit_pairs[qp.name].macros["cz_unipolar"].flux_pulse_control_label
            control_qb = node.machine.qubit_pairs[qp.name].qubit_control
            control_qb.z.operations[pulse_name] = SquarePulse(length=100, amplitude=0.25)
            control_qb.z.operations[pulse_name].length = pulse_length
            control_qb.z.operations[pulse_name].amplitude = pulse_amp

        if hasattr(qp.macros, "cz_flattop"):
            continue
        else:
            print(f"Creating CZ Flattop gate macro for {qp.name}")
            cz_pulse = FlatTopGaussianPulse(length=100, amplitude=0.5, flat_length=50, id="cz_flattop_pulse")
            cz = CZGate(flux_pulse_control=cz_pulse)
            node.machine.qubit_pairs[qp.name].macros["cz_flattop"] = cz
            pulse_length = (
                node.machine.qubit_pairs[qp.name].macros["cz_flattop"].flux_pulse_control.get_reference() + "/length"
            )
            flat_length = (
                node.machine.qubit_pairs[qp.name].macros["cz_flattop"].flux_pulse_control.get_reference()
                + "/flat_length"
            )
            pulse_amp = (
                node.machine.qubit_pairs[qp.name].macros["cz_flattop"].flux_pulse_control.get_reference() + "/amplitude"
            )
            pulse_name = node.machine.qubit_pairs[qp.name].macros["cz_flattop"].flux_pulse_control_label
            control_qb = node.machine.qubit_pairs[qp.name].qubit_control
            control_qb.z.operations[pulse_name] = FlatTopGaussianPulse(length=100, amplitude=0.5, flat_length=50)
            control_qb.z.operations[pulse_name].length = pulse_length
            control_qb.z.operations[pulse_name].amplitude = pulse_amp
            control_qb.z.operations[pulse_name].flat_length = flat_length

    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes[qp.name] == "failed":
                continue
            else:
                qp.macros["cz_unipolar"].flux_pulse_control.amplitude = node.results["fit_results"][qp.name]["cz_amp"]
                qp.macros["cz_flattop"].flux_pulse_control.amplitude = node.results["fit_results"][qp.name]["cz_amp"]
                # Round up to the upper 4 ns to be compatible with the hardware time resolution
                qp.macros["cz_unipolar"].flux_pulse_control.length = int(
                    np.ceil(node.results["fit_results"][qp.name]["cz_len"] / 4) * 4
                )
                qp.macros["cz_flattop"].flux_pulse_control.flat_length = int(
                    np.ceil(node.results["fit_results"][qp.name]["cz_len"] / 4) * 4
                )
                qp.macros["cz_flattop"].flux_pulse_control.length = (
                    int(np.ceil(node.results["fit_results"][qp.name]["cz_len"] / 4) * 4) + 20
                )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()


# %%

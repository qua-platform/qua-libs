# %% {Imports}
import warnings
from dataclasses import asdict, dataclass
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.chevron_cz import (
    Parameters,
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
from quam_builder.architecture.superconducting.custom_gates.cz import CZGate
from quam_config import Quam
from scipy.optimize import curve_fit

from quam.components.pulses import SquarePulse

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
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="13b_chevron_cz_1ns",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    node.parameters.qubit_pairs = ["q1-2"]
    node.parameters.reset_type = "active"
    node.parameters.use_state_discrimination = True
    # node.parameters.amp_step = 0.003
    # node.parameters.amp_range = 0.2
    # node.parameters.load_data_id = 1979
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    u = unit(coerce_to_integer=True)
    # node.namespace["qubit_pairs"] = qubit_pairs = [
    #     node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs
    # ]

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    # define the amplitudes for the flux pulses
    pulse_amplitudes = {}
    for qp in qubit_pairs:
        detuning = qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency - qp.qubit_target.anharmonicity
        pulse_amplitudes[qp.name] = float(np.sqrt(-detuning / qp.qubit_control.freq_vs_flux_01_quad_term))

    node.namespace["pulse_amplitudes"] = pulse_amplitudes

    node.namespace["qubits"] = qubits = [qp.qubit_control for qp in qubit_pairs] + [
        qp.qubit_target for qp in qubit_pairs
    ]
    num_qubits = len(qubits)
    num_qubit_pairs = len(qubit_pairs)
    n_avg = node.parameters.num_shots  # The number of averages

    # Loop parameters
    amplitudes = np.arange(1 - node.parameters.amp_range, 1 + node.parameters.amp_range, node.parameters.amp_step)
    times_cycles = np.arange(1, node.parameters.max_time_in_ns)

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "amplitude": xr.DataArray(amplitudes, attrs={"long_name": "amplitudes of the flux pulse"}),
        "time": xr.DataArray(times_cycles, attrs={"long_name": "pulse duration", "units": "ns"}),
    }

    baked_config = node.machine.generate_config()

    def baked_waveform(qubit):
        pulse_segments = []  # Stores the baking objects
        # Create the different baked sequences, each one corresponding to a different truncated duration
        waveform = [0.5] * 16

        for i in range(1, 17):  # from first item up to pulse_duration (16)
            with baking(baked_config, padding_method="right") as b:
                wf = waveform[:i]
                b.add_op(f"flux_pulse{i}", qubit.z.name, wf)
                b.play(f"flux_pulse{i}", qubit.z.name)
            # Append the baking object in the list to call it from the QUA program
            pulse_segments.append(b)

        return pulse_segments

    baked_signals = {qubits.qubit_control.name: baked_waveform(qubits.qubit_control) for qubits in qubit_pairs}

    node.namespace["baked_config"] = baked_config

    with program() as node.namespace["qua_program"]:
        t = declare(int)  # QUA variable for the flux pulse segment indexz
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

        for qubit in node.machine.active_qubits:
            node.machine.initialize_qpu(target=qubit)
            align()

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(a, amplitudes)):
                    # rest of the pulse
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

                            # play the flux pulse
                            with if_(t <= 16):
                                with switch_(t):
                                    for j in range(1, 17):
                                        with case_(j):
                                            baked_signals[qp.qubit_control.name][j - 1].run(
                                                amp_array=[
                                                    (qp.qubit_control.z.name, pulse_amplitudes[qp.name] / 0.5 * a)
                                                ]
                                            )

                            with else_():
                                assign(t_cycles, t >> 2)  # Right shift by 2 is a quick way to divide by 4
                                assign(
                                    t_left_ns, t - (t_cycles << 2)
                                )  # left shift by 2 is a quick way to multiply by 4
                                with switch_(t_left_ns):
                                    with case_(0):
                                        align()
                                        qp.qubit_control.z.play(
                                            "const",
                                            duration=t_cycles,
                                            amplitude_scale=pulse_amplitudes[qp.name]
                                            / qp.qubit_control.z.operations["const"].amplitude
                                            * a,
                                        )
                                    for j in range(1, 4):
                                        with case_(j):
                                            align()
                                            qp.qubit_control.z.play(
                                                "const",
                                                duration=t_cycles,
                                                amplitude_scale=pulse_amplitudes[qp.name]
                                                / qp.qubit_control.z.operations["const"].amplitude
                                                * a,
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
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
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
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
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
            print(f"Creating CZ gate macro for {qp.name}")
            cz_pulse = SquarePulse(length=100, amplitude=0.5, id="cz_unipolar_pulse")
            cz = CZGate(flux_pulse_control=cz_pulse)
            node.machine.qubit_pairs[qp.name].macros = {"cz_unipolar": cz}

    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes[qp.name] == "failed":
                continue
            else:
                qp.macros["cz_unipolar"].flux_pulse_control.amplitude = node.results["fit_results"][qp.name]["cz_amp"]
                qp.macros["cz_unipolar"].flux_pulse_control.length = node.results["fit_results"][qp.name]["cz_len"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
from typing import List, Literal, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.cryoscope import (
    Parameters,
    cryoscope_frequency,
    estimate_fir_coefficients,
    expdecay,
    fit_raw_data,
    plot_normalized_flux,
    plot_raw_data,
    process_raw_dataset,
    savgol,
    two_expdecay,
)
from qm.qua import *
from qualang_tools.bakery import baking
from qualang_tools.multi_user import qm_session
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit
from qualibrate import NodeParameters, QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_idle_times_in_clock_cycles, get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam
from scipy import signal
from scipy.optimize import curve_fit, minimize
from scipy.signal import convolve, deconvolve, lfilter
from sklearn.metrics import fowlkes_mallows_score

# %% {Node_parameters}
description = """
        cryoscope
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="12_cryoscope",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["qC1"]
    node.parameters.reset_type = "active"
    node.parameters.cryoscope_len = 500
    node.parameters.use_state_discrimination = True
    node.parameters.amp_factor = 0.04
    node.parameters.num_shots = 200
    # node.parameters.simulate = True
    node.parameters.simulation_duration_ns = 20_000
    node.parameters.timeout = 10000
    node.parameters.buffer = 10
    node.parameters.multiplexed = True
    # node.parameters.load_data_id = 836
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

    n_avg = node.parameters.num_shots  # The number of averages
    cryoscope_len = node.parameters.cryoscope_len  # The length of the cryoscope in nanoseconds
    amplitude_factor = node.parameters.amp_factor  # The amplitude factor for the flux pulse
    # assert cryoscope_len % 16 == 0, "cryoscope_len is not multiple of 16 nanoseconds"

    buffer = node.parameters.buffer  # Buffer time in ns

    # cryoscope_time = np.concatenate([np.arange(1 - buffer, 1, 1), np.arange(1, cryoscope_len + 1, 1)])  # x-axis for plotting - must be in ns
    cryoscope_time = np.arange(1, cryoscope_len + 1, 1)  # x-axis for plotting - must be in ns

    baked_config = node.machine.generate_config()

    def baked_waveform(waveform_amp, qubit):
        pulse_segments = []  # Stores the baking objects
        # Create the different baked sequences, each one corresponding to a different truncated duration
        waveform = [waveform_amp] * 16

        for i in range(1, 17):  # from first item up to pulse_duration (16)
            with baking(baked_config, padding_method="right") as b:
                wf = waveform[:i]
                b.add_op(f"flux_pulse{i}", qubit.z.name, wf)
                b.play(f"flux_pulse{i}", qubit.z.name)
                # b.wait(cryoscope_len - i, qubit.z.name)
            # Append the baking object in the list to call it from the QUA program
            pulse_segments.append(b)

        return pulse_segments

    baked_signals = {qubit.name: baked_waveform(amplitude_factor, qubit) for qubit in qubits}

    node.namespace["baked_config"] = baked_config

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "time": xr.DataArray(cryoscope_time, attrs={"long_name": "Cryoscope pulse duration", "units": "ns"}),
        "axis": xr.DataArray(["x", "y"], attrs={"long_name": "Tomography rotation axis"}),
    }
    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        t = declare(int)  # QUA variable for the flux pulse segment index
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        t_left_ns = declare(int)  # QUA variable for the flux pulse segment index
        t_cycles = declare(int)  # QUA variable for the flux pulse segment index
        idx = declare(int)
        flag = declare(bool)

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            # Outer loop for averaging
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                # with for_(idx, -buffer + 1, idx <= cryoscope_len, idx + 1):
                with for_(idx, 1, idx <= cryoscope_len, idx + 1):
                    # Alternate between X/2 and Y/2 pulses
                    # for tomo in ['x90', 'y90']:
                    with for_each_(flag, [True, False]):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()

                        # with if_(idx <= 0):
                        #     align()
                        #     for i, qubit in multiplexed_qubits.items():
                        #         qubit.xy.play("x90")
                        #         qubit.xy.wait((cryoscope_len + 32) // 4)
                        #         # Play second X/2 or Y/2
                        #         # if tomo == 'x90':
                        #         with if_(flag):
                        #             qubit.xy.play("x90")
                        #         # elif tomo == 'y90':
                        #         with else_():
                        #             qubit.xy.play("y90")

                        # with elif_(((idx > 0) & (idx <= 16))):
                        with if_(idx <= 16):
                            with switch_(idx):
                                for j in range(1, 17):
                                    with case_(j):
                                        align()
                                        for i, qubit in multiplexed_qubits.items():
                                            qubit.xy.play("x90")
                                            qubit.z.wait((qubit.xy.operations["x90"].length + 16) // 4)
                                            baked_signals[qubit.name][j - 1].run()
                                            qubit.xy.wait((cryoscope_len + 32) // 4)
                                            # Play second X/2 or Y/2
                                            # if tomo == 'x90':
                                            with if_(flag):
                                                qubit.xy.play("x90")
                                            # elif tomo == 'y90':
                                            with else_():
                                                qubit.xy.play("y90")

                        with else_():
                            assign(t_cycles, idx >> 2)  # Right shift by 2 is a quick way to divide by 4
                            assign(t_left_ns, idx - (t_cycles << 2))  # left shift by 2 is a quick way to multiply by 4
                            with switch_(t_left_ns):
                                with case_(0):
                                    align()
                                    for i, qubit in multiplexed_qubits.items():
                                        qubit.xy.play("x90")
                                        qubit.z.wait((qubit.xy.operations["x90"].length + 16) // 4)
                                        qubit.z.play(
                                            "const",
                                            duration=t_cycles,
                                            amplitude_scale=amplitude_factor / qubit.z.operations["const"].amplitude,
                                        )
                                        qubit.xy.wait((cryoscope_len + 32) // 4)
                                        # Play second X/2 or Y/2
                                        # if tomo == 'x90':
                                        with if_(flag):
                                            qubit.xy.play("x90")
                                        # elif tomo == 'y90':
                                        with else_():
                                            qubit.xy.play("y90")
                                for j in range(1, 4):
                                    with case_(j):
                                        align()
                                        for i, qubit in multiplexed_qubits.items():
                                            qubit.xy.play("x90")
                                            qubit.z.wait((qubit.xy.operations["x90"].length + 16) // 4)
                                            qubit.z.play(
                                                "const",
                                                duration=t_cycles,
                                                amplitude_scale=amplitude_factor
                                                / qubit.z.operations["const"].amplitude,
                                            )
                                            baked_signals[qubit.name][j - 1].run()
                                            qubit.xy.wait((cryoscope_len + 32) // 4)
                                            # Play second X/2 or Y/2
                                            # if tomo == 'x90':
                                            with if_(flag):
                                                qubit.xy.play("x90")
                                            # elif tomo == 'y90':
                                            with else_():
                                                qubit.xy.play("y90")
                        # Wait for the idle time set slightly above the maximum flux pulse duration
                        # to ensure that the 2nd x90 pulse arrives after the longest flux pulse

                        # Measure resonator state after the sequence
                        align()
                        for i, qubit in multiplexed_qubits.items():
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])

        with stream_processing():
            # for the progress counter
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    # state_st[i].buffer(2).buffer(cryoscope_len + buffer).average().save(f"state{i + 1}")
                    state_st[i].buffer(2).buffer(cryoscope_len).average().save(f"state{i + 1}")
                else:
                    # I_st[i].buffer(2).buffer(cryoscope_len + buffer).average().save(f"I{i + 1}")
                    # Q_st[i].buffer(2).buffer(cryoscope_len + buffer).average().save(f"Q{i + 1}")
                    I_st[i].buffer(2).buffer(cryoscope_len).average().save(f"I{i + 1}")
                    Q_st[i].buffer(2).buffer(cryoscope_len).average().save(f"Q{i + 1}")


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
    # print(f"QPU time: {node.namespace["job"].result_handles.get("__qpu_execution_time_seconds").fetch_all()}")
    # print(f"Total Python time: {node.namespace["job"].result_handles.get("__total_python_runtime_seconds").fetch_all()}")

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
    node.results["ds_fit"] = fit_raw_data(node.results["ds_raw"], node)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """
    Plot the raw and fitted data in specific figures whose shape is given by
    qubit.grid_location.
    """

    fig_raw = plot_raw_data(node.results["ds_raw"], node.namespace["qubits"], fits=node.results["ds_fit"])
    fig_flux = plot_normalized_flux(node.results["ds_raw"], node.namespace["qubits"], fits=node.results["ds_fit"])

    node.results["figure_raw"] = fig_raw
    node.results["figure_flux"] = fig_flux
    # node.save()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            # if node.outcomes[q.name] == "failed":
            #     continue

            fit_1exp = node.results["ds_fit"].sel(qubit=q.name).fit_results.fit_1exp
            fit_2exp = node.results["ds_fit"].sel(qubit=q.name).fit_results.fit_2exp

            print(f"fit_1exp: {fit_1exp}")
            print(f"fit_2exp: {fit_2exp}")

            node.machine.qubits[q.name].z.opx_output.exponential_filter = [
                (-fit_2exp[1], fit_2exp[2]),
                (-fit_2exp[3], fit_2exp[4]),
            ]

            node.save()


# %%

# %% {Imports}
from dataclasses import asdict

import numpy as np
import xarray as xr
from calibration_utils.cryoscope import (
    Parameters,
    baked_waveform,
    fit_raw_data,
    log_fitted_results,
    plot_fit,
    process_raw_dataset,
)
from qm.qua import *
from qualang_tools.bakery import baking
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Node_parameters}
description = """
CRYOSCOPE
The goal of this protocol is to measure the step response of the flux line and design
proper FIR and IIR filters (implemented on the OPX) to pre-distort the flux pulses and
improve the two-qubit gates fidelity. Since the flux line ends on the qubit chip, it is
not possible to measure the flux pulse after propagation through the fridge. The idea is
to exploit the flux dependency of the qubit frequency, measured with a modified Ramsey
sequence, to estimate the flux amplitude received by the qubit as a function of time.

The sequence consists of a Ramsey sequence ("x90" - idle time - "x90" or "y90") with a
fixed dephasing time. A flux pulse with varying duration is played during the idle time.
The Sx and Sy components of the Bloch vector are measured by alternatively closing the
Ramsey sequence with a "x90" or "y90" gate in order to extract the qubit dephasing as a
function of the flux pulse duration.

The results are then post-processed to retrieve the step function of the flux line which
is fitted with an exponential function. The corresponding exponential parameters are
then used to derive the FIR and IIR filter taps that will compensate for the distortions
introduced by the flux line (wiring, bias-tee...). Such digital filters are then
implemented on the OPX. Note that these filters will introduce a global delay on all the
output channels that may rotate the IQ blobs so that you may need to recalibrate them for
state discrimination or active reset protocols. More details on these filters:
https://docs.quantum-machines.co/0.1/qm-qua-sdk/docs/Guides/output_filter/?h=filter#hardware-implementation

The protocol is inspired from https://doi.org/10.1063/1.5133894, which contains more
details about the sequence and the post-processing of the data.

This version sweeps the flux pulse duration using the baking tool, which means that the
flux pulse can be scanned with a 1ns resolution, but must be shorter than ~260ns. For
longer pulses either reduce the resolution (2ns steps) or use the 4ns version
(`cryoscope_4ns.py`).

Prerequisites:
        - Resonator spectroscopy performed.
        - Qubit gates (x90, y90) calibrated: spectroscopy, rabi_chevron, power_rabi, Ramsey
            and configuration updated.

Next steps before going to the next node:
        - Update the FIR and IIR filter taps in the configuration:
                - OPX+: config/controllers/con1/analog_outputs/"filter": {"feedforward": fir,
                    "feedback": iir}
                - OPX1000: config/controllers/con1/analog_outputs/"filter": {"feedforward": [],
                    "exponential": [(A, tau)]}
        - WARNING: digital filters add a global delay: recalibrate IQ blobs (rotation_angle &
            ge_threshold).
"""
# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="18_cryoscope",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1"]
    pass


# Instantiate the QUAM class from the state file
node.machine = stored_machine = Quam.load()

loaded_fractions = node.parameters.exponential_fit_time_fractions
stored_gui_update_flag = node.parameters.update_state_from_GUI


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    qubit = qubits[0]

    assert num_qubits == 1, "This node only supports one qubit at the time."

    n_avg = node.parameters.num_shots  # The number of averages
    cryoscope_len = node.parameters.cryoscope_len  # The length of the cryoscope in nanoseconds

    # Absolute amplitude of the Cryoscope pulse
    amplitude = float(np.sqrt(-node.parameters.detuning_target_in_MHz * 1e6 / qubits[0].freq_vs_flux_01_quad_term))

    cryoscope_time = np.arange(1, cryoscope_len + 1, 1)  # x-axis for plotting - must be in ns

    frames = np.linspace(0, 1, node.parameters.num_frames)

    baked_config = node.machine.generate_config()

    baked_signals = {qubit.name: baked_waveform(baked_config, amplitude, qubit, max_length=16) for qubit in qubits}

    node.namespace["baked_config"] = baked_config

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "time": xr.DataArray(cryoscope_time, attrs={"long_name": "Cryoscope pulse duration", "units": "ns"}),
        "frame": xr.DataArray(frames, attrs={"long_name": "Frame rotation index"}),
    }
    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        t_left_ns = declare(int)  # QUA variable for the remainding ns to add to the flux pulse multiple of 4
        t_cycles = declare(int)  # QUA variable for the flux pulse multiple of 4
        idx = declare(int)
        frame = declare(fixed)

        # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)

        node.machine.initialize_qpu(target=qubit)
        align()

        # Outer loop for averaging
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            # Loop over the cryoscope pulse time duration (idx represents the duration in ns)
            with for_(idx, 1, idx <= cryoscope_len, idx + 1):
                # Loop over the phase of the second ramsey x90 pulse to reconstruct the qubit phase
                with for_each_(frame, frames):
                    # Qubit initialization
                    qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()
                    ################################################################################################
                    # The duration argument in the play command can only produce pulses with duration multiple of  #
                    # 4ns. To overcome this limitation we use the baking tool from the qualang-tools package to    #
                    # generate pulses with 1ns granularity. To avoid creating custom waveforms for each iteration  #
                    # we combine baked pulses with dynamically stretched (multiple of 4ns) pulses.                 #
                    ################################################################################################
                    # For the first 16ns we play baked pulses exclusively. Loop the time idx counter until 16.
                    with if_(idx <= 16):
                        # Swich case to select the baked pulse with duration idx ns
                        with switch_(idx):
                            for j in range(1, 17):
                                # The Ramsey sequence is embedded in the switch case to allow gapless execution
                                with case_(j):
                                    align()
                                    qubit.xy.play("x90")
                                    qubit.z.wait((qubit.xy.operations["x90"].length + 16) // 4)
                                    baked_signals[qubit.name][j - 1].run()  # Play the baked pulse
                                    qubit.xy.wait((cryoscope_len + 16) >> 2)  # 16ns buffer between pulses
                                    qubit.xy.frame_rotation_2pi(frame)
                                    qubit.xy.play("x90")
                    # For pulse durations above 16ns we combine baking with regular play statements.
                    with else_():
                        # We calculate the closest lower multiple of 4 of the time index
                        assign(t_cycles, idx >> 2)  # Right shift by 2 is a quick way to divide by 4
                        # Calculate the duration to add to pulse multiple of 4.
                        assign(t_left_ns, idx - (t_cycles << 2))  # left shift by 2 is a quick way to multiply by 4
                        # Switch case with the 4 possible sequences:
                        with switch_(t_left_ns):
                            # Play only the pulse multiple of 4
                            with case_(0):
                                align()
                                qubit.xy.play("x90")
                                qubit.z.wait((qubit.xy.operations["x90"].length + 16) // 4)
                                qubit.z.play(
                                    "const",
                                    duration=t_cycles,
                                    amplitude_scale=amplitude / qubit.z.operations["const"].amplitude,
                                )
                                qubit.xy.wait((cryoscope_len + 16) // 4)
                                qubit.xy.frame_rotation_2pi(frame)
                                qubit.xy.play("x90")
                            # Play the pulse multiple of 4 followed by the baked pulse of the missing duration
                            for j in range(1, 4):
                                with case_(j):
                                    align()
                                    qubit.xy.play("x90")
                                    qubit.z.wait((qubit.xy.operations["x90"].length + 16) // 4)
                                    qubit.z.play(
                                        "const",
                                        duration=t_cycles,
                                        amplitude_scale=amplitude / qubit.z.operations["const"].amplitude,
                                    )
                                    baked_signals[qubit.name][j - 1].run()
                                    qubit.xy.wait((cryoscope_len + 16) // 4)
                                    qubit.xy.frame_rotation_2pi(frame)
                                    qubit.xy.play("x90")
                    # Wait for the idle time set slightly above the maximum flux pulse duration
                    # to ensure that the 2nd x90 pulse arrives after the longest flux pulse

                    # Measure resonator state after the sequence
                    align()

                    if node.parameters.use_state_discrimination:
                        qubit.readout_state(state[0])
                        save(state[0], state_st[0])
                    else:
                        qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
                        save(I[0], I_st[0])
                        save(Q[0], Q_st[0])

        with stream_processing():
            # for the progress counter
            n_st.save("n")
            if node.parameters.use_state_discrimination:
                state_st[0].buffer(len(frames)).buffer(cryoscope_len).average().save("state1")
            else:
                I_st[0].buffer(len(frames)).buffer(cryoscope_len).average().save("I1")
                Q_st[0].buffer(len(frames)).buffer(cryoscope_len).average().save("Q1")


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
    """Connect to the QOP, execute the QUA program, fetch the raw data and store it
    in an xarray dataset called "ds_raw".
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
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)
    node.parameters.exponential_fit_time_fractions = loaded_fractions
    node.parameters.update_state_from_GUI = stored_gui_update_flag
    if node.parameters.update_state_from_GUI:
        node.machine = stored_machine
        node.parameters.update_state = True
        print("State update from GUI is enabled")


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data, store the fitted data in an xarray dataset "ds_fit" and
    the fitted results in the "fit_results" dictionary.
    """
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)

    # Log the relevant information extracted from the data analysis
    log_fitted_results(fit_results, log_callable=node.log)

    # Convert to dict format for storage and create outcomes
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    node.outcomes = {
        qubit_name: ("successful" if fit_result.success else "failed") for qubit_name, fit_result in fit_results.items()
    }


# % {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """
    Plot the raw and fitted data in specific figures whose shape is given by
    qubit.grid_location.
    """

    fig_flux = plot_fit(node.results["ds_fit"], node.namespace["qubits"], fits=node.results["ds_fit"])

    node.results["figure_flux"] = fig_flux


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    if not node.parameters.update_state:
        return
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

        components = node.results["fit_results"][q.name]["components"]
        a_dc = node.results["fit_results"][q.name]["a_dc"]
        A_list = [amp / a_dc for amp, _ in components]
        tau_list = [tau for _, tau in components]
        node.machine.qubits[q.name].z.opx_output.exponential_filter.extend(list(zip(A_list, tau_list)))


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()


# %%

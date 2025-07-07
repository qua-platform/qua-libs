# %% {Imports}
from dataclasses import asdict

import numpy as np
import xarray as xr
from calibration_utils.cryoscope import Parameters, fit_raw_data, plot_normalized_flux, process_raw_dataset
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
    The goal of this protocol is to measure the step response of the flux line and design proper FIR and IIR filters
    (implemented on the OPX) to pre-distort the flux pulses and improve the two-qubit gates fidelity.
    Since the flux line ends on the qubit chip, it is not possible to measure the flux pulse after propagation through the
    fridge. The idea is to exploit the flux dependency of the qubit frequency, measured with a modified Ramsey sequence, to
    estimate the flux amplitude received by the qubit as a function of time.

    The sequence consists of a Ramsey sequence ("x90" - idle time - "x90" or "y90") with a fixed dephasing time.
    A flux pulse with varying duration is played during the idle time. The Sx and Sy components of the Bloch vector are
    measured by alternatively closing the Ramsey sequence with a "x90" or "y90" gate in order to extract the qubit dephasing
    as a function of the flux pulse duration.

    The results are then post-processed to retrieve the step function of the flux line which is fitted with an exponential
    function. The corresponding exponential parameters are then used to derive the FIR and IIR filter taps that will
    compensate for the distortions introduced by the flux line (wiring, bias-tee...).
    Such digital filters are then implemented on the OPX. Note that these filters will introduce a global delay on all the
    output channels that may rotate the IQ blobs so that you may need to recalibrate them for state discrimination or
    active reset protocols for instance. You can read more about these filters here:
    https://docs.quantum-machines.co/0.1/qm-qua-sdk/docs/Guides/output_filter/?h=filter#hardware-implementation

    The protocol is inspired from https://doi.org/10.1063/1.5133894, which contains more details about the sequence and
    the post-processing of the data.

    This version sweeps the flux pulse duration using the baking tool, which means that the flux pulse can be scanned with
    a 1ns resolution, but must be shorter than ~260ns. If you want to measure longer flux pulse, you can either reduce the
    resolution (do 2ns steps instead of 1ns) or use the 4ns version (cryoscope_4ns.py).

    Prerequisites:
        - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
        - Having calibrated qubit gates (x90 and y90) by running qubit spectroscopy, rabi_chevron, power_rabi, Ramsey and updated the configuration.

    Next steps before going to the next node:
        - Update the FIR and IIR filter taps in the configuration:
            - For OPX+: (config/controllers/con1/analog_outputs/"filter": {"feedforward": fir, "feedback": iir}).
            - For OPX1000: (config/controllers/con1/analog_outputs/"filter": {"feedforward": [], "exponential": [(A, tau)]}).
        - WARNING: the digital filters will add a global delay --> need to recalibrate IQ blobs (rotation_angle & ge_threshold).
"""
# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="12b_cryoscope",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.load_data_id = 1289
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

    cryoscope_time = np.arange(1, cryoscope_len + 1, 1)  # x-axis for plotting - must be in ns

    frames = np.linspace(0, 1, node.parameters.num_frames)

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
            # Append the baking object in the list to call it from the QUA program
            pulse_segments.append(b)

        return pulse_segments

    baked_signals = {qubit.name: baked_waveform(amplitude_factor, qubit) for qubit in qubits}

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
        t_left_ns = declare(int)  # QUA variable for the flux pulse segment index
        t_cycles = declare(int)  # QUA variable for the flux pulse segment index
        idx = declare(int)
        frame = declare(fixed)

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            # Outer loop for averaging
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(idx, 1, idx <= cryoscope_len, idx + 1):
                    with for_each_(frame, frames):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()

                        with if_(idx <= 16):
                            with switch_(idx):
                                for j in range(1, 17):
                                    with case_(j):
                                        align()
                                        for i, qubit in multiplexed_qubits.items():
                                            qubit.xy.play("x90")
                                            qubit.z.wait((qubit.xy.operations["x90"].length + 16) // 4)
                                            baked_signals[qubit.name][j - 1].run()
                                            qubit.xy.wait((cryoscope_len + 16) // 4)  # 16ns buffer between pulses
                                            qubit.xy.frame_rotation_2pi(frame)
                                            qubit.xy.play("x90")

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
                                        qubit.xy.wait((cryoscope_len + 16) // 4)
                                        qubit.xy.frame_rotation_2pi(frame)
                                        qubit.xy.play("x90")
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
                                            qubit.xy.wait((cryoscope_len + 16) // 4)
                                            qubit.xy.frame_rotation_2pi(frame)
                                            qubit.xy.play("x90")
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
                    state_st[i].buffer(len(frames)).buffer(cryoscope_len).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(frames)).buffer(cryoscope_len).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(frames)).buffer(cryoscope_len).average().save(f"Q{i + 1}")


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
    node.parameters.exp_1_tau_guess = 10


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""

    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """
    Plot the raw and fitted data in specific figures whose shape is given by
    qubit.grid_location.
    """
    fig_flux = plot_normalized_flux(node.results["ds_raw"], node.namespace["qubits"], fits=node.results["ds_fit"])
    node.results["figure_flux"] = fig_flux


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:

            if node.parameters.number_of_exponents == 1:
                if node.results["fit_results"][q.name]["fit1_success"]:
                    node.machine.qubits[q.name].z.opx_output.exponential_filter = [
                        (
                            node.results["fit_results"][q.name]["fit1_A"],
                            node.results["fit_results"][q.name]["fit1_tau"],
                        ),
                    ]
                else:
                    print(f"Warning: fit_1exp for qubit {q.name} was not successful. No filter will be applied.")

            elif node.parameters.number_of_exponents == 2:
                if node.results["fit_results"][q.name]["fit2_success"]:
                    node.machine.qubits[q.name].z.opx_output.exponential_filter = [
                        (
                            node.results["fit_results"][q.name]["fit2_A1"],
                            node.results["fit_results"][q.name]["fit2_tau1"],
                        ),
                        (
                            node.results["fit_results"][q.name]["fit2_A2"],
                            node.results["fit_results"][q.name]["fit2_tau2"],
                        ),
                    ]
                else:
                    print(f"Warning: fit_2exp for qubit {q.name} was not successful. No filter will be applied.")

            node.save()

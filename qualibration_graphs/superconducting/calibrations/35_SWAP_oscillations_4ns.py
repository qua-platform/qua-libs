"""
Rabi Chevron Calibration:

This sequence measures the time and detuning required for a Rabi chevron. The process involves:

1. Preparing both qubits in their excited states.
2. Applying a flux pulse with varying amplitude and duration.
3. Measuring the resulting state populations as a function of these parameters.
4. Fitting the results to a Ramsey-Chevron pattern.

Outcomes:
- Extracted coupling strength (J2) between the qubits.
- Optimal flux pulse amplitude and duration for a CPhase gate.
"""
from dataclasses import asdict

from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot

from quam_config import Quam
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm.qua import *
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.swap_oscillations.parameters import Parameters
from calibration_utils.swap_oscillations.analysis import (
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    FitParameters
)
from calibration_utils.swap_oscillations.plotting import plot_control_state, plot_target_state, plot_raw_data_with_fit

# %% {Description}
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

The Ramsey-Chevron pattern emerges due to the interplay between the qubit-qubit coupling and the flux-induced detuning, allowing us to precisely calibrate the CPhase gate.

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair.
- Calibrated readout for both qubits.
- Initial estimate of the flux pulse amplitude range.

Outcomes:
- Extracted J2 coupling strength.
- Optimal flux pulse amplitude and duration for the CPhase gate.
- Fitted Ramsey-Chevron pattern for visualization and verification.
"""

node = QualibrationNode(
    name="swap_oscillations",
    description=description,
    parameters=Parameters()
)

# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubit_pairs = ["q0-1"]
    node.parameters.reset_type = "active"
    node.parameters.use_state_discrimination = True
    node.parameters.min_wait_time_in_ns  = 16
    node.parameters.max_wait_time_in_ns  = 250
    node.parameters.wait_time_num_points  = 100
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()





n_avg = node.parameters.num_averages
flux_point = node.parameters.flux_point_joint_or_independent


amplitudes = np.arange(1 - node.parameters.amp_range, 1 + node.parameters.amp_range, node.parameters.amp_step)
times_cycles = np.arange(0, node.parameters.max_time_in_ns // 4)


@node.run_action()
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and node parameters."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    # Define and store the amplitudes for the flux pulses
    pulse_amplitudes = {}
    for qp in qubit_pairs:
        detuning = qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency
        pulse_amplitudes[qp.name] = float(np.sqrt(-detuning / qp.qubit_control.freq_vs_flux_01_quad_term))
    node.namespace["pulse_amplitudes"] = pulse_amplitudes

    node.namespace["qubits"] = qubits = [qp.qubit_control for qp in qubit_pairs] + [
        qp.qubit_target for qp in qubit_pairs
    ]

    num_qubits = len(qubits)
    num_qubit_pairs = len(qubit_pairs)
    # Loop parameters
    n_avg = node.parameters.num_shots  # The number of averages
    amplitudes = np.arange(1 - node.parameters.amp_range, 1 + node.parameters.amp_range, node.parameters.amp_step)
    times_cycles = np.arange(0, node.parameters.max_time_in_ns // 4)


    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "amplitude": xr.DataArray(amplitudes, attrs={"long_name": "amplitudes of the flux pulse"}),
        "time": xr.DataArray(times_cycles, attrs={"long_name": "pulse duration", "units": "ns"}),
    }

    with program() as node.namespace["qua_program"]:
        t = declare(int)
        amp = declare(fixed)
        n = declare(int)
        n_st = declare_stream()
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
            wait(1000)

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(amp, amplitudes)):
                    with for_(*from_array(t, times_cycles)):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            # Qubit initialization
                            qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                            align()
                            # set control qubits to the excited state
                            qp.qubit_control.xy.play("x180")

                            qp.align()

                            with if_(t > 4):
                                qp.qubit_control.z.play(
                                    'const',
                                    duration=t,
                                    amplitude_scale=pulse_amplitudes[qp.name] / qp.qubit_control.z.operations[
                                        'const'].amplitude * amp
                                )
                            # wait for the flux pulse to end and some extra time
                            for qubit in [qp.qubit_control, qp.qubit_target]:
                                qubit.xy.wait(node.parameters.max_time_in_ns // 4 + 10)
                            qp.align()

                            # measure both qubits
                            if node.parameters.use_state_discrimination:
                                qp.qubit_control.readout_state(state_c[ii])
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
        detuning = qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency
        pulse_amplitudes[qp.name] = float(np.sqrt(-detuning / qp.qubit_control.freq_vs_flux_01_quad_term))
    node.namespace["pulse_amplitudes"] = pulse_amplitudes
    node.namespace["qubits"] = [qp.qubit_control for qp in qubit_pairs] + [qp.qubit_target for qp in qubit_pairs]
    node.namespace["qubit_pairs"] = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]



# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results,  amplitudes, lengths, zero_paddings, detunings = fit_raw_data(node.results["ds_raw"], node, pulse_amplitudes, qubit_pairs)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}



    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }



@node.run_action()
def plot_data(node: QualibrationNode[Parameters, Quam]):
    grid_control = plot_control_state(node.results["ds_fit"], qubit_pairs)
    grid_target = plot_target_state(node.results["ds_fit"], qubit_pairs)
    plt.show()
    node.results["figure_control"] = grid_control.fig
    node.results["figure_target"] = grid_target.fig


@node.run_action()
def update_state(node: QualibrationNode[Parameters, Quam]):
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qp in qubit_pairs:
                fp = node.results["fit_results"][qp.name]
                qp.gates['SWAP_unipolar'] = CZGate(
                    flux_pulse_control=FluxPulse(
                        length=fp.optimal_length,
                        amplitude=fp.optimal_amplitude / 2,
                        zero_padding=fp.zero_padding,
                        id='flux_pulse_control_' + qp.qubit_target.name
                    )
                )
                qp.gates['SWAP'] = f"#./SWAP_unipolar"

                qp.J2 = fp.J
                qp.detuning = fp.detuning


@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
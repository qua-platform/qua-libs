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

from qualibrate import QualibrationNode
from quam_config import Quam
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm.qua import *
import matplotlib.pyplot as plt
import numpy as np

from calibration_utils.swap_oscillations.parameters import Parameters
from calibration_utils.swap_oscillations.analysis import (
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    FitParameters
)
from calibration_utils.swap_oscillations.plotting import plot_control_state, plot_target_state



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
    # node.parameters.qubits = ["q1", "q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()



if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

num_qubit_pairs = len(qubit_pairs)

config = machine.generate_config()

if node.parameters.load_data_id is None:
    qmm = machine.connect()

n_avg = node.parameters.num_averages
flux_point = node.parameters.flux_point_joint_or_independent

# Define and store the amplitudes for the flux pulses
pulse_amplitudes = {}
for qp in qubit_pairs:
    detuning = qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency
    pulse_amplitudes[qp.name] = float(np.sqrt(-detuning / qp.qubit_control.freq_vs_flux_01_quad_term))

node.namespace["pulse_amplitudes"] = pulse_amplitudes
node.namespace["qubit_pairs"] = qubit_pairs

amplitudes = np.arange(1 - node.parameters.amp_range, 1 + node.parameters.amp_range, node.parameters.amp_step)
times_cycles = np.arange(0, node.parameters.max_time_in_ns // 4)


@node.run_action()
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)
    """Create the sweep axes and generate the QUA program from the pulse sequence and node parameters."""
    with program() as rabi_chevron:
        t = declare(int)
        amp = declare(fixed)
        n = declare(int)
        n_st = declare_stream()
        state_control = [declare(int) for _ in range(num_qubit_pairs)]
        state_target = [declare(int) for _ in range(num_qubit_pairs)]
        state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]

        for i, qp in enumerate(qubit_pairs):
            if flux_point == "independent":
                machine.apply_all_flux_to_min()
            elif flux_point == "joint":
                machine.apply_all_flux_to_joint_idle()
            else:
                machine.apply_all_flux_to_zero()
            wait(1000)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(amp, amplitudes)):
                    with for_(*from_array(t, times_cycles)):
                        if node.parameters.reset_type == "active":
                            active_reset(qp.qubit_control)
                            qp.align()
                            active_reset(qp.qubit_target)
                            qp.align()
                        else:
                            wait(qp.qubit_control.thermalization_time * u.ns)

                        qp.qubit_control.xy.play("x180")
                        qp.align()

                        with if_(t > 0):
                            qp.qubit_control.z.play(
                                'const',
                                duration=t,
                                amplitude_scale=pulse_amplitudes[qp.name] / qp.qubit_control.z.operations[
                                    'const'].amplitude * amp
                            )

                        for qubit in [qp.qubit_control, qp.qubit_target]:
                            qubit.xy.wait(node.parameters.max_time_in_ns // 4 + 10)
                        qp.align()

                        readout_state(qp.qubit_control, state_control[i])
                        readout_state(qp.qubit_target, state_target[i])
                        save(state_control[i], state_st_control[i])
                        save(state_target[i], state_st_target[i])

            align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                state_st_control[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(
                    f"state_control{i + 1}")
                state_st_target[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(
                    f"state_target{i + 1}")

    node.namespace["qua_program"] = rabi_chevron


@node.run_action()
def simulate_or_execute(node: QualibrationNode[Parameters, QuAM]):
    if node.parameters.simulate:
        simulation_config = SimulationConfig(duration=10_000)
        job = qmm.simulate(config, node.namespace["qua_program"], simulation_config)
        job.get_simulated_samples().con1.plot()
        node.results["simulation"] = {"figure": plt.gcf()}
        node.machine = machine
        node.save()
    elif node.parameters.load_data_id is None:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(node.namespace["qua_program"])

            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                n = results.fetch_all()[0]
                progress_counter(n, n_avg, start_time=results.start_time)


@node.run_action()
def fetch_and_process_data(node: QualibrationNode[Parameters, Quam]):
    if node.parameters.load_data_id is None:
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"time": 4 * times_cycles, "amp": amplitudes})
        node.results["ds"] = ds
    else:
        ds, loaded_machine = load_dataset(node.parameters.load_data_id)
        if loaded_machine is not None:
            machine = loaded_machine

    # Process and fit the data
    node.results["ds_raw"] = process_raw_dataset(ds, node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the fitted results
    log_fitted_results(node.results["fit_results"], log_callable=node.log)


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
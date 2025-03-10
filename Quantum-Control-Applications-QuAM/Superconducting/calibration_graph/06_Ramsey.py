# %% {Imports}
from qm.qua import *
import matplotlib.pyplot as plt
from dataclasses import asdict
import xarray as xr

from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array

from quam_config import QuAM
from qualibrate import QualibrationNode
from qualibrate.utils.logger_m import logger

# Import experiment‑specific helpers from the Ramsey module.
from quam_experiments.experiments.ramsey.parameters import Parameters, get_idle_times_in_clock_cycles
from quam_experiments.experiments.ramsey.analysis.fitting import fit_frequency_detuning_and_t2_decay
from quam_experiments.experiments.ramsey.plotting import plot_ramseys_data_with_fit
from quam_experiments.parameters.qubits_experiment import get_qubits
from quam_experiments.workflow.simulation import simulate_and_plot

# %% {Node_parameters}
description = """
        RAMSEY WITH VIRTUAL Z ROTATIONS
The program consists in playing a Ramsey sequence (x90 – idle_time – x90 – measurement) for different idle times.
Instead of detuning the qubit gates, the frame of the second x90 pulse is rotated (de-phased) to mimic an accumulated
phase acquired for a given detuning after the idle time.
From the results, one can fit the Ramsey oscillations and precisely measure the qubit resonance frequency and T2*.

Prerequisites:
    - The resonator frequency must be determined (resonator_spectroscopy).
    - Qubit pulses (x90, x180) must be calibrated.
    - (Optional) Readout calibration (readout_frequency, amplitude, duration_optimization IQ_blobs) improves SNR.

State update:
    - Update the qubits’ frequency and T2_ramsey in the state.
    - Save the current state.
"""

node = QualibrationNode[Parameters, QuAM](
    name="06_Ramsey",
    description=description,
    parameters=Parameters(
        qubits=None,
        num_averages=100,
        frequency_detuning_in_mhz=1.0,
        min_wait_time_in_ns=16,
        max_wait_time_in_ns=3000,
        wait_time_num_points=500,
        log_or_linear_sweep="log",
        use_state_discrimination=False,
        multiplexed=False,
        simulate=False,
    ),
)

# Immediately load the machine.
node.machine = QuAM.load()


# %% {Custom Parameter Action}
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, QuAM]):
    # Adjust parameters for debugging/development when running from the IDE.
    node.parameters.num_averages = 5000
    pass


# %% {QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, QuAM]):
    """
    Create the sweep axes and generate the QUA program for the Ramsey sequence with virtual Z rotations.
    """
    # Instantiate the unit converter locally.
    u = unit(coerce_to_integer=True)
    # Retrieve active qubits via the helper.
    node.namespace["qubits"] = get_qubits(node)
    qubits = get_qubits(node)
    num_qubits = len(qubits)
    n_avg = node.parameters.num_averages
    idle_times = get_idle_times_in_clock_cycles(node.parameters)
    # Frequency detuning in rad/s (using u.MHz) for virtual phase accumulation.
    detuning = node.parameters.frequency_detuning_in_mhz * u.MHz
    # Define the two possible detuning signs.
    detuning_signs = [-1, 1]
    # Register sweep axes: qubit names, idle_time (converted to ns) and detuning_sign.
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray([q.name for q in qubits], attrs={"long_name": "qubit"}),
        "idle_time": xr.DataArray(4 * idle_times, attrs={"long_name": "idle time", "units": "ns"}),
        "detuning_sign": xr.DataArray(detuning_signs, attrs={"long_name": "detuning sign"}),
    }
    # Determine flux bias mode (assumed to be part of Parameters).
    flux_point = node.parameters.flux_point_joint_or_independent
    # Build the QUA program.
    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.qua_declaration()
        idle_time = declare(int)
        detuning_sign = declare(int)
        # Declare virtual detuning phases for each qubit.
        virtual_detuning_phases = [declare(fixed) for _ in range(num_qubits)]
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        # Loop over qubits using batching.
        for multiplexed_qubits in qubits.batch():
            # Initialize each qubit.
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_each_(idle_time, idle_times):
                    with for_(*from_array(detuning_sign, detuning_signs)):
                        # For each qubit in the current batch, compute the virtual phase.
                        for i, qubit in multiplexed_qubits.items():
                            with if_(detuning_sign == 1):
                                assign(
                                    virtual_detuning_phases[i], Cast.mul_fixed_by_int(detuning * 1e-9, 4 * idle_time)
                                )
                            with else_():
                                assign(
                                    virtual_detuning_phases[i], Cast.mul_fixed_by_int(-detuning * 1e-9, 4 * idle_time)
                                )
                        align()
                        # Pulse sequence: x90 – apply virtual rotation – idle – x90.
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("x90")
                            qubit.xy.frame_rotation_2pi(virtual_detuning_phases[i])
                            qubit.xy.wait(idle_time)
                            qubit.xy.play("x90")
                        align()
                        # Measurement.
                        for i, qubit in multiplexed_qubits.items():
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])
                        align()
                        # Reset qubits.
                        for i, qubit in multiplexed_qubits.items():
                            qubit.resonator.wait(qubit.thermalization_time * u.ns)
                            reset_frame(qubit.xy.name)
            align()
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(detuning_signs)).buffer(len(idle_times)).average().save(f"state{i+1}")
                else:
                    I_st[i].buffer(len(detuning_signs)).buffer(len(idle_times)).average().save(f"I{i+1}")
                    Q_st[i].buffer(len(detuning_signs)).buffer(len(idle_times)).average().save(f"Q{i+1}")


# %% {Simulate_or_execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, QuAM]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, QuAM]):
    """
    Connect to the QOP, execute the QUA program, and fetch the raw data into a xarray dataset.
    (Data fetching is handled here via XarrayDataFetcher.)
    """
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(data_fetcher["n"], node.parameters.num_averages, start_time=data_fetcher.t_start)
        print(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Data_loading_and_dataset_creation}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, QuAM]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Data_analysis}
@node.run_action(skip_if=node.parameters.simulate)
def data_analysis(node: QualibrationNode[Parameters, QuAM]):
    """
    Analyze the raw Ramsey data by fitting the frequency detuning and T2 decay.
    Only ds_fit and fit_results are stored.
    """
    ds = node.results["ds_raw"]
    # Fit using the experiment‑specific fitting function.
    ds_fit, fit_results = fit_frequency_detuning_and_t2_decay(ds, node.namespace["qubits"], node.parameters)
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}


# %% {Plotting}
@node.run_action(skip_if=node.parameters.simulate)
def data_plotting(node: QualibrationNode[Parameters, QuAM]):
    """Plot the raw and fitted Ramsey data."""
    fig = plot_ramseys_data_with_fit(
        node.results["ds_raw"], node.namespace["qubits"], node.parameters, node.results["ds_fit"]
    )
    plt.tight_layout()
    plt.show()
    node.results["figure"] = fig


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def state_update(node: QualibrationNode[Parameters, QuAM]):
    """
    Update the qubit calibration state with the fitted frequency offset and T2_ramsey.
    For each qubit, subtract the frequency offset from its intermediate frequency and update T2ramsey.
    """
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            # Update the qubit’s virtual frequency and T2_ramsey based on the fit results.
            q.xy.intermediate_frequency -= float(node.results["fit_results"][q.name].freq_offset)
            q.T2ramsey = float(node.results["fit_results"][q.name].decay)


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, QuAM]):
    node.save()

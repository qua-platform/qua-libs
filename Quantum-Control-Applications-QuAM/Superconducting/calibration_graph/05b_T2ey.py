# %% {Imports}
from qm.qua import *
import matplotlib.pyplot as plt
from dataclasses import asdict
import xarray as xr
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qualang_tools.results import progress_counter
from qualang_tools.loops import from_array
from qualang_tools.xarray_data_fetcher import XarrayDataFetcher

from quam_config import QuAM
from qualibrate import QualibrationNode
from qualibrate.utils.logger_m import logger
from quam_experiments.parameters.sweep_parameters import get_idle_times_in_clock_cycles
from quam_experiments.parameters.qubits_experiment import get_qubits
from quam_experiments.workflow import simulate_and_plot

# Import T2 echo–specific helpers
from parameters import Parameters
from fitting import fit_t2e_decay
from plotting import plot_t2e_data_with_fit

# %% {Node_parameters}
description = """
        T2 ECHO MEASUREMENT
The sequence consists in putting the qubit in the excited state with a T2 echo sequence and measuring the resonator
after a varying time. The qubit T2 echo is extracted by fitting the exponential decay of the measured quadratures.

Prerequisites:
    - The resonator frequency must be determined (resonator_spectroscopy).
    - Qubit pulses (x90, x180) must be calibrated.
    - (Optional) Readout calibration should be performed (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Desired flux biases must be set.

State update:
    - The T2 echo for each qubit: qubit.T2e
"""

node = QualibrationNode[Parameters, QuAM](name="05_T2e", description=description, parameters=Parameters())


# Instantiate the QuAM class from the state file
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
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Instantiate unit converter locally.
    u = unit(coerce_to_integer=True)
    # Retrieve active qubits using the helper function.
    node.namespace["qubits"] = get_qubits(node)
    qubits = get_qubits(node)
    num_qubits = len(qubits)
    n_avg = node.parameters.num_averages
    idle_times = get_idle_times_in_clock_cycles(node.parameters)
    # Register the sweep axes (instead of storing idle_times) for dataset creation.
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray([q.name for q in qubits], attrs={"long_name": "qubit"}),
        "idle_time": xr.DataArray(4 * idle_times, attrs={"long_name": "idle time", "units": "ns"}),
    }
    # Determine the flux bias offset.
    flux_point = node.parameters.flux_point_joint_or_independent_or_arbitrary
    if flux_point == "arbitrary":
        arb_flux_bias_offset = {q.name: q.z.arbitrary_offset for q in qubits}
    else:
        arb_flux_bias_offset = {q.name: 0.0 for q in qubits}

    # Build the QUA program.
    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.qua_declaration()
        t = declare(int)
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        # Iterate over qubits using batch().
        for multiplexed_qubits in qubits.batch():
            # Initialize each qubit using initialize_qpu.
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_each_(t, idle_times):
                    # Reset the qubits.
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset_qubit(node.parameters.reset_type, node.parameters.simulate, logger=logger)
                    # T2 echo pulse sequence.
                    for i, qubit in multiplexed_qubits.items():
                        qubit.align()
                        qubit.xy.play("x90")
                        qubit.align()
                        qubit.z.play(
                            "const",
                            amplitude_scale=arb_flux_bias_offset[qubit.name] / qubit.z.operations["const"].amplitude,
                            duration=t,
                        )
                        qubit.align()
                        qubit.xy.play("x180")
                        qubit.align()
                        qubit.z.play(
                            "const",
                            amplitude_scale=arb_flux_bias_offset[qubit.name] / qubit.z.operations["const"].amplitude,
                            duration=t,
                        )
                        qubit.align()
                        qubit.xy.play("x90")
                        qubit.xy.play("x180")
                        qubit.align()
                        # Measurement.
                        if node.parameters.use_state_discrimination:
                            qubit.readout_state(state[i])
                            save(state[i], state_st[i])
                        else:
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
            align()
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(idle_times)).average().save(f"state{i+1}")
                else:
                    I_st[i].buffer(len(idle_times)).average().save(f"I{i+1}")
                    Q_st[i].buffer(len(idle_times)).average().save(f"Q{i+1}")


# %% {Simulate_or_execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, QuAM]):
    """Connect to the QOP and simulate the QUA program"""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, QuAM]):
    """Connect to the QOP, execute the QUA program and fetch the raw data into a xarray dataset."""
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
    """Analyze the raw data and store the fitted data and fit results."""
    node.results["ds_fit"], fit_results = fit_t2e_decay(node.results["ds_raw"], node.parameters)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}


# %% {Plotting}
@node.run_action(skip_if=node.parameters.simulate)
def data_plotting(node: QualibrationNode[Parameters, QuAM]):
    """Plot the raw and fitted T2 echo data."""
    fig = plot_t2e_data_with_fit(
        node.results["ds_raw"], node.namespace["qubits"], node.parameters, node.results["ds_fit"]
    )
    plt.show()
    node.results["figure"] = fig


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def state_update(node: QualibrationNode[Parameters, QuAM]):
    """Update the T2 echo for each qubit in the calibration state, if the fit is successful."""
    with node.record_state_updates():
        for index, q in enumerate(node.namespace["qubits"]):
            if node.results["ds_fit"].sel(qubit=q.name).success:
                q.T2e = float(node.results["ds_fit"].sel(qubit=q.name).tau.values) * 1e-9


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, QuAM]):
    node.save()

"""
        T1 MEASUREMENT
The sequence consists in putting the qubit in the excited stated by playing the x180 pulse and measuring the resonator
after a varying time. The qubit T1 is extracted by fitting the exponential decay of the measured quadratures.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit spectroscopy, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update the qubit T1 in the state.
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_config import QuAM
from quam_experiments.macros import qua_declaration, active_reset, readout_state
from quam_libs.qua_datasets import convert_IQ_to_V
from quam_libs.plot_utils import QubitGrid, grid_iter
from quam_libs.save_utils import fetch_results_as_xarray
from quam_experiments.analysis.fit import fit_decay_exp, decay_exp
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    num_averages: int = 100
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 100000
    wait_time_step_in_ns: int = 600
    flux_point_joint_or_independent_or_arbitrary: Literal["joint", "independent", "arbitrary"] = "independent"
    reset_type: Literal["active", "thermal"] = "thermal"
    use_state_discrimination: bool = False
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = 3
    multiplexed: bool = False


node = QualibrationNode(name="05_T1", parameters=Parameters())
# Instantiate the QuAM class from the state file
node.machine = QuAM.load()
node.results["initial_parameters"] = node.parameters.model_dump()

# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)




# %% {Initialize_QuAM_and_QOP}
# @node.run_action(skip_if=node.parameters.simulate or node.parameters.load_data_id is not None)
# def initialize_the_machine(node: QualibrationNode[Parameters, QuAM]):
#     if node.parameters.load_data_id is None:
#         qmm = node.machine.connect()
#         # node.machine.




# %% {QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, QuAM]):
    if node.parameters.qubits is None or node.parameters.qubits == "":
        qubits = node.machine.active_qubits
    else:
        qubits = [node.machine.qubits[q] for q in node.parameters.qubits]
    num_qubits = len(qubits)

    n_avg = node.parameters.num_averages  # The number of averages
    # Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
    idle_times = np.arange(
        node.parameters.min_wait_time_in_ns // 4,
        node.parameters.max_wait_time_in_ns // 4,
        node.parameters.wait_time_step_in_ns // 4,
    )

    with program() as node.qua_program:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
        t = declare(int)  # QUA variable for the idle time
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        for i, qubit in enumerate(qubits):

            # Bring the active qubits to the desired frequency point
            node.machine.set_all_fluxes(flux_point=node.parameters.flux_point_joint_or_independent_or_arbitrary, target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(t, idle_times)):
                    if node.parameters.reset_type == "active":
                        active_reset(qubit, "readout")
                    else:
                        qubit.resonator.wait(qubit.thermalization_time * u.ns)
                        qubit.align()

                    qubit.xy.play("x180")
                    qubit.align()
                    qubit.wait(t)
                    qubit.align()

                    # Measure the state of the resonators
                    if node.parameters.use_state_discrimination:
                        readout_state(qubit, state[i])
                        save(state[i], state_st[i])
                    else:
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
            # Measure sequentially
            if not node.parameters.multiplexed:
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(idle_times)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(idle_times)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(idle_times)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, QuAM]):
    # Open Communication with the QOP
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns // 4)
    job = qmm.simulate(config, node.qua_program, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": fig}
    node.save()

@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, QuAM]):
    # Open Communication with the QOP
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.job = job = qm.execute(node.qua_program) # TODO: how to pass the job between actions?
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, node.parameters.num_averages, start_time=results.start_time)
        print(job.execution_report())

# %% {Data_fetching_and_dataset_creation}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, QuAM]):
    node = node.load_from_id(node.parameters.load_data_id)
    # Add the dataset to the node
    node.results = {"ds": node.results["ds"]}


@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def fetch_data(node: QualibrationNode[Parameters, QuAM]):
    # TODO: improve this since we call it everywhere...
    if node.parameters.qubits is None or node.parameters.qubits == "":
        qubits = node.machine.active_qubits
    else:
        qubits = [node.machine.qubits[q] for q in node.parameters.qubits]
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    #TODO: avoid recalculate it here
    idle_times = np.arange(
        node.parameters.min_wait_time_in_ns // 4,
        node.parameters.max_wait_time_in_ns // 4,
        node.parameters.wait_time_step_in_ns // 4,
    )
    ds = fetch_results_as_xarray(node.job.result_handles, qubits, {"idle_time": idle_times})
    # Convert IQ data into volts
    ds = convert_IQ_to_V(ds, qubits)
    # Convert time into µs
    ds = ds.assign_coords(idle_time=4 * ds.idle_time / u.us)  # convert to µs
    ds.idle_time.attrs = {"long_name": "idle time", "units": "µs"}
    # Add the dataset to the node
    node.results = {"ds": ds}

# %% {Data_analysis}
@node.run_action(skip_if=node.parameters.simulate)
def data_analysis(node: QualibrationNode[Parameters, QuAM]):
    ds = node.results["ds"]
    # Fit the exponential decay
    if node.parameters.use_state_discrimination:
        fit_data = fit_decay_exp(ds.state, "idle_time")
    else:
        fit_data = fit_decay_exp(ds.I, "idle_time")
    fit_data.attrs = {"long_name": "time", "units": "µs"}
    node.results["fitted_data"] = fit_data.to_dataset(name="fitted_data")
    # Fitted decay
    node.results["fitted"] = decay_exp(
        ds.idle_time,
        node.results["fitted_data"].sel(fit_vals="a"),
        node.results["fitted_data"].sel(fit_vals="offset"),
        node.results["fitted_data"].sel(fit_vals="decay"),
    )
    # T1
    tau = -1 / node.results["fitted_data"].sel(fit_vals="decay")
    node.results["fitted_data"] = node.results["fitted_data"].assign_coords(tau=("qubit", tau.fitted_data.data))
    node.results["fitted_data"].tau.attrs = {"long_name": "T1", "units": "µs"}
    # and its uncertainty
    decay = node.results["fitted_data"].sel(fit_vals="decay")
    decay_res = node.results["fitted_data"].sel(fit_vals="decay_decay")
    tau_error = -tau * (np.sqrt(decay_res) / decay)
    node.results["fitted_data"] = node.results["fitted_data"].assign_coords(tau_error=("qubit", tau_error.fitted_data.data))
    node.results["fitted_data"].tau_error.attrs = {"long_name": "T1 error", "units": "µs"}
    # Assess whether the fit was successful or not
    success_criteria = (tau.fitted_data.values > 0) & (tau_error.fitted_data.values / tau.fitted_data.values < 1)
    node.results["fitted_data"] = node.results["fitted_data"].assign_coords(success=("qubit", success_criteria))

# %% {Plotting}
@node.run_action(skip_if=node.parameters.simulate)
def data_analysis(node: QualibrationNode[Parameters, QuAM]):
    ds = node.results["ds"]
    # TODO: improve this since we call it everywhere...
    if node.parameters.qubits is None or node.parameters.qubits == "":
        qubits = node.machine.active_qubits
    else:
        qubits = [node.machine.qubits[q] for q in node.parameters.qubits]

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
            ax.set_ylabel("State")
        else:
            ds.sel(qubit=qubit["qubit"]).I.plot(ax=ax)
            ax.set_ylabel("I (V)")
        ax.plot(ds.idle_time, node.results["fitted"].fitted_data.loc[qubit], "r--")
        ax.set_title(qubit["qubit"])
        ax.set_xlabel("Idle_time (uS)")
        ax.text(
            0.1,
            0.9,
            f'T1 = {node.results["fitted_data"].sel(qubit=qubit["qubit"]).tau.values:.1f} ± {node.results["fitted_data"].sel(qubit=qubit["qubit"]).tau_error.values:.1f} µs',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )
    grid.fig.suptitle("T1")
    plt.tight_layout()
    plt.show()
    node.results["figure_raw"] = grid.fig

# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def state_update(node: QualibrationNode[Parameters, QuAM]):
    # TODO: improve this since we call it everywhere...
    if node.parameters.qubits is None or node.parameters.qubits == "":
        qubits = node.machine.active_qubits
    else:
        qubits = [node.machine.qubits[q] for q in node.parameters.qubits]
    with node.record_state_updates():
        for index, q in enumerate(qubits):
            if node.results["fitted_data"].sel(qubit=q.name).success:
                q.T1 = float(node.results["fitted_data"].sel(qubit=q.name).tau.values) * 1e-6

# %% {Save_results}
@node.run_action(skip_if=node.parameters.simulate)
def save_results(node: QualibrationNode[Parameters, QuAM]):
    node.save()

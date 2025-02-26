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
from qualibrate import QualibrationNode
from quam_config import QuAM
from quam_experiments.macros import qua_declaration, readout_state, reset_qubit
from quam_libs.plot_utils import QubitGrid, grid_iter
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm.qua import *
import matplotlib.pyplot as plt
import numpy as np

from quam_experiments.parameters.sweep_parameters import get_idle_times_in_clock_cycles
from quam_experiments.parameters.qubits_experiment import get_qubits_used_in_node
from quam_experiments.workflow.simulation import simulate_and_plot
from quam_experiments.experiments.T1.parameters import Parameters
from quam_experiments.experiments.T1.fetch_dataset import fetch_dataset
from quam_experiments.experiments.T1.fitting import fit_t1_decay
from quam_experiments.experiments.T1.plotting import plot_t1s_data_with_fit

# %% {Node_parameters}
node = QualibrationNode(
    name="05_T1",
    parameters=Parameters(
        num_averages=100,
        min_wait_time_in_ns=16,
        max_wait_time_in_ns=100000,
        wait_time_num_points=151,
        log_or_linear_sweep="linear",
        use_state_discrimination=False,
        qubits=None,
        multiplexed=False,
        flux_point_joint_or_independent="joint",
        timeout=120,
        load_data_id=None,
        simulate=False,
        simulation_duration_ns=25000,
        use_waveform_report=True,
    ),
)

# Instantiate the QuAM class from the state file
node.machine = QuAM.load()
node.results["initial_parameters"] = node.parameters.model_dump()

# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)


# %% {QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, QuAM]):
    node.qubits = qubits = get_qubits_used_in_node(node.machine, node.parameters)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_averages  # The number of averages
    idle_times = get_idle_times_in_clock_cycles(node.parameters)

    with program() as node.qua_program:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
        t = declare(int)  # QUA variable for the idle time
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for i, qubit in enumerate(qubits):
            # Bring the active qubits to the desired frequency point
            # TODO: need to get rid of this for fixed frequency transmons
            node.machine.set_all_fluxes(flux_point=node.parameters.flux_point_joint_or_independent, target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_each_(t, idle_times):
                    # reset_qubit(qubit, node.parameters)
                    qubit.align()
                    qubit.xy.play("x180")
                    qubit.align()
                    qubit.resonator.wait(t)
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
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.qua_program, node.parameters)
    # todo: we can't serialize the simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report}
    node.save()


@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, QuAM]):
    # Open Communication with the QOP
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.job = job = qm.execute(node.qua_program)  # TODO: how to pass the job between actions?
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
    # TODO: temp fix
    load_data_id = node.parameters.load_data_id
    node = node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Add the dataset to the node
    node.results = {"ds": node.results["ds"]}


@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def fetch_data(node: QualibrationNode[Parameters, QuAM]):
    ds = fetch_dataset(node.job, node.qubits, node.parameters)
    node.results = {"ds": ds}


# %% {Data_analysis}
@node.run_action(skip_if=node.parameters.simulate)
def data_analysis(node: QualibrationNode[Parameters, QuAM]):
    # todo check the units with real data
    node.results["fit_results"] = fit_t1_decay(node.results["ds"], node.parameters)


# %% {Plotting}
@node.run_action(skip_if=node.parameters.simulate)
def data_plotting(node: QualibrationNode[Parameters, QuAM]):
    qubits = get_qubits_used_in_node(node.machine, node.parameters)
    fig = plot_t1s_data_with_fit(node.results["ds"], qubits, node.parameters, node.results["fit_results"])
    node.results["figure"] = fig

    plt.tight_layout()
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def state_update(node: QualibrationNode[Parameters, QuAM]):
    qubits = get_qubits_used_in_node(node.machine, node.parameters)
    with node.record_state_updates():
        for index, q in enumerate(qubits):
            if node.results["fit_results"].sel(qubit=q.name).success:
                q.T1 = float(node.results["fit_results"].sel(qubit=q.name).tau.values) * 1e-9


# %% {Save_results}
@node.run_action(skip_if=node.parameters.simulate)
def save_results(node: QualibrationNode[Parameters, QuAM]):
    node.save()

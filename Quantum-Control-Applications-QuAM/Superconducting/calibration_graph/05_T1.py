# %% {Imports}
from qm.qua import *
import matplotlib.pyplot as plt
from dataclasses import asdict
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from quam_config import QuAM
from quam_experiments.parameters.sweep_parameters import get_idle_times_in_clock_cycles
from quam_experiments.parameters.qubits_experiment import get_qubits
from quam_experiments.macros import qua_declaration, readout_state, reset_qubit
from quam_experiments.workflow import simulate_and_plot, fetch_dataset, print_progress_bar
from quam_experiments.experiments.T1 import Parameters, fit_t1_decay, log_t1, plot_t1s_data_with_fit


# %% {Node_parameters}
description = """
        T1 MEASUREMENT
The sequence consists in putting the qubit in the excited stated by playing the x180 pulse and measuring the resonator
after a varying time. The qubit T1 is extracted by fitting the exponential decay of the measured quadratures.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit spectroscopy, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the desired flux biases if relevant.

State update:
    - The T1 relaxation time for each qubit: qubit.T1
"""

# Be sure to include [Parameters, QuAM] so the node has proper type hinting
node = QualibrationNode[Parameters, QuAM](
    name="05_T1",  # Name should be unique
    description=description,
    parameters=Parameters(
    ),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
if node.modes.interactive:
    # node.parameters.some_parameter = 123
    pass

# Instantiate the QuAM class from the state file
node.machine = QuAM.load()


# %% {QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, QuAM]):
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)

    node.namespace["qubits"] = get_qubits(node)
    num_qubits = len(node.namespace["qubits"])

    n_avg = node.parameters.num_averages  # The number of averages
    idle_times = get_idle_times_in_clock_cycles(node.parameters)

    # Register the sweep axes to be added to the dataset when fetching data
    #todo: set as a DataArray instead
    node.namespace["sweep_axes"] = {
        "idle_time": {
            "data": 4 * idle_times,
            "attrs": {"long_name": "idle time", "units": "ns"},
        },
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
        t = declare(int)  # QUA variable for the idle time
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for i, qubit in enumerate(node.namespace["qubits"]):
            # Bring the active qubits to the desired frequency point
            # TODO: need to get rid of this for fixed frequency transmons
            node.machine.set_all_fluxes(flux_point=node.parameters.flux_point_joint_or_independent, target=qubit)
            # todo: remove the flux points from the parameters and use initialize
            # node.machine.initialize_qubit(target=qubit)
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_each_(t, idle_times):
                    reset_qubit(qubit, node.parameters)
                    # todo: qubit.reset(node.parameters)
                    # qubit.reset_thermal()
                    # qubit.reset_active()
                    # qubit.reset_active_gef()
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
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])  # TODO: how to pass the job between actions?
        print_progress_bar(job, iteration_variable="n", total_number_of_iterations=node.parameters.num_averages)
        print(job.execution_report())


# %% {Data_loading_and_dataset_creation}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, QuAM]):
    # TODO: temp fix
    load_data_id = node.parameters.load_data_id
    node = node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Add the dataset to the node
    node.results = {"ds": node.results["ds"]}


# %% {Data_fetching_and_dataset_creation}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def fetch_data(node: QualibrationNode[Parameters, QuAM]):
    ds = fetch_dataset(node.namespace["job"], node.namespace["qubits"], node.parameters, node.namespace["sweep_axes"])
    node.results = {"ds": ds}


# %% {Data_analysis}
@node.run_action(skip_if=node.parameters.simulate)
def data_analysis(node: QualibrationNode[Parameters, QuAM]):
    # todo check the units with real data
    node.results["fit_data"], fit_results = fit_t1_decay(node.results["ds"], node.parameters)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    # todo: How to get the looger to print on the console?
    from qualibrate.utils.logger_m import logger

    log_t1(node.results["fit_data"], logger)


# %% {Plotting}
@node.run_action(skip_if=node.parameters.simulate)
def data_plotting(node: QualibrationNode[Parameters, QuAM]):
    fig = plot_t1s_data_with_fit(node.results["ds"], node.namespace["qubits"], node.parameters, node.results["fit_data"])
    node.results["figure"] = fig
    plt.tight_layout()
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def state_update(node: QualibrationNode[Parameters, QuAM]):
    with node.record_state_updates():
        for index, q in enumerate(node.namespace["qubits"]):
            if node.results["fit_data"].sel(qubit=q.name).success:
                q.T1 = float(node.results["fit_data"].sel(qubit=q.name).tau.values) * 1e-9


# %% {Save_results}
@node.run_action(skip_if=node.parameters.simulate)
def save_results(node: QualibrationNode[Parameters, QuAM]):
    node.save()

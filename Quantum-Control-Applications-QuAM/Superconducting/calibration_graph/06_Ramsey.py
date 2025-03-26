# %% {Imports}
from dataclasses import asdict
import matplotlib.pyplot as plt

from qm.qua import *

from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import QuAM
from quam_experiments.experiments.ramsey.analysis.fetch_dataset import fetch_dataset
from quam_experiments.experiments.ramsey.analysis.fitting import (
    fit_frequency_detuning_and_t2_decay,
)
from quam_experiments.experiments.ramsey.parameters import (
    Parameters,
    get_idle_times_in_clock_cycles,
)
from quam_experiments.experiments.ramsey.plotting import plot_ramseys_data_with_fit
from quam_experiments.parameters.qubits_experiment import get_qubits
from quam_experiments.workflow.simulation import simulate_and_plot


description = """
        RAMSEY WITH VIRTUAL Z ROTATIONS
The program consists in playing a Ramsey sequence (x90 - idle_time - x90 - measurement) for different idle times.
Instead of detuning the qubit gates, the frame of the second x90 pulse is rotated (de-phased) to mimic an accumulated
phase acquired for a given detuning after the idle time.
This method has the advantage of playing gates on resonance as opposed to the detuned Ramsey.

From the results, one can fit the Ramsey oscillations and precisely measure the qubit resonance frequency and T2*.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit spectroscopy, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Update the qubits frequency and T2_ramsey in the state.
    - Save the current state
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


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, QuAM]):
    # You can get type hinting in your IDE by typing node.parameters.
    pass


# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)

node.machine = QuAM.load()

node.namespace["qubits"] = qubits = get_qubits(node)
num_qubits = len(qubits)

config = node.machine.generate_config()
if node.parameters.load_data_id is None:
    qmm = node.machine.connect()

# %% {QUA_program}
n_avg = node.parameters.num_averages
idle_times = get_idle_times_in_clock_cycles(node.parameters)
detuning = node.parameters.frequency_detuning_in_mhz * u.MHz
flux_point = node.parameters.flux_point_joint_or_independent

detuning_signs = [-1, 1]

with program() as ramsey:
    I, I_st, Q, Q_st, n, n_st = node.machine.qua_declaration()
    idle_time = declare(int)
    detuning_sign = declare(int)
    virtual_detuning_phases = [declare(fixed) for _ in range(num_qubits)]

    if node.parameters.use_state_discrimination:
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]

    for multiplexed_qubits in qubits.batch():
        # todo: is this the right behaviour?
        for qubit in multiplexed_qubits.values():
            node.machine.set_all_fluxes(flux_point, target=qubit)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            with for_each_(idle_time, idle_times):
                with for_(*from_array(detuning_sign, detuning_signs)):
                    for i, qubit in multiplexed_qubits.items():
                        with if_(detuning_sign == 1):
                            assign(
                                virtual_detuning_phases[i],
                                Cast.mul_fixed_by_int(detuning * 1e-9, 4 * idle_time),
                            )
                        with else_():
                            assign(
                                virtual_detuning_phases[i],
                                Cast.mul_fixed_by_int(-detuning * 1e-9, 4 * idle_time),
                            )

                    align()
                    # with strict_timing_():
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.play("x90")
                        qubit.xy.frame_rotation_2pi(virtual_detuning_phases[i])
                        qubit.xy.wait(idle_time)
                        qubit.xy.play("x90")

                    align()
                    for i, qubit in multiplexed_qubits.items():
                        if node.parameters.use_state_discrimination:
                            qubit.readout_state(state[i])
                            save(state[i], state_st[i])
                        else:
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.wait(qubit.thermalization_time * u.ns)
                        reset_frame(qubit.xy.name)

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            if node.parameters.use_state_discrimination:
                state_st[i].buffer(len(detuning_signs)).buffer(
                    len(idle_times)
                ).average().save(f"state{i + 1}")
            else:
                I_st[i].buffer(len(detuning_signs)).buffer(
                    len(idle_times)
                ).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(detuning_signs)).buffer(
                    len(idle_times)
                ).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    samples, fig = simulate_and_plot(qmm, config, ramsey, node.parameters)
    node.results = {"figure": fig}
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(ramsey)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)


# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds = fetch_dataset(job, qubits, node.parameters)
        node.results = {"ds": ds}
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]

    # %% {Data_analysis}
    fits = fit_frequency_detuning_and_t2_decay(ds, qubits, node.parameters)
    node.results["fit_results"] = {k: asdict(v) for k, v in fits.items()}

    for fit in fits.values():
        fit.log_frequency_offset()
        fit.log_t2()
    node.outcomes = {q.name: "successful" for q in node.namespace["qubits"]}

    # %% {Plotting}
    fig = plot_ramseys_data_with_fit(ds, qubits, node.parameters, fits)
    node.results["figure"] = fig

    plt.tight_layout()
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def state_update(node: QualibrationNode[Parameters, QuAM]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            fit_results = node.results["fit_results"][q.name]
            q.xy.intermediate_frequency -= float(fit_results["freq_offset"])
            q.T2ramsey = float(fit_results["decay"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, QuAM]):
    node.save()

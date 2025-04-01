# %% {Imports}
import logging
from typing import Optional, List
import numpy as np
from tqdm import tqdm

from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit

from qm import Program, generate_qua_script
from qm.qua import *

from qualibrate import QualibrationNode
from quam_experiments.experiments.readout_optimization_3d.analysis.calculate_readout_fidelity import (
    calculate_readout_fidelity,
    get_maximum_fidelity_per_qubit,
)
from quam_experiments.experiments.readout_optimization_3d.analysis.combine_batches import (
    combine_batches,
)
from quam_experiments.experiments.readout_optimization_3d.analysis.fetch_dataset import (
    fetch_dataset,
)
from quam_experiments.experiments.readout_optimization_3d.analysis.filtering import (
    filter_readout_fidelity,
)
from quam_experiments.experiments.readout_optimization_3d.analysis.plotting import (
    plot_fidelity_3d,
    plot_fidelity_2d,
)
from quam_experiments.experiments.readout_optimization_3d.make_qua_streams_per_qubit import (
    make_qua_streams_per_qubit,
)
from quam_experiments.experiments.readout_optimization_3d.make_qua_variables_per_qubit import (
    make_qua_variables_per_qubit,
)
from quam_experiments.experiments.readout_optimization_3d.measurement_batching import (
    generate_measurement_batches,
    get_max_accumulated_readouts,
)
from quam_experiments.parameters.qubits_experiment import get_qubits
from qualibration_libs.trackable_object import tracked_updates
from quam_config import QuAM
from quam_experiments.experiments.readout_optimization_3d.parameters import (
    Parameters,
    get_durations,
)
from quam_experiments.experiments.readout_optimization_3d.parameters import (
    get_frequency_detunings_in_hz,
    get_amplitude_factors,
)
from quam_experiments.workflow.simulation import simulate_and_plot
from quam_experiments.parameters.qubits_experiment import get_qubits


# %% {Initialisation}
description = """
        READOUT OPTIMISATION: FREQUENCY, POWER, DURATION
This sequence involves measuring the state of the resonator in two scenarios: first,
after thermalization (with the qubit in the |g> state) and then after applying a pi
pulse (with the qubit in the |e> state). This is done while varying the readout pulse
frequency, power and duration.

The average I & Q quadratures for the qubit states |g> and |e>, along with their
variances, are extracted to determine the Signal-to-Noise Ratio (SNR). The readout
parameters that yield the highest SNR is selected as the optimal choice.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under
      study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit spectroscopy, power_rabi
      and updated the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the readout frequency and dispersive shift chi in the state.
    - Save the current state
"""

node = QualibrationNode[Parameters, QuAM](
    name="08c_readout_frequency_duration_power_optimization",
    description=description,
    parameters=Parameters(
        qubits=None,
        multiplexed=True,
        flux_point_joint_or_independent="joint",
        num_runs=20,
        frequency_span_in_mhz=5,
        frequency_step_in_mhz=0.2,
        min_amplitude_factor=0.5,
        max_amplitude_factor=1.99,
        num_amplitudes=15,
        max_duration_in_ns=1040,
        num_durations=4,
        load_data_id=None,
        plotting_dimension="2D",
        fidelity_smoothing_intensity=0.5,
        max_readout_amplitude=0.125,
        # simulate=True,
        # simulation_duration_ns=10000,
        # use_waveform_report=True
    ),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, QuAM]):
    # You can get type hinting in your IDE by typing node.parameters.
    pass


node.machine = QuAM.load()


# %% {Create_QUA_program}
u = unit(coerce_to_integer=True)

if node.parameters.load_data_id is None:
    qmm = node.machine.connect()

# Get the active qubits from the node and organize them by batches
node.namespace["qubits"] = qubits = get_qubits(node)
num_qubits = len(qubits)

readout_pulse_name = "readout"

# temporarily set readout length to be maximum duration in sweep
tracked_resonators = []
for resonator in [qubit.resonator for qubit in qubits]:
    with tracked_updates(resonator, auto_revert=False, dont_assign_to_none=True) as tracked_resonator:
        tracked_resonator.operations[readout_pulse_name].length = node.parameters.max_duration_in_ns
        tracked_resonators.append(tracked_resonator)

config = node.machine.generate_config()

n_avg = node.parameters.num_runs

dfs = get_frequency_detunings_in_hz(node.parameters)
amps = get_amplitude_factors(node.parameters)
durations = get_durations(node.parameters)

flux_point = node.parameters.flux_point_joint_or_independent


def readout_optimization_3d_measured_in_batches(n_avg: int, measurement_batch: Optional[List[str]] = None) -> Program:
    """
    Returns the 3D readout optimization program, but only measures those
    qubits which appears in the `measurement_batch` list to avoid exceeding
    resource allocation limits.
    """
    with program() as readout_optimization_3d:
        n = declare(int)
        df = declare(int)
        a = declare(fixed)

        # Create QUA variables and streams only if the qubit is in the measurement batch
        II_g, IQ_g, QI_g, QQ_g, I_g, Q_g, II_e, IQ_e, QI_e, QQ_e, I_e, Q_e = make_qua_variables_per_qubit(
            measurement_batch, node.parameters
        )
        I_g_st, Q_g_st, I_e_st, Q_e_st = make_qua_streams_per_qubit(measurement_batch)
        n_st = declare_stream()

        for multiplexed_qubits in qubits.batch():
            node.machine.set_all_fluxes(flux_point=flux_point, target=list(multiplexed_qubits.values())[0])

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(df, dfs)):
                    with for_(*from_array(a, amps)):
                        for qubit in multiplexed_qubits.values():
                            update_frequency(
                                qubit.resonator.name,
                                qubit.resonator.intermediate_frequency + df,
                            )

                        if not node.parameters.simulate:
                            wait(node.machine.thermalization_time * u.ns)

                        align()
                        for qubit in multiplexed_qubits.values():
                            if qubit in measurement_batch:
                                i = measurement_batch.index(qubit)
                                qubit.resonator.measure_accumulated(
                                    pulse_name=readout_pulse_name,
                                    qua_vars=(II_g[i], IQ_g[i], QI_g[i], QQ_g[i]),
                                    num_segments=node.parameters.num_durations,
                                    amplitude_scale=a,
                                )
                            else:
                                # play, but don't demodulate
                                qubit.resonator.play(pulse_name=readout_pulse_name, amplitude_scale=a)

                        align()
                        if not node.parameters.simulate:
                            wait(node.machine.thermalization_time * u.ns)

                        for qubit in multiplexed_qubits.values():
                            qubit.xy.play("x180")

                        align()
                        for qubit in multiplexed_qubits.values():
                            if qubit in measurement_batch:
                                i = measurement_batch.index(qubit)
                                qubit.resonator.measure_accumulated(
                                    pulse_name=readout_pulse_name,
                                    qua_vars=(II_e[i], IQ_e[i], QI_e[i], QQ_e[i]),
                                    num_segments=node.parameters.num_durations,
                                    amplitude_scale=a,
                                )
                            else:
                                # play, but don't demodulate
                                qubit.resonator.play(pulse_name=readout_pulse_name, amplitude_scale=a)

                        for i, qubit in enumerate(measurement_batch):
                            j = declare(int)
                            with for_(j, 0, j < node.parameters.num_durations, j + 1):
                                assign(I_g[i][j], II_g[i][j] + IQ_g[i][j])
                                save(I_g[i][j], I_g_st[i])
                                assign(Q_g[i][j], QI_g[i][j] + QQ_g[i][j])
                                save(Q_g[i][j], Q_g_st[i])
                                assign(I_e[i][j], II_e[i][j] + IQ_e[i][j])
                                save(I_e[i][j], I_e_st[i])
                                assign(Q_e[i][j], QI_e[i][j] + QQ_e[i][j])
                                save(Q_e[i][j], Q_e_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(len(measurement_batch)):
                streams = {"I_g": I_g_st, "Q_g": Q_g_st, "I_e": I_e_st, "Q_e": Q_e_st}
                for name, stream in streams.items():
                    stream[i].buffer(node.parameters.num_durations).buffer(len(amps)).buffer(len(dfs)).buffer(
                        n_avg
                    ).save(f"{name}{i + 1}")

        return readout_optimization_3d


max_accumulated_readouts = get_max_accumulated_readouts(qubits, node.parameters)
measurement_batches = generate_measurement_batches(qubits, max_accumulated_readouts)

# This number represents how many times a qubit is present in any batch
qubit_representation = len(measurement_batches) * max_accumulated_readouts / len(qubits)

if len(measurement_batches) > 1:
    logging.info(
        f"Number of qubits measured simultaneously exceeds the resource limit of {max_accumulated_readouts}. "
        f"Splitting into {len(measurement_batches)} batches."
    )

if n_avg % len(measurement_batches) != 0:
    raise ValueError(
        f"Expected the number of averages {n_avg} to be a multiple of {qubit_representation} "
        f"in order to be measured {n_avg} times over {len(measurement_batches)} batches."
    )

n_avg = n_avg // qubit_representation

programs = []
for measurement_batch in measurement_batches:
    programs.append(readout_optimization_3d_measured_in_batches(n_avg, measurement_batch))

with open("debug.py", "w+") as f:
    f.write(generate_qua_script(programs[0], config))

# %% {Simulate}
if node.parameters.simulate:
    samples, fig = simulate_and_plot(qmm, config, programs[0], node.parameters)
    node.results = {"figure": fig}
    node.save()

# %% {Execute}
elif node.parameters.load_data_id is None:
    datasets = []
    for i, program in enumerate(tqdm(programs, unit="measurement batch")):
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(program)
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                n = results.fetch_all()[0]
                progress_counter(n, n_avg, start_time=results.start_time)

        run_axis = np.arange(i * n_avg, (i + 1) * n_avg)
        datasets.append(fetch_dataset(job, measurement_batches[i], run_axis, node.parameters))

    ds = combine_batches(datasets)


# %% {Analyse_data}
if not node.parameters.simulate:
    if node.parameters.load_data_id is not None:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        node.parameters.load_data_id = load_data_id
        ds = node.results["ds"]

    node.results = {"ds": ds}

    ds["raw_fidelity"] = calculate_readout_fidelity(ds)
    ds["fidelity"] = filter_readout_fidelity(ds, node.parameters)
    node.results["optimal_ds"] = optimal_ds = get_maximum_fidelity_per_qubit(ds)

    if node.parameters.plotting_dimension == "3D":
        # doesn't save the figure
        fig = plot_fidelity_3d(ds, optimal_ds)
        fig.show()

    elif node.parameters.plotting_dimension == "2D":
        figs = plot_fidelity_2d(ds, optimal_ds)
        for i, q in enumerate(ds.qubit):
            node.results[f"fig_{q.item()}"] = figs[i]

    # undo temporary readout length
    for tracked_resonator in tracked_resonators:
        tracked_resonator.revert_changes()
    node.outcomes = {q.name: "successful" for q in node.namespace["qubits"]}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, QuAM]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        optimal_output_powers = {}
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            # TODO Remove `readout_pulse_name` from the arguments
            optimal_qubit_ds = node.results["optimal_ds"].sel(qubit=q.name)
            q.resonator.intermediate_frequency += int(optimal_qubit_ds.freq.data)
            operation = q.resonator.operations[readout_pulse_name]
            operation.length = int(optimal_qubit_ds.duration.data)
            operation.amplitude *= float(optimal_qubit_ds.amp.data)
            optimal_output_powers[q] = q.resonator.get_output_power(readout_pulse_name)

        # If the amplitude increased above the maximum readout amplitude, increase the power
        # and reduce the amplitude below its limit to protect against saturating the output channel.
        #  However, since different resonators need different powers, but can share the same port,
        # take the safe approach and define a *lower* bound for the power of the port by
        # sorting in descending order of power requirements.
        lowest_possible_full_scale_power_dbm = None
        for qubit, power in sorted(optimal_output_powers.items(), key=lambda item: item[1], reverse=True):
            power_settings = qubit.resonator.set_output_power(
                power_in_dbm=power,
                full_scale_power_dbm=lowest_possible_full_scale_power_dbm,
                max_amplitude=node.parameters.max_readout_amplitude,
                operation=readout_pulse_name,
            )

            if lowest_possible_full_scale_power_dbm is None:
                lowest_possible_full_scale_power_dbm = power_settings["full_scale_power_dbm"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, QuAM]):
    node.save()

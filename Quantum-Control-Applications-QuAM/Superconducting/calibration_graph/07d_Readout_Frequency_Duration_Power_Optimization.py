"""
        READOUT OPTIMISATION: FREQUENCY, POWER, DURATION
This sequence involves measuring the state of the resonator in two scenarios: first, after thermalization
(with the qubit in the |g> state) and then after applying a pi pulse to the qubit (transitioning the qubit to the
|e> state). This is done while varying the readout pulse frequency, power and duration.

The average I & Q quadratures for the qubit states |g> and |e>, along with their variances, are extracted to
determine the Signal-to-Noise Ratio (SNR). The readout parameters that yield the highest fidelity is selected
as the optimal choice.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit spectroscopy, power_rabi and updated the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the readout frequency and dispersive shift chi in the state.
    - Save the current state
"""

# %% {Imports}
import matplotlib.pyplot as plt

from qualibrate import QualibrationNode

from quam_libs.experiments.readout_optimization_3d.analysis.calculate_readout_fidelity import \
    calculate_readout_fidelity, get_maximum_fidelity_per_qubit
from quam_libs.experiments.readout_optimization_3d.analysis.fetch_dataset import fetch_dataset
from quam_libs.experiments.readout_optimization_3d.analysis.filtering import filter_readout_fidelity
from quam_libs.experiments.readout_optimization_3d.analysis.plotting import plot_fidelity_3d, plot_fidelity_2d
from quam_libs.lib.instrument_limits import instrument_limits
from quam_libs.trackable_object import tracked_updates
from quam_libs.components import QuAM
from quam_libs.experiments.readout_optimization_3d.parameters import Parameters, get_durations
from quam_libs.experiments.readout_optimization_3d.parameters import (
    get_frequency_detunings_in_hz,
    get_amplitude_factors
)
from quam_libs.experiments.simulation import simulate_and_plot

from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit

from qm.qua import *


# %% {Node_parameters}
node = QualibrationNode(
    name="07d_Readout_Frequency_Duration_Power_Optimization",
    parameters=Parameters(
        qubits=["q1", "q2"],
        multiplexed=True,
        flux_point_joint_or_independent="joint",
        num_runs=1000,
        frequency_span_in_mhz=5,
        frequency_step_in_mhz=0.2,
        min_amplitude_factor=0.5,
        max_amplitude_factor=1.99,
        num_amplitudes=15,
        max_duration_in_ns=1040,
        num_durations=4,
        load_data_id=1100,
        plotting_dimension="2D",
        fidelity_smoothing_intensity=0.5,
        max_readout_amplitude=0.125
        # simulate=True,
        # simulation_duration_ns=1000,
        # use_waveform_report=True
    )
)

# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)

machine = QuAM.load()

if node.parameters.load_data_id is None:
    qmm = machine.connect()

qubits = machine.get_qubits_used_in_node(node.parameters)
num_qubits = len(qubits)

readout_pulse_name = "readout"

# temporarily set readout length to be maximum duration in sweep
tracked_resonators = []
for resonator in [qubit.resonator for qubit in qubits]:
    with tracked_updates(resonator, auto_revert=False, dont_assign_to_none=True) as tracked_resonator:
        tracked_resonator.operations[readout_pulse_name].length = node.parameters.max_duration_in_ns
        tracked_resonators.append(tracked_resonator)

config = machine.generate_config()

# %% {QUA_program}
n_avg = node.parameters.num_runs

dfs = get_frequency_detunings_in_hz(node.parameters)
amps = get_amplitude_factors(node.parameters)
durations = get_durations(node.parameters)

flux_point = node.parameters.flux_point_joint_or_independent

with program() as readout_optimization_3d:
    n = declare(int)
    df = declare(int)
    a = declare(fixed)

    II_g = [declare(fixed, size=node.parameters.num_durations) for _ in range(num_qubits)]
    IQ_g = [declare(fixed, size=node.parameters.num_durations) for _ in range(num_qubits)]
    QI_g = [declare(fixed, size=node.parameters.num_durations) for _ in range(num_qubits)]
    QQ_g = [declare(fixed, size=node.parameters.num_durations) for _ in range(num_qubits)]
    I_g = [declare(fixed, size=node.parameters.num_durations) for _ in range(num_qubits)]
    Q_g = [declare(fixed, size=node.parameters.num_durations) for _ in range(num_qubits)]

    II_e = [declare(fixed, size=node.parameters.num_durations) for _ in range(num_qubits)]
    IQ_e = [declare(fixed, size=node.parameters.num_durations) for _ in range(num_qubits)]
    QI_e = [declare(fixed, size=node.parameters.num_durations) for _ in range(num_qubits)]
    QQ_e = [declare(fixed, size=node.parameters.num_durations) for _ in range(num_qubits)]
    I_e = [declare(fixed, size=node.parameters.num_durations) for _ in range(num_qubits)]
    Q_e = [declare(fixed, size=node.parameters.num_durations) for _ in range(num_qubits)]

    I_g_st = [declare_stream() for _ in range(num_qubits)]
    Q_g_st = [declare_stream() for _ in range(num_qubits)]
    I_e_st = [declare_stream() for _ in range(num_qubits)]
    Q_e_st = [declare_stream() for _ in range(num_qubits)]
    n_st = declare_stream()

    for multiplexed_qubits in qubits.batch():
        machine.set_all_fluxes(flux_point=flux_point, target=list(multiplexed_qubits.values())[0])
        
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            with for_(*from_array(df, dfs)):
                with for_(*from_array(a, amps)):
                    for i, qubit in multiplexed_qubits.items():
                        update_frequency(
                            qubit.resonator.name,
                            qubit.resonator.intermediate_frequency + df
                        )

                    wait(machine.thermalization_time * u.ns)

                    align()
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure_accumulated(
                            pulse_name=readout_pulse_name,
                            qua_vars=(II_g[i], IQ_g[i], QI_g[i], QQ_g[i]),
                            num_segments=node.parameters.num_durations,
                            amplitude_scale=a,
                        )

                    align()
                    wait(machine.thermalization_time * u.ns)

                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.play("x180")

                    align()
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure_accumulated(
                            pulse_name=readout_pulse_name,
                            qua_vars=(II_e[i], IQ_e[i], QI_e[i], QQ_e[i]),
                            num_segments=node.parameters.num_durations,
                            amplitude_scale=a,
                        )

                    for i, qubit in multiplexed_qubits.items():
                        j = declare(int)
                        with for_(j, 0, j < node.parameters.num_durations, j+1):
                            assign(I_g[i][j], II_g[i][j] + IQ_g[i][j])
                            assign(Q_g[i][j], QI_g[i][j] + QQ_g[i][j])
                            assign(I_e[i][j], II_e[i][j] + IQ_e[i][j])
                            assign(Q_e[i][j], QI_e[i][j] + QQ_e[i][j])
                            save(I_g[i][j], I_g_st[i])
                            save(Q_g[i][j], Q_g_st[i])
                            save(I_e[i][j], I_e_st[i])
                            save(Q_e[i][j], Q_e_st[i])

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            streams = {"I_g": I_g_st, "Q_g": Q_g_st, "I_e": I_e_st, "Q_e": Q_e_st}
            for name, stream in streams.items():
                stream[i]\
                    .buffer(node.parameters.num_durations) \
                    .buffer(len(amps)) \
                    .buffer(len(dfs)) \
                    .buffer(node.parameters.num_runs) \
                    .save(f"{name}{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    samples, fig = simulate_and_plot(qmm, config, readout_optimization_3d, node.parameters)
    node.results = {"figure": fig}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(readout_optimization_3d)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds = fetch_dataset(job, qubits, node.parameters)
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        node.parameters.load_data_id = load_data_id
        ds = node.results["ds"]

    node.results = {"ds": ds}

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

    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            optimal_output_powers = {}
            for q in qubits:
                optimal_ds_for_this_qubit = optimal_ds.sel(qubit=q.name)
                q.resonator.intermediate_frequency += int(optimal_ds_for_this_qubit.freq.data)
                q.resonator.operations[readout_pulse_name].length = int(optimal_ds_for_this_qubit.duration.data)
                q.resonator.operations[readout_pulse_name].amplitude *= float(optimal_ds_for_this_qubit.amp.data)
                optimal_output_powers[q] = q.resonator.get_output_power(operation=readout_pulse_name)

            # If the amplitude increased above the maximum readout amplitude, increase the power
            # and reduce the amplitude below its limit to protect against saturating the output channel.
            #  However, since different resonators need different powers, but can share the same port,
            # take the safe approach and define a *lower* bound for the power of the port by
            # sorting in descending order of power requirements.
            lowest_possible_full_scale_power_dbm = None
            for qubit, power in sorted(optimal_output_powers.items(), key=lambda item: item[1], reverse=True):
                optimal_ds_for_this_qubit = optimal_ds.sel(qubit=qubit.name)

                power_settings = qubit.resonator.set_output_power(
                    power_in_dbm=power,
                    full_scale_power_dbm=lowest_possible_full_scale_power_dbm,
                    max_amplitude=node.parameters.max_readout_amplitude,
                    operation=readout_pulse_name
                )

                if lowest_possible_full_scale_power_dbm is None:
                    lowest_possible_full_scale_power_dbm = power_settings["full_scale_power_dbm"]

    # %% {Save_results}
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

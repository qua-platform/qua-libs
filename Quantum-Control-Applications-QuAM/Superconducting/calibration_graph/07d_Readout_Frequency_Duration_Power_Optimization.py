"""
        READOUT OPTIMISATION: FREQUENCY, POWER, DURATION
This sequence involves measuring the state of the resonator in two scenarios: first, after thermalization
(with the qubit in the |g> state) and then after applying a pi pulse to the qubit (transitioning the qubit to the
|e> state). This is done while varying the readout frequency.
The average I & Q quadratures for the qubit states |g> and |e>, along with their variances, are extracted to
determine the Signal-to-Noise Ratio (SNR). The readout frequency that yields the highest SNR is selected as the
optimal choice.

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
import numpy as np

from qualibrate import QualibrationNode

from quam_libs.experiments.readout_optimization_3d.analysis.fetch_dataset import fetch_dataset
from quam_libs.trackable_object import tracked_updates
from quam_libs.components import QuAM
from quam_libs.experiments.readout_optimization_3d.parameters import Parameters, get_durations
from quam_libs.experiments.readout_optimization_3d.parameters import (
    get_frequency_detunings_in_hz,
    get_amplitude_factors
)
from quam_libs.experiments.simulation import simulate_and_plot
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset

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
        num_averages=100,
        frequency_span_in_mhz=10,
        frequency_step_in_mhz=0.1,
        min_amplitude_factor=0.5,
        max_amplitude_factor=1.99,
        num_amplitudes=10,
        max_duration_in_ns=4000,
        num_durations=8,
        simulate=True,
        simulation_duration_ns=1000,
        use_waveform_report=True
    )
)

# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)

machine = QuAM.load("/home/dean/src/qm/qm/quams/quam_state_as_beta8")

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
n_avg = node.parameters.num_averages

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
        machine.set_all_fluxes(flux_point=flux_point, target=multiplexed_qubits[0])
        
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
                    .average() \
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
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]

    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    # Get the readout detuning as the index of the maximum of the cumulative average of D
    detuning = ds.D.rolling({"freq": 5}).mean("freq").idxmax("freq")
    # Get the dispersive shift as the distance between the resonator frequency when the qubit is in |g> and |e>
    chi = (ds.IQ_abs_e.idxmin(dim="freq") - ds.IQ_abs_g.idxmin(dim="freq")) / 2

    # Save fitting results
    fit_results = {q.name: {"detuning": detuning.loc[q.name].values, "chi": chi.loc[q.name].values} for q in qubits}
    node.results["fit_results"] = fit_results

    for q in qubits:
        print(f"{q.name}: Shifting readout frequency by {fit_results[q.name]['detuning']/1e3:.0f} kHz")
        print(f"{q.name}: Chi = {fit_results[q.name]['chi']:.2f} \n")

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        (1e3 * ds.assign_coords(freq_MHz=ds.freq / 1e6).D.loc[qubit]).plot(ax=ax, x="freq_MHz", label=None)
        ax.axvline(
            fit_results[qubit["qubit"]]["detuning"] / 1e6,
            color="red",
            linestyle="--",
            label="applied detuning",
        )
        ax.set_xlabel("Detuning [MHz]")
        ax.set_ylabel("Distance between IQ blobs [mv]")
        ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        (1e3 * ds.assign_coords(freq_MHz=ds.freq / 1e6).IQ_abs_g.loc[qubit]).plot(ax=ax, x="freq_MHz", label="g.s")
        (1e3 * ds.assign_coords(freq_MHz=ds.freq / 1e6).IQ_abs_e.loc[qubit]).plot(ax=ax, x="freq_MHz", label="e.s")
        ax.axvline(
            fit_results[qubit["qubit"]]["detuning"] / 1e6,
            color="red",
            linestyle="--",
            label="applied detuning",
        )
        ax.set_xlabel("Detuning [MHz]")
        ax.set_ylabel("Resonator response [mV]")
        ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    node.results["figure2"] = grid.fig

    # undo temporary readout length
    for tracked_resonator in tracked_resonators:
        tracked_resonator.revert_changes()

    # %% {Update_state}
    if node.parameters.load_data_id is None:
        for q in qubits:
            with node.record_state_updates():
                q.resonator.intermediate_frequency += int(fit_results[q.name]["detuning"])
                q.chi = float(fit_results[q.name]["chi"])

        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()


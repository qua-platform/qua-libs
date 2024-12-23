# %%
"""
        READOUT OPTIMISATION: FREQUENCY
This sequence involves measuring the state of the resonator in two scenarios: first, after thermalization
(with the qubit in the |g> state) and then after applying a pi pulse to the qubit (transitioning the qubit to the
|e> state). This is done while varying the readout frequency.
The average I & Q quadratures for the qubit states |g> and |e>, along with their variances, are extracted to
determine the Signal-to-Noise Ratio (SNR). The readout frequency that yields the highest SNR is selected as the
optimal choice.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the readout frequency  in the state.
    - Save the current state by calling machine.save("quam")
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from quam.components import pulses


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ["q1","q2","q3"]
    num_averages: int = 50
    ro_frequency_span_in_mhz: float = 10
    ro_frequency_step_in_mhz: float = 0.25
    detuning_span_in_mhz: float = 50
    detuning_step_in_mhz: float = 2
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    timeout: int = 100


node = QualibrationNode(name="11e_Readout_Frequency_Optimization_G_E_F_vs_detuning", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)
for q in qubits:  # TODO: weird since operation is a single string
    # Check if an optimized GEF frequency exists
    if not hasattr(q, "GEF_frequency_shift"):
        q.resonator.GEF_frequency_shift = 0
    # check if an EF_x180 operation exists
    if "EF_x180" in q.xy.operations:
        operation = "EF_x180"
    else:
        operation = "x180"


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# The frequency sweep around the resonator resonance frequency
ro_dfs = np.arange(
    -node.parameters.ro_frequency_span_in_mhz * u.MHz / 2,
    +node.parameters.ro_frequency_span_in_mhz * u.MHz / 2,
    node.parameters.ro_frequency_step_in_mhz * u.MHz,
)
detunings = np.arange(
    -node.parameters.detuning_span_in_mhz * u.MHz / 2,
    +node.parameters.detuning_span_in_mhz * u.MHz / 2,
    node.parameters.detuning_step_in_mhz * u.MHz,
)
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

with program() as ro_freq_opt:
    n = declare(int)
    I_g = [declare(fixed) for _ in range(num_qubits)]
    Q_g = [declare(fixed) for _ in range(num_qubits)]
    I_e = [declare(fixed) for _ in range(num_qubits)]
    Q_e = [declare(fixed) for _ in range(num_qubits)]
    I_f = [declare(fixed) for _ in range(num_qubits)]
    Q_f = [declare(fixed) for _ in range(num_qubits)]
    ro_df = declare(int)
    detuning = declare(int)
    I_g_st = [declare_stream() for _ in range(num_qubits)]
    Q_g_st = [declare_stream() for _ in range(num_qubits)]
    I_e_st = [declare_stream() for _ in range(num_qubits)]
    Q_e_st = [declare_stream() for _ in range(num_qubits)]
    I_f_st = [declare_stream() for _ in range(num_qubits)]
    Q_f_st = [declare_stream() for _ in range(num_qubits)]
    n_st = declare_stream()

    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point, qubit)

        align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(ro_df, ro_dfs)):
                # Update the resonator frequencies
                update_frequency(
                    qubit.resonator.name, ro_df + qubit.resonator.intermediate_frequency + qubit.resonator.GEF_frequency_shift
                )
                align()
                with for_(*from_array(detuning, detunings)):
                    wait(qubit.thermalization_time * u.ns)
                    # Measure the state of the resonators
                    qubit.resonator.measure("readout", qua_vars=(I_g[i], Q_g[i]))
                    qubit.align()
                    # Wait for thermalization again in case of measurement induced transitions
                    wait(qubit.thermalization_time * u.ns)      
                    save(I_g[i], I_g_st[i])
                    save(Q_g[i], Q_g_st[i])

                    wait(qubit.thermalization_time * u.ns)
                    # Play the x180 gate to put the qubits in the excited state
                    qubit.xy.play("x180")
                    # Align the elements to measure after playing the qubit pulses.
                    qubit.align()
                    # Measure the state of the resonators
                    qubit.resonator.measure("readout", qua_vars=(I_e[i], Q_e[i]))
                    # wait(1000)
                    qubit.align()
                    # Wait for thermalization again in case of measurement induced transitions
                    
                    save(I_e[i], I_e_st[i])
                    save(Q_e[i], Q_e_st[i])

                    wait(qubit.thermalization_time * u.ns)
                    # Play the x180 gate and EFx180 gate to put the qubits in the f state
                    qubit.xy.play("x180")
                    qubit.align()
                    update_frequency(qubit.xy.name, detuning + qubit.xy.intermediate_frequency - qubit.anharmonicity)
                    qubit.align()
                    qubit.xy.play(operation)
                    qubit.align()
                    update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency)
                    # Align the elements to measure after playing the qubit pulses.
                    qubit.align()
                    # Measure the state of the resonators
                    qubit.resonator.measure("readout", qua_vars=(I_f[i], Q_f[i]))
                    # Wait for the qubits to decay to the ground state
                    
                    save(I_f[i], I_f_st[i])
                    save(Q_f[i], Q_f_st[i])

        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].buffer(len(detunings)).buffer(len(ro_dfs)).average().save(f"I_g{i + 1}")
            Q_g_st[i].buffer(len(detunings)).buffer(len(ro_dfs)).average().save(f"Q_g{i + 1}")
            I_e_st[i].buffer(len(detunings)).buffer(len(ro_dfs)).average().save(f"I_e{i + 1}")
            Q_e_st[i].buffer(len(detunings)).buffer(len(ro_dfs)).average().save(f"Q_e{i + 1}")
            I_f_st[i].buffer(len(detunings)).buffer(len(ro_dfs)).average().save(f"I_f{i + 1}")
            Q_f_st[i].buffer(len(detunings)).buffer(len(ro_dfs)).average().save(f"Q_f{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, ro_freq_opt, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(ro_freq_opt)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:

    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"detuning": detunings, "ro_freq": ro_dfs})
    # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2) for |g>, [e> and |f> as well as the distance between the two blobs D
    ds = ds.assign(
        {
            "Dge": np.sqrt((ds.I_g - ds.I_e) ** 2 + (ds.Q_g - ds.Q_e) ** 2),
            "Def": np.sqrt((ds.I_e - ds.I_f) ** 2 + (ds.Q_e - ds.Q_f) ** 2),
            "Dgf": np.sqrt((ds.I_g - ds.I_f) ** 2 + (ds.Q_g - ds.Q_f) ** 2),
            "IQ_abs_g": np.sqrt(ds.I_g**2 + ds.Q_g**2),
            "IQ_abs_e": np.sqrt(ds.I_e**2 + ds.Q_e**2),
            "IQ_abs_f": np.sqrt(ds.I_f**2 + ds.Q_f**2),
        }
    )
    # Define D as  TODO
    ds["D"] = ds[["Dge", "Def", "Dgf"]].to_array().min("variable")
    # Add the absolute frequency to the dataset
    ds = ds.assign_coords(
        {
            "ro_freq_full": (
                ["qubit", "ro_freq"],
                np.array([ro_dfs + q.resonator.RF_frequency for q in qubits]),
            )
        }
    )
    ds.ro_freq_full.attrs["long_name"] = "Frequency"
    ds.ro_freq_full.attrs["units"] = "GHz"
    
    ds = ds.assign_coords(
        {
            "detuning_full": (
                ["qubit", "detuning"],
                np.array([detunings - q.anharmonicity for q in qubits]),
            )
        }
    )
    # Add the dataset to the node
    node.results = {"ds": ds}
# %%

for q in qubits:
    ds.sel(qubit=q.name).D.plot(x="ro_freq", y="detuning")
    plt.show()


# %% {Data_analysis}
if not node.parameters.simulate:

    # Find the indices of maximum detuning for each qubit
    max_indices = ds.D.argmax(dim=["ro_freq", "detuning"])

    # Extract the optimal freq and drive_freq values
    optimal_ro_freq = ds.ro_freq[max_indices["ro_freq"]]
    optimal_detuning = ds.detuning[max_indices["detuning"]]

    # Save fitting results
    fit_results = {
        q.name: {"GEF_ro_freq": int(optimal_ro_freq.loc[q.name].values), "GEF_detuning": float(optimal_detuning.loc[q.name].values)}
        for q in qubits
    }
    node.results["fit_results"] = fit_results

    for q in qubits:
        print(
            f"{q.name}: GEF readout frequency is shifted by {fit_results[q.name]['GEF_ro_freq']/1e6:.1f} MHz "
            f"and drive detuning by {fit_results[q.name]['GEF_detuning']/1e6:.1f} MHz\n"
        )


# %% {Plotting}


if not node.parameters.simulate:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ds.assign_coords(ro_freq_MHz=ds.ro_freq / 1e6).assign_coords(detuning_MHz=ds.detuning_full / 1e6).D.loc[qubit].plot(ax=ax, x="ro_freq_MHz", y="detuning_MHz", label="D")
        ax.plot(
            fit_results[qubit["qubit"]]["GEF_ro_freq"] / 1e6,
            (fit_results[qubit["qubit"]]["GEF_detuning"] - machine.qubits[qubit["qubit"]].anharmonicity) / 1e6,
            "ro",
            label="Optimal",
        )
        ax.set_xlabel("R/O Freq. [MHz]")
        ax.set_ylabel("Drive Detuning [MHz]")
        # ax.legend()
    plt.tight_layout()
    plt.suptitle("Maximal difference between g.e.f. resonance")
    plt.show()
    node.results["figure3"] = grid.fig

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        best_ds = ds.sel(detuning=fit_results[qubit["qubit"]]["GEF_detuning"])
        (1e3 * best_ds.assign_coords(ro_freq_MHz=best_ds.ro_freq_full / 1e6).Dge.loc[qubit]).plot(ax=ax, x="ro_freq_MHz", label="GE")
        (1e3 * best_ds.assign_coords(ro_freq_MHz=best_ds.ro_freq_full / 1e6).Def.loc[qubit]).plot(ax=ax, x="ro_freq_MHz", label="EF")
        (1e3 * best_ds.assign_coords(ro_freq_MHz=best_ds.ro_freq_full / 1e6).Dgf.loc[qubit]).plot(ax=ax, x="ro_freq_MHz", label="GF")
        (1e3 * best_ds.assign_coords(ro_freq_MHz=best_ds.ro_freq_full / 1e6).D.loc[qubit]).plot(ax=ax, x="ro_freq_MHz")
        # ax.axvline(
        #     fit_results[qubit["qubit"]]["GEF_detuning"] / 1e6,
        #     color="red",
        #     linestyle="--",
        # )
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Distance between IQ blobs [m.v.]")
        ax.legend()
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        best_ds = ds.sel(detuning=fit_results[qubit["qubit"]]["GEF_detuning"])
        (1e3 * best_ds.assign_coords(ro_freq_MHz=best_ds.ro_freq_full / 1e6).IQ_abs_g.loc[qubit]).plot(
            ax=ax, x="ro_freq_MHz", label="g.s."
        )
        (1e3 * best_ds.assign_coords(ro_freq_MHz=best_ds.ro_freq_full / 1e6).IQ_abs_e.loc[qubit]).plot(
            ax=ax, x="ro_freq_MHz", label="e.s."
        )
        (1e3 * best_ds.assign_coords(ro_freq_MHz=best_ds.ro_freq_full / 1e6).IQ_abs_f.loc[qubit]).plot(
            ax=ax, x="ro_freq_MHz", label="f.s."
        )
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Resonator response [mV]")
        ax.legend()
    plt.tight_layout()
    plt.show()
    node.results["figure2"] = grid.fig

# %% {Update_state}
if not node.parameters.simulate:
    for q in qubits:
        with node.record_state_updates():
            q.resonator.GEF_frequency_shift += int(fit_results[q.name]["GEF_ro_freq"])
            q.anharmonicity -= int(fit_results[q.name]["GEF_detuning"])

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%

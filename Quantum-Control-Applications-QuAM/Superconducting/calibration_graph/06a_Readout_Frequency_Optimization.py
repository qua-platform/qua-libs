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


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_averages: int = 100
    frequency_span_in_mhz: float = 10
    frequency_step_in_mhz: float = 0.1
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    timeout: int = 100


node = QualibrationNode(
    name="06a_Readout_Frequency_Optimization", parameters=Parameters()
)


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


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# The frequency sweep around the resonator resonance frequency 
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

with program() as ro_freq_opt:
    n = declare(int)
    I_g = [declare(fixed) for _ in range(num_qubits)]
    Q_g = [declare(fixed) for _ in range(num_qubits)]
    I_e = [declare(fixed) for _ in range(num_qubits)]
    Q_e = [declare(fixed) for _ in range(num_qubits)]
    df = declare(int)
    I_g_st = [declare_stream() for _ in range(num_qubits)]
    Q_g_st = [declare_stream() for _ in range(num_qubits)]
    I_e_st = [declare_stream() for _ in range(num_qubits)]
    Q_e_st = [declare_stream() for _ in range(num_qubits)]
    n_st = declare_stream()

    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            qubit.z.to_independent_idle()
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()

        # Wait for the flux bias to settle
        for qb in qubits:
            wait(1000, qb.z.name)

        align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(df, dfs)):
                # Update the resonator frequencies
                update_frequency(
                    qubit.resonator.name, df + qubit.resonator.intermediate_frequency
                )
                # Wait for the qubits to decay to the ground state
                wait(qubit.thermalization_time * u.ns)
                align()
                # Measure the state of the resonators
                qubit.resonator.measure("readout", qua_vars=(I_g[i], Q_g[i]))

                align()
                # Wait for thermalization again in case of measurement induced transitions
                wait(qubit.thermalization_time * u.ns)
                # Play the x180 gate to put the qubits in the excited state
                qubit.xy.play("x180")
                # Align the elements to measure after playing the qubit pulses.
                align()
                # Measure the state of the resonators
                qubit.resonator.measure("readout", qua_vars=(I_e[i], Q_e[i]))

                # Derive the distance between the blobs for |g> and |e>
                save(I_g[i], I_g_st[i])
                save(Q_g[i], Q_g_st[i])
                save(I_e[i], I_e_st[i])
                save(Q_e[i], Q_e_st[i])

        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].buffer(len(dfs)).average().save(f"I_g{i + 1}")
            Q_g_st[i].buffer(len(dfs)).average().save(f"Q_g{i + 1}")
            I_e_st[i].buffer(len(dfs)).average().save(f"I_e{i + 1}")
            Q_e_st[i].buffer(len(dfs)).average().save(f"Q_e{i + 1}")


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

        # %% {Live_plot}
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)

    # %% {Data_fetching_and_dataset_creation}
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs})
    # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2) for |g> and |e> as well as the distance between the two blobs D
    ds = ds.assign(
        {
            "D": np.sqrt((ds.I_g - ds.I_e) ** 2 + (ds.Q_g - ds.Q_e) ** 2),
            "IQ_abs_g": np.sqrt(ds.I_g**2 + ds.Q_g**2),
            "IQ_abs_e": np.sqrt(ds.I_e**2 + ds.Q_e**2),
        }
    )
    # Add the absolute frequency to the dataset
    ds = ds.assign_coords(
        {
            "freq_full": (
                ["qubit", "freq"],
                np.array([dfs + q.resonator.RF_frequency for q in qubits]),
            )
        }
    )
    ds.freq_full.attrs["long_name"] = "Frequency"
    ds.freq_full.attrs["units"] = "GHz"
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    # Get the readout detuning as the index of the maximum of the cumulative average of D
    detuning = ds.D.rolling({"freq": 5}).mean("freq").idxmax("freq")
    # Get the dispersive shift as the distance between the resonator frequency when the qubit is in |g> and |e>
    chi = (ds.IQ_abs_e.idxmin(dim="freq") - ds.IQ_abs_g.idxmin(dim="freq")) / 2
    
    # Save fitting results
    fit_results = {
        q.name: {"detuning": detuning.loc[q.name].values, "chi": chi.loc[q.name].values}
        for q in qubits
    }
    node.results["fit_results"] = fit_results

    for q in qubits:
        print(
            f"{q.name}: Shifting readout frequency by {fit_results[q.name]['detuning']/1e3:.0f} kHz"
        )
        print(f"{q.name}: Chi = {fit_results[q.name]['chi']:.2f} \n")

    # %% {Plotting}
    grid_names = [f"{q.name}_0" for q in qubits]
    grid = QubitGrid(ds, grid_names)
    for ax, qubit in grid_iter(grid):
        (1e3 * ds.assign_coords(freq_MHz=ds.freq / 1e6).D.loc[qubit]).plot(
            ax=ax, x="freq_MHz"
        )
        ax.axvline(
            fit_results[qubit["qubit"]]["detuning"] / 1e6, color="red", linestyle="--"
        )
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Distance between IQ blobs [m.v.]")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    grid = QubitGrid(ds, [f"q-{i}_0" for i in range(num_qubits)])
    for ax, qubit in grid_iter(grid):
        (1e3 * ds.assign_coords(freq_MHz=ds.freq / 1e6).IQ_abs_g.loc[qubit]).plot(
            ax=ax, x="freq_MHz", label="g.s"
        )
        (1e3 * ds.assign_coords(freq_MHz=ds.freq / 1e6).IQ_abs_e.loc[qubit]).plot(
            ax=ax, x="freq_MHz", label="e.s"
        )
        ax.axvline(
            fit_results[qubit["qubit"]]["detuning"] / 1e6, color="red", linestyle="--"
        )
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Resonator response [mV]")
        ax.legend()
    plt.tight_layout()
    plt.show()
    node.results["figure2"] = grid.fig

    # %% {Update_state}
    for q in qubits:
        with node.record_state_updates():
            q.resonator.intermediate_frequency += int(fit_results[q.name]["detuning"])
            q.chi = float(fit_results[q.name]["chi"])

    # %% {Save_results}
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

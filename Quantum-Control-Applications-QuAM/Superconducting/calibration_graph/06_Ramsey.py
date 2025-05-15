"""
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


# %% {Imports}
from datetime import datetime, timezone, timedelta
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset, get_node_id, save_node
from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp
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
    frequency_detuning_in_mhz: float = 1.0
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 3000
    num_time_points: int = 100
    log_or_linear_sweep: Literal["log", "linear"] = "linear"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    use_state_discrimination: bool = False
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True

node = QualibrationNode(name="06_Ramsey", parameters=Parameters())
node_id = get_node_id()

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
    
# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
if node.parameters.log_or_linear_sweep == "linear":
    idle_times = (
        np.linspace(
            node.parameters.min_wait_time_in_ns,
            node.parameters.max_wait_time_in_ns,
            node.parameters.num_time_points,
        )
        // 4
    ).astype(int)
else:
    idle_times = np.unique(
        np.geomspace(
            node.parameters.min_wait_time_in_ns,
            node.parameters.max_wait_time_in_ns,
            node.parameters.num_time_points,
        )
        // 4
    ).astype(int)

# Detuning converted into virtual Z-rotations to observe Ramsey oscillation and get the qubit frequency
detuning = int(1e6 * node.parameters.frequency_detuning_in_mhz)
flux_point = node.parameters.flux_point_joint_or_independent

with program() as ramsey:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    t = declare(int)  # QUA variable for the idle time
    sign = declare(int)  # QUA variable to change the sign of the detuning
    # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)
    phi = declare(fixed)
    if node.parameters.use_state_discrimination:
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]

    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_each_(t, idle_times):
                #  with for_(*from_array(t, idle_times)):
                with for_(*from_array(sign, [-1, 1])):
                    # Rotate the frame of the second x90 gate to implement a virtual Z-rotation
                    assign(phi, Util.cond((sign == 1),  Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t), Cast.mul_fixed_by_int(-detuning * 1e-9, 4 * t)))
                    qubit.align()
                    # # Strict_timing ensures that the sequence will be played without gaps
                    # with strict_timing_():
                    qubit.xy.play("x90")
                    qubit.xy.wait(t)
                    qubit.xy.frame_rotation_2pi(phi)
                    qubit.xy.play("x90")

                    # Align the elements to measure after playing the qubit pulse.
                    qubit.align()
                    # Measure the state of the resonators and save data
                    if node.parameters.use_state_discrimination:
                        readout_state(qubit, state[i], wait_depletion_time=False)
                        save(state[i], state_st[i])
                    else:
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])

                    # Wait for the qubits to decay to the ground state
                    qubit.resonator.wait(qubit.thermalization_time * u.ns)
                    # Reset the frame of the qubits in order not to accumulate rotations
                    reset_frame(qubit.xy.name)
        # Measure sequentially
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            if node.parameters.use_state_discrimination:
                state_st[i].buffer(2).buffer(len(idle_times)).average().save(f"state{i + 1}")
            else:
                I_st[i].buffer(2).buffer(len(idle_times)).average().save(f"I{i + 1}")
                Q_st[i].buffer(2).buffer(len(idle_times)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, ramsey, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(ramsey)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)


# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"sign": [-1, 1], "time": idle_times})
        # Convert IQ data into volts
        if not node.parameters.use_state_discrimination:
            ds = convert_IQ_to_V(ds, qubits)
        # Add the absolute time to the dataset
        ds = ds.assign_coords({"time": (["time"], 4 * idle_times)})
        ds.time.attrs["long_name"] = "idle_time"
        ds.time.attrs["units"] = "ns"
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    # Add the dataset to the node
    node.results = {"ds": ds}


    # %% {Data_analysis}
    # Fit the Ramsey oscillations based on the qubit state or the 'I' quadrature
    if node.parameters.use_state_discrimination:
        fit = fit_oscillation_decay_exp(ds.state, "time")
    else:
        fit = fit_oscillation_decay_exp(ds.I, "time")
    fit.attrs = {"long_name": "time", "units": "µs"}
    fitted = oscillation_decay_exp(
        ds.time,
        fit.sel(fit_vals="a"),
        fit.sel(fit_vals="f"),
        fit.sel(fit_vals="phi"),
        fit.sel(fit_vals="offset"),
        fit.sel(fit_vals="decay"),
    )
    # TODO: add meaningful comments
    frequency = fit.sel(fit_vals="f")
    frequency.attrs = {"long_name": "frequency", "units": "MHz"}

    decay = fit.sel(fit_vals="decay")
    decay.attrs = {"long_name": "decay", "units": "nSec"}

    frequency = frequency.where(frequency > 0, drop=True)

    decay = fit.sel(fit_vals="decay")
    decay.attrs = {"long_name": "decay", "units": "nSec"}

    decay_res = fit.sel(fit_vals="decay_decay")
    decay_res.attrs = {"long_name": "decay", "units": "nSec"}

    tau = 1 / fit.sel(fit_vals="decay")
    tau.attrs = {"long_name": "T2*", "units": "uSec"}

    tau_error = tau * (np.sqrt(decay_res) / decay)
    tau_error.attrs = {"long_name": "T2* error", "units": "uSec"}

    within_detuning = (1e9 * frequency < 2 * detuning).mean(dim="sign") == 1
    positive_shift = frequency.sel(sign=1) > frequency.sel(sign=-1)
    freq_offset = (
        within_detuning * (frequency * fit.sign).mean(dim="sign")
        + ~within_detuning * positive_shift * frequency.mean(dim="sign")
        - ~within_detuning * ~positive_shift * frequency.mean(dim="sign")
    )
    decay = 1e-9 * tau.mean(dim="sign")
    decay_error = 1e-9 * tau_error.mean(dim="sign")

    # Save fitting results
    fit_results = {
        q.name: {
            "freq_offset": 1e9 * freq_offset.loc[q.name].values,
            "decay": decay.loc[q.name].values,
            "decay_error": decay_error.loc[q.name].values,
        }
        for q in qubits
    }
    node.results["fit_results"] = fit_results
    for q in qubits:
        print(f"Frequency offset for qubit {q.name} : {(fit_results[q.name]['freq_offset']/1e6):.2f} MHz ")
        print(f"T2* for qubit {q.name} : {1e6*fit_results[q.name]['decay']:.2f} us")


    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            ds.sel(sign=1).loc[qubit].state.plot(
                ax=ax, x="time", c="C0", marker=".", ms=5.0, ls="", label="$\Delta$ = +"
            )
            ds.sel(sign=-1).loc[qubit].state.plot(
                ax=ax, x="time", c="C1", marker=".", ms=5.0, ls="", label="$\Delta$ = -"
            )
            ax.plot(ds.time, fitted.loc[qubit].sel(sign=1), c="C0", ls="-", lw=1)
            ax.plot(ds.time, fitted.loc[qubit].sel(sign=-1), c="C1", ls="-", lw=1)
            ax.set_ylabel("State")
        else:
            (ds.sel(sign=1).loc[qubit].I * 1e3).plot(
                ax=ax, x="time", c="C0", marker=".", ms=5.0, ls="", label="$\Delta$ = +"
            )
            (ds.sel(sign=-1).loc[qubit].I * 1e3).plot(
                ax=ax, x="time", c="C1", marker=".", ms=5.0, ls="", label="$\Delta$ = -"
            )
            ax.set_ylabel("Trans. amp. I [mV]")
            ax.plot(ds.time, 1e3 * fitted.loc[qubit].sel(sign=1), c="C0", ls="-", lw=1)
            ax.plot(ds.time, 1e3 * fitted.loc[qubit].sel(sign=-1), c="C1", ls="-", lw=1)

        ax.set_xlabel("Idle time [ns]")
        ax.set_title(qubit["qubit"])
        ax.text(
            0.1,
            0.9,
            f'T2* = {1e6*fit_results[qubit["qubit"]]["decay"]:.1f} ± {1e6*fit_results[qubit["qubit"]]["decay_error"]:.1f} µs',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )
        ax.legend()
    grid.fig.suptitle(f"Ramsey : I vs. idle time \n {date_time} GMT+3 #{node_id} \n multiplexed = {node.parameters.multiplexed}")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig


    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for q in qubits:
                q.xy.intermediate_frequency -= float(fit_results[q.name]["freq_offset"])
                q.T2ramsey = float(fit_results[q.name]["decay"])


        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        save_node(node)

# %%

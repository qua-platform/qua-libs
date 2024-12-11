"""
        POWER RABI WITH ERROR AMPLIFICATION
This sequence involves repeatedly executing the qubit pulse (such as x180) 'N' times and
measuring the state of the resonator across different qubit pulse amplitudes and number of pulses.
By doing so, the effect of amplitude inaccuracies is amplified, enabling a more precise measurement of the pi pulse
amplitude. The results are then analyzed to determine the qubit pulse amplitude suitable for the selected duration.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated the IQ mixer connected to the qubit drive line (external mixer or Octave port)
    - Having found the rough qubit frequency and set the desired pi pulse duration (qubit spectroscopy).
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the qubit pulse amplitude in the state.
    - Save the current state
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset
from quam_libs.lib.instrument_limits import instrument_limits
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_oscillation, oscillation
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
    num_averages: int = 50
    operation_x180_or_any_90: Literal["z180", "z90", "-z90"] = "-z90"
    min_amp_factor: float = 0.9
    max_amp_factor: float = 1.1
    amp_factor_step: float = 0.005
    max_number_rabi_pulses_per_sweep: int = 50
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    state_discrimination: bool = True
    update_x90: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False

node = QualibrationNode(name="22_Z_gate_error_amplification", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
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
N_pi = node.parameters.max_number_rabi_pulses_per_sweep  # Number of applied Rabi pulses sweep
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
state_discrimination = node.parameters.state_discrimination
operation = node.parameters.operation_x180_or_any_90  # The qubit operation to play
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
amps = np.arange(
    node.parameters.min_amp_factor,
    node.parameters.max_amp_factor,
    node.parameters.amp_factor_step,
)

# Number of applied Rabi pulses sweep
if operation == "z180":
    N_pi_vec = np.arange(1, N_pi, 2).astype("int")
elif operation in ["z90", "-z90"]:
    N_pi_vec = np.arange(2, N_pi, 4).astype("int")
else:
    raise ValueError(f"Unrecognized operation {operation}.")



with program() as power_rabi:
    I, I_st, Q, Q_st, _, n_st = qua_declaration(num_qubits=num_qubits)
    if state_discrimination:
        state = [declare(bool) for _ in range(num_qubits)]
        state_stream = [declare_stream() for _ in range(num_qubits)]

    shots = [declare(int) for _ in range(num_qubits)]
    a = [declare(fixed) for _ in range(num_qubits)]  # QUA variable for the qubit drive amplitude pre-factor
    npi = [declare(int) for _ in range(num_qubits)]  # QUA variable for the number of qubit pulses
    count = [declare(int) for _ in range(num_qubits)]  # QUA variable for counting the qubit pulses

    if node.parameters.multiplexed:
        for i , qubit in enumerate(qubits):
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)

    for i, qubit in enumerate(qubits):
        if not node.parameters.multiplexed:
            # Bring the active qubits to the desired frequency point
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(shots[i], 0, shots[i] < n_avg, shots[i] + 1):
            save(shots[i], n_st)
            with for_(*from_array(npi[i], N_pi_vec)):
                with for_(*from_array(a[i], amps)):
                    # Initialize the qubits
                    if reset_type == "active":
                        active_reset(qubit, "readout")
                    else:
                        qubit.wait(qubit.thermalization_time * u.ns)

                    # Loop for error amplification (perform many qubit pulses)
                    qubit.xy.play("x90")
                    qubit.align()
                    with for_(count[i], 0, count[i] < npi[i], count[i] + 1):
                        qubit.align()
                        qubit.z.play(operation, amplitude_scale=a[i])
                        qubit.align()
                    qubit.align() 
                    qubit.xy.play("x90")                        
                    qubit.align()
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    if state_discrimination:
                        assign(state[i], I[i] > qubit.resonator.operations["readout"].threshold)
                        save(state[i], state_stream[i])
                    else:
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            if state_discrimination:
                state_stream[i].boolean_to_int().buffer(len(amps)).buffer(len(N_pi_vec)).average().save(
                    f"state{i + 1}"
                )
            else:
                I_st[i].buffer(len(amps)).buffer(len(N_pi_vec)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(amps)).buffer(len(N_pi_vec)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, power_rabi, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(power_rabi)
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
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"amp": amps, "N": N_pi_vec})
        if not state_discrimination:
            ds = convert_IQ_to_V(ds, qubits)
        # Add the qubit pulse absolute amplitude to the dataset
        ds = ds.assign_coords(
        {
            "abs_amp": (
                ["qubit", "amp"],
                np.array([q.z.operations[operation].amplitude * amps for q in qubits]),
            )
            }
        )
    else:
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    fit_results = {}

    # Get the average along the number of pulses axis to identify the best pulse amplitude
    if state_discrimination:
        I_n = ds.state.mean(dim="N")
    else:
        I_n = ds.I.mean(dim="N")
    data_max_idx = I_n.argmin(dim="amp")


    # Save fitting results
    for q in qubits:
        new_pi_amp = float(ds.abs_amp.sel(qubit=q.name)[data_max_idx.sel(qubit=q.name)].data)
        fit_results[q.name] = {}
        limits = instrument_limits(q.xy)
        if new_pi_amp < limits.max_x180_wf_amplitude:
            fit_results[q.name]["Pi_amplitude"] = new_pi_amp
            print(
                f"amplitude for Pi pulse is modified by a factor of {I_n.idxmax(dim='amp').sel(qubit = q.name):.2f}"
            )
            print(f"new amplitude is {1e3 * new_pi_amp:.2f} {limits.units} \n")
        else:
            print(f"Fitted amplitude too high, new amplitude is {limits.max_x180_wf_amplitude} \n")
            fit_results[q.name]["Pi_amplitude"] = limits.max_x180_wf_amplitude

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        if state_discrimination:
            ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qubit].state.plot(ax=ax, x="amp_mV", y="N")
        else:
            (ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qubit].I * 1e3).plot(ax=ax, x="amp_mV", y="N")
        ax.set_ylabel("num. of pulses")
        ax.axvline(1e3 * ds.abs_amp.loc[qubit][data_max_idx.loc[qubit]], color="r", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Amplitude [mV]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle(f"Z gate error amplification {operation}")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for q in qubits:
                q.z.operations[operation].amplitude = fit_results[q.name]["Pi_amplitude"]

    # %% {Save_results}
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%

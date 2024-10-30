# %%
"""
        DRAG PULSE CALIBRATION (GOOGLE METHOD)
The sequence consists in applying an increasing number of x180 and -x180 pulses successively while varying the DRAG
coefficient alpha. After such a sequence, the qubit is expected to always be in the ground state if the DRAG
coefficient has the correct value. Note that the idea is very similar to what is done in power_rabi_error_amplification.

This protocol is described in more details in https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.190503

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit spectroscopy, rabi_chevron, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR and state discrimination.
    - Set the DRAG coefficient to a non-zero value in the config: such as drag_coef = -1
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update the DRAG coefficient (alpha) in the state.
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.trackable_object import tracked_updates
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
    num_averages: int = 10
    operation: str = "x180"
    min_amp_factor: float = 0.0001
    max_amp_factor: float = 2.0
    amp_factor_step: float = 0.02
    max_number_pulses_per_sweep: int = 40
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    simulate: bool = False
    timeout: int = 100


node = QualibrationNode(
    name="09b_DRAG_Calibration_180_minus_180", parameters=Parameters()
)


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
operation = node.parameters.operation  # The qubit operation to play

if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

# Update the readout power to match the desired range, this change will be reverted at the end of the node.
tracked_qubits = []
for q in qubits:
    with tracked_updates(q, auto_revert=False, dont_assign_to_none=True) as q:
        q.xy.operations[operation].alpha = -1.0
        tracked_qubits.append(q)

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()

if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_thermal_or_active
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
amps = np.arange(
    node.parameters.min_amp_factor,
    node.parameters.max_amp_factor,
    node.parameters.amp_factor_step,
)
# Number of applied Rabi pulses sweep
N_pi = node.parameters.max_number_pulses_per_sweep  # Maximum number of qubit pulses
N_pi_vec = np.linspace(1, N_pi, N_pi).astype("int")

with program() as drag_calibration:
    I, _, Q, _, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(bool) for _ in range(num_qubits)]
    state_stream = [declare_stream() for _ in range(num_qubits)]
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    npi = declare(int)  # QUA variable for the number of qubit pulses
    count = declare(int)  # QUA variable for counting the qubit pulses

    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the desired frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            qubit.z.to_independent_idle()
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(npi, N_pi_vec)):
                with for_(*from_array(a, amps)):
                    # Initialize the qubits
                    if reset_type == "active":
                        active_reset(qubit, "readout")
                    else:
                        qubit.resonator.wait(qubit.thermalization_time * u.ns)
                    qubit.align()
                    # Loop for error amplification (perform many qubit pulses)
                    with for_(count, 0, count < npi, count + 1):
                        if operation == "x180":
                            play(operation * amp(1, 0, 0, a), qubit.xy.name)
                            play(operation * amp(-1, 0, 0, -a), qubit.xy.name)
                        elif operation == "x90":
                            play(operation * amp(1, 0, 0, a), qubit.xy.name)
                            play(operation * amp(1, 0, 0, a), qubit.xy.name)
                            play(operation * amp(-1, 0, 0, -a), qubit.xy.name)
                            play(operation * amp(-1, 0, 0, -a), qubit.xy.name)

                    qubit.align()
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    assign(
                        state[i], I[i] > qubit.resonator.operations["readout"].threshold
                    )
                    save(state[i], state_stream[i])
        # Measure sequentially
        align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            state_stream[i].boolean_to_int().buffer(len(amps)).buffer(
                N_pi
            ).average().save(f"state{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, drag_calibration, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(drag_calibration)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

    # %% {Data_fetching_and_dataset_creation}
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(
        job.result_handles, qubits, {"amp": amps, "N": N_pi_vec}
    )
    # Add the qubit pulse absolute alpha coefficient to the dataset
    ds = ds.assign_coords(
        {
            "alpha": (
                ["qubit", "amp"],
                np.array([q.xy.operations[operation].alpha * amps for q in qubits]),
            )
        }
    )
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    # Get the average along the number of pulses axis to identify the best DRAG coefficient
    state_n = ds.state.mean(dim="N")
    data_max_idx = state_n.argmin(dim="amp")
    alphas = ds.amp[data_max_idx]
    # Save fitting results
    fit_results = {
        qubit.name: {
            "alpha": float(
                alphas.sel(qubit=qubit.name).values
                * qubit.xy.operations[operation].alpha
            )
        }
        for qubit in qubits
    }
    for q in qubits:
        print(f"DRAG coefficient for {q.name} is {fit_results[q.name]['alpha']}")
    node.results["fit_results"] = fit_results

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ds.loc[qubit].state.plot(ax=ax, x="alpha", y="N")
        ax.axvline(fit_results[qubit["qubit"]]["alpha"], color="r")
        ax.set_ylabel("num. of pulses")
        ax.set_xlabel(r"DRAG coeff $\alpha$")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("DRAG calibration")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Update_state}
    # Revert the change done at the beginning of the node
    for qubit in tracked_qubits:
        qubit.revert_changes()
    # Update the state
    with node.record_state_updates():
        for q in qubits:
            q.xy.operations[operation].alpha = fit_results[q.name]["alpha"]

    # %% {Save_results}
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%

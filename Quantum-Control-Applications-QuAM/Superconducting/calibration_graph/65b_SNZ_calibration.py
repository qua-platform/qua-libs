# %%
"""
Unipolar CPhase Gate Calibration

This sequence measures the time and detuning required for a unipolar CPhase gate. The process involves:

1. Preparing both qubits in their excited states.
2. Applying a flux pulse with varying amplitude and duration.
3. Measuring the resulting state populations as a function of these parameters.
4. Fitting the results to a Ramsey-Chevron pattern.

From this pattern, we extract:
- The coupling strength (J2) between the qubits.
- The optimal gate parameters (amplitude and duration) for the CPhase gate.

The Ramsey-Chevron pattern emerges due to the interplay between the qubit-qubit coupling and the flux-induced detuning, allowing us to precisely calibrate the CPhase gate.

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair.
- Calibrated readout for both qubits.
- Initial estimate of the flux pulse amplitude range.

Outcomes:
- Extracted J2 coupling strength.
- Optimal flux pulse amplitude and duration for the CPhase gate.
- Fitted Ramsey-Chevron pattern for visualization and verification.
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef, active_reset_gef, active_reset_simple
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset, save_node
from quam_libs.lib.fit import fit_oscillation
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import warnings
from qualang_tools.bakery import baking
from quam_libs.lib.fit import fit_oscillation, oscillation, fix_oscillation_phi_2pi
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from scipy.optimize import curve_fit
from quam_libs.components.gates.two_qubit_gates import CZGate, CZWithCompensationGate
from quam_libs.lib.pulses import FluxPulse, SNZPulse


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["coupler_q1_q2"]
    num_averages: int = 1000
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal["active", "thermal"] = "active"
    simulate: bool = False
    timeout: int = 100
    method: Literal["coarse", "fine"] = "fine"
    amp_range: float = 0.3
    amp_step: float = 0.01
    rel_start: float = 0.2
    rel_stop: float = 1.0
    rel_num: int = 4
    spacing: int = 2
    extra_flux_pulse_length: int = 2
    num_frames: int = 10
    load_data_id: Optional[int] = None


node = QualibrationNode(name="65b_SNZ_calibration", parameters=Parameters())
assert not (
    node.parameters.simulate and node.parameters.load_data_id is not None
), "If simulate is True, load_data_id must be None, and vice versa."


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]
# if any([qp.q1.z is None or qp.q2.z is None for qp in qubit_pairs]):
#     warnings.warn("Found qubit pairs without a flux line. Skipping")

num_qubit_pairs = len(qubit_pairs)

# Generate the OPX and Octave configurations
config = machine.generate_config()

octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
# %%

####################
# Helper functions #
####################


def generate_snz_config(config, rel_vals: list, spacing: int):
    for qp in qubit_pairs:
        amplitude = qp.gates["Cz_unipolar"].flux_pulse_control.amplitude
        length = qp.gates["Cz_unipolar"].flux_pulse_control.length
        for rel_num, rel in enumerate(rel_vals):
            pulse = SNZPulse(
                amplitude=amplitude,
                step_amplitude=amplitude * rel,
                length=length + node.parameters.extra_flux_pulse_length,
                step_length=1,
                spacing=spacing,
                id=f"CZ_snz_{rel_num}_{qp.qubit_target.name}",
            )
            pulse.parent = qp.qubit_control.z
            pulse.apply_to_config(config)
            plt.plot(pulse.waveform_function())
            config["elements"][qp.qubit_control.z.name]["operations"][pulse.id] = pulse.pulse_name
    return config


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

# define the amplitudes for the flux pulses
rel_vals = np.linspace(node.parameters.rel_start, node.parameters.rel_stop, node.parameters.rel_num)
amplitudes = np.arange(1 - node.parameters.amp_range, 1 + node.parameters.amp_range, node.parameters.amp_step)
frames = np.arange(0, 1, 1 / node.parameters.num_frames)

config = generate_snz_config(config, rel_vals=rel_vals, spacing=node.parameters.spacing)


# %%

with program() as CPhase_Oscillations:

    amplitude = declare(fixed)
    frame = declare(fixed)
    rel_val = declare(int)
    rel_ind = declare(int)
    control_initial = declare(int)
    n = declare(int)

    comp_flux_qubit = declare(fixed)
    comp_flux_coupler = declare(fixed)
    
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]

    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    n_st = declare_stream()

    for i, qp in enumerate(qubit_pairs):
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point, qp.qubit_control)

        assign(comp_flux_coupler, qp.extras["CZ_coupler_flux"]) 

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(frame, frames)):

                with for_(*from_array(amplitude, amplitudes)):

                    with for_(*from_array(rel_val, np.arange(len(rel_vals)))):

                        with for_(*from_array(control_initial, [0, 1])):
                            # reset
                            if node.parameters.reset_type == "active":
                                active_reset_simple(qp.qubit_control)
                                active_reset_simple(qp.qubit_target)
                            else:
                                wait(qp.qubit_control.thermalization_time * u.ns)

                            qp.align()
                            with if_(control_initial == 1):
                                qp.qubit_control.xy.play("x180")
                                qp.qubit_target.xy.play("x180")
                            qp.qubit_target.xy.play("x90")
                            qp.qubit_control.z.wait(qp.qubit_target.xy.operations["x90"].length + 5)
                            
                            qp.align()

                            with switch_(rel_val):
                                for rel_ind in range(len(rel_vals)):
                                    with case_(rel_ind):
                                        qp.qubit_control.z.play(
                                            f"CZ_snz_{rel_ind}_{qp.qubit_target.name}",
                                            amplitude_scale=amplitude,
                                            validate=False,
                                        )
                            qp.coupler.play("const", amplitude_scale = comp_flux_coupler / qp.coupler.operations["const"].amplitude, duration = qp.gates["Cz_unipolar"].flux_pulse_control.length // 4 + 3)
                            
                            qp.align()
                            frame_rotation_2pi(frame, qp.qubit_target.xy.name)
                            qp.qubit_target.xy.play("x90")
                            qp.align()

                            # measure both qubits
                            readout_state_gef(qp.qubit_control, state_control[i])
                            readout_state(qp.qubit_target, state_target[i])
                            save(state_control[i], state_st_control[i])
                            save(state_target[i], state_st_target[i])

        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_st_control[i].buffer(2).buffer(len(rel_vals)).buffer(len(amplitudes)).buffer(len(frames)).buffer(
                n_avg
            ).save(f"state_control{i + 1}")
            state_st_target[i].buffer(2).buffer(len(rel_vals)).buffer(len(amplitudes)).buffer(len(frames)).buffer(
                n_avg
            ).save(f"state_target{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=300)  # In clock cycles = 4ns
    job = qmm.simulate(config, CPhase_Oscillations, simulation_config)
    job.get_simulated_samples().con1.plot()
    job.get_simulated_samples().con2.plot()
    plt.xlim(400, 300 * 4)
    plt.legend(loc="upper right")
    node.results = {"figure_simulated": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout ) as qm:
        job = qm.execute(CPhase_Oscillations)

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
        ds = fetch_results_as_xarray(
            job.result_handles,
            qubit_pairs,
            {"control_ax": [0, 1], "rel_val": rel_vals, "amp": amplitudes, "frame": frames, "avg": np.arange(n_avg)},
        )

        def abs_amp(qp, amp):
            return amp * qp.gates["Cz_unipolar"].flux_pulse_control.amplitude

        ds = ds.assign_coords({"amp_full": (["qubit", "amp"], np.array([abs_amp(qp, ds.amp) for qp in qubit_pairs]))})
    else:
        ds, loaded_machine = load_dataset(node.parameters.load_data_id)
        if loaded_machine is not None:
            machine = loaded_machine

    node.results = {"ds": ds}


# %%
if not node.parameters.simulate:
    prob = ds.state_target.where(ds.state_target != 2).mean("avg")
    leak = (ds.sel(control_ax=1).state_control == 2).sum(dim={"avg", "frame"}) / (ds.avg.size * ds.frame.size)
    fit_data = fit_oscillation(prob, "frame")
    phase = fix_oscillation_phi_2pi(fit_data)
    phase_diff = phase.diff(dim="control_ax")

    node.results["fit_results"] = {}
    best_points = {}
    phase_diffs = {}
    optimal_amps = {}
    leaks = {}
    fitted = {}

    # %%
    phase_diff_roll = phase_diff.rolling(rel_val=1, center=True).mean().rolling(amp=1, center=True).mean()
    leak_roll = leak.rolling(rel_val=1, center=True).mean().rolling(amp=1, center=True).mean()

    mask = np.abs(phase_diff_roll - 0.5) < 0.05
    leak_mask = leak_roll * mask + (1 - mask)
    min_value = leak_mask.min(dim=["amp", "rel_val", "control_ax"])
    min_coords = {}
    for q in phase_diff.qubit.values:
        min_coords[q] = leak_mask.sel(qubit=q).where(leak_mask.sel(qubit=q) == min_value.sel(qubit=q), drop=True)[0]

    for q in phase_diff.qubit.values:
        f, ax = plt.subplots(1, 2, figsize=(6, 3))
        # Plot the main 2D data
        (phase_diff - 0.5).sel(qubit=q).plot(ax=ax[0], x="amp", cmap="twilight")
        # ax[0].scatter(min_coords[q].amp, min_coords[q].rel_val, color="red", s=20)
        leak.sel(qubit=q).plot(ax=ax[1], x="amp")
        # ax[1].scatter(min_coords[q].amp, min_coords[q].rel_val, color="red", s=20)

        # Get the data for this qubit and control_ax=1
        mask_q = mask.sel(qubit=q, control_ax=1)

        # Create 2D arrays of coordinates
        amp_coords = mask_q.amp.values
        rel_coords = mask_q.rel_val.values
        # Get the indices where mask is True
        true_indices = np.where(mask_q.values)

        # Get the corresponding coordinate values
        valid_amps = amp_coords[true_indices[0]]
        valid_rels = rel_coords[true_indices[1]]

        if len(valid_amps) > 0:
            ax[1].scatter(valid_amps, valid_rels, alpha=0.5, color="red", s=1)


# %%
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):

        (phase_diff - 0.5).loc[qubit_pair["qubit"]].assign_coords(
            amp_full_mV=1e3 * ds.amp_full.loc[qubit_pair["qubit"]]
        ).plot(ax=ax, x="amp_full_mV", cmap="twilight", cbar_kwargs={"label": "CPhase"})
        ax.scatter(
            1e3 * min_coords[qubit_pair["qubit"]].amp_full,
            min_coords[qubit_pair["qubit"]].rel_val,
            color="red",
            s=20,
        )
        ax.set_xlabel("Flux amplitude [V]")
        ax.set_ylabel("Relative step size")
        quad = machine.qubit_pairs[qubit_pair["qubit"]].qubit_control.freq_vs_flux_01_quad_term

        def detuning_to_flux(det, quad=quad):
            return 1e3 * np.sqrt(-1e6 * det / quad)

        def flux_to_detuning(flux, quad=quad):
            return -1e-6 * (flux / 1e3) ** 2 * quad

        ax2 = ax.secondary_xaxis("top", functions=(flux_to_detuning, detuning_to_flux))
        ax2.set_xlabel("Detuning [MHz]")

        ax.set_title(qubit_pair["qubit"])

    # plt.suptitle("Obtained Cphase")
    plt.show()
    node.results["figure_phase"] = grid.fig

    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):

        im = (
            leak.loc[qubit_pair["qubit"]]
            .assign_coords(amp_full_mV=1e3 * ds.amp_full.loc[qubit_pair["qubit"]])
            .plot(ax=ax, x="amp_full_mV", cmap="rainbow", cbar_kwargs={"label": "Leakage"})
        )

        ax.scatter(
            1e3 * min_coords[qubit_pair["qubit"]].amp_full,
            min_coords[qubit_pair["qubit"]].rel_val,
            color="red",
            s=20,
        )
        ax.set_xlabel("Flux amplitude [V]")
        ax.set_ylabel("Relative step size")
        # im = leak.loc[qubit_pair["qubit"]].assign_coords(amp_full_mV=1e3 * ds.amp_full.loc[qubit_pair["qubit"]]).plot(
        #     ax=ax, x="amp_full_mV", cmap="rainbow"
        # )

        # Get the data for this qubit and control_ax=1
        mask_q = mask.loc[qubit_pair["qubit"]].sel(control_ax=1)

        # Create 2D arrays of coordinates
        amp_coords = 1e3 * mask_q.amp_full.values
        rel_coords = mask_q.rel_val.values
        # Get the indices where mask is True
        true_indices = np.where(mask_q.values)

        # Get the corresponding coordinate values
        valid_amps = amp_coords[true_indices[0]]
        valid_rels = rel_coords[true_indices[1]]

        if len(valid_amps) > 0:
            ax.scatter(valid_amps, valid_rels, alpha=0.5, color="red", s=1)

        quad = machine.qubit_pairs[qubit_pair["qubit"]].qubit_control.freq_vs_flux_01_quad_term

        def detuning_to_flux(det, quad=quad):
            return 1e3 * np.sqrt(-1e6 * det / quad)

        def flux_to_detuning(flux, quad=quad):
            return -1e-6 * (flux / 1e3) ** 2 * quad

        ax2 = ax.secondary_xaxis("top", functions=(flux_to_detuning, detuning_to_flux))
        ax2.set_xlabel("Detuning [MHz]")

        ax.set_title(qubit_pair["qubit"])

    # plt.suptitle("Obtained Cphase")
    plt.show()
    node.results["figure_leak"] = grid.fig

# %%
for qp in qubit_pairs:
    # Check if current Cz gate is CZwithcompensations
    if "CZWithCompensationGate" in str(qp.gates["Cz"]):
        print(f"Warning: {qp.name} already has CZwithcompensations gate. Skipping update.")
        continue
# %%

# %% {Update_state}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qp in qubit_pairs:
                if "Cz_SNZ" in qp.gates:
                    qp.gates["Cz_SNZ"].flux_pulse_control.amplitude = min_coords[qp.name].amp_full.values.tolist()
                    qp.gates["Cz_SNZ"].flux_pulse_control.step_amplitude = (
                        min_coords[qp.name].amp_full.values.tolist()
                        * min_coords[qp.name].rel_val.values[0].tolist()
                    )
                    qp.gates["Cz_SNZ"].flux_pulse_control.length = (
                        qp.gates["Cz_unipolar"].flux_pulse_control.length + node.parameters.extra_flux_pulse_length
                    )
                    qp.gates["Cz_SNZ"].flux_pulse_control.spacing = node.parameters.spacing
                    qp.gates["Cz_SNZ"].flux_pulse_control.step_length = 1
                    qp.gates["Cz"] = f"#./Cz_SNZ"
                else:
                    if "CZWithCompensationGate" in str(qp.gates["Cz"]):
                        qp.gates["Cz_SNZ"] = CZGate(
                            flux_pulse_control=SNZPulse(
                                length=qp.gates["Cz_unipolar"].flux_pulse_control.length
                                + node.parameters.extra_flux_pulse_length,
                                amplitude=min_coords[qp.name].amp_full.values.tolist(),
                                step_amplitude=min_coords[qp.name].amp_full.values.tolist() * min_coords[qp.name].rel_val.values[0].tolist(),
                                step_length=1,
                                spacing=node.parameters.spacing,
                                id="CZ_snz_" + qp.qubit_target.name,
                            )
                        )
                        qp.gates["Cz_SNZ"].compensations = []
                        for comp in qp.gates["Cz"].compensations: 
                            qp.gates["Cz_SNZ"].compensations.append({"qubit": f'#/qubits/{comp["qubit"].name}', "shift": comp["shift"], "phase": comp["phase"]})
                        qp.gates["Cz"] = f"#./Cz_SNZ"
                    else:
                        qp.gates["Cz_SNZ"] = CZGate(
                            flux_pulse_control=SNZPulse(
                                length=qp.gates["Cz_unipolar"].flux_pulse_control.length
                                + node.parameters.extra_flux_pulse_length,
                                amplitude=min_coords[qp.name].amp_full.values.tolist(),
                                step_amplitude=min_coords[qp.name].amp_full.values.tolist()
                                * min_coords[qp.name].rel_val.values[0].tolist(),
                                step_length=1,
                                spacing=node.parameters.spacing,
                                id="CZ_snz_" + qp.qubit_target.name,
                            )
                        )
                    qp.gates["Cz"] = f"#./Cz_SNZ"

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    save_node(node)

# %%

# %%

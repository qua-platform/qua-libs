description = """"""

# %% {Imports}
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LinearSegmentedColormap
from qm import SimulationConfig
from qm.qua import *
from qualibrate import QualibrationNode, NodeParameters

from quam_config.my_quam import Quam

from qualibration_libs.legacy.macros import active_reset, readout_state, readout_state_gef, active_reset_gef, active_reset_simple
from qualibration_libs.legacy.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from qualibration_libs.legacy.lib.save_utils import fetch_results_as_xarray, load_dataset

from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit

from qualibration_libs.legacy.lib.fit import fit_oscillation, oscillation, fix_oscillation_phi_2pi
from qualibration_libs.legacy.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names

# %% {Node_parameters}
qubit_pair_indexes = [1]  # The indexes of the qubit pairs to measure
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = [f"q{i}-{i+1}" for i in qubit_pair_indexes]
    num_averages: int = 200
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None

    coupler_flux_min : float = -0.01 # relative to the coupler set point
    coupler_flux_max : float = 0.03 # relative to the coupler set point

    coupler_flux_step : float = 0.001

    qubit_flux_min : float = -0.03 # relative to the qubit pair detuning
    qubit_flux_max : float = 0.03 # relative to the qubit pair detuning
    qubit_flux_step : float = 0.001

    use_state_discrimination: bool = True
    num_frames : int = 10#20


node = QualibrationNode(
    name="18c_coupler_leakage_phase_optimization", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = Quam.load()

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

####################
# Helper functions #
####################
# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise  # 'independent' or 'joint' or 'pairwise'
# Loop parameters
fluxes_coupler = (
    np.arange(
        node.parameters.coupler_flux_min,
        node.parameters.coupler_flux_max + 0.0001,
        node.parameters.coupler_flux_step
    )
    + qubit_pairs[0].macros["Cz"].coupler_flux_pulse.amplitude
)

fluxes_qubit = np.arange(
    node.parameters.qubit_flux_min,
    node.parameters.qubit_flux_max + 0.0001,
    node.parameters.qubit_flux_step
)

fluxes_qp = {}
for qp in qubit_pairs:
    # estimate the flux shift to get the control qubit to the target qubit frequency
    fluxes_qp[qp.name] = fluxes_qubit + qp.detuning
    pulse_duration = qp.macros["Cz"].coupler_flux_pulse.length - qp.macros["Cz"].coupler_flux_pulse.zero_padding
    assert pulse_duration % 4 == 0, f"Expected pulse_duration to be a multiple of 4, got {pulse_duration}"

reset_coupler_bias = False
frames = np.arange(0, 1, 1 / node.parameters.num_frames)

with program() as CPhase_Oscillations:
    n = declare(int)
    flux_coupler = declare(float)
    flux_qubit = declare(float)
    comp_flux_qubit = declare(float)
    n_st = declare_stream()
    qua_pulse_duration = declare(int, value = int(pulse_duration/4))
    frame = declare(fixed)
    control_initial = declare(int)

    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    leakage_control = [declare(fixed) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    leakage_control_st = [declare_stream() for _ in range(num_qubit_pairs)]
    I_control = [declare(float) for _ in range(num_qubit_pairs)]
    Q_control = [declare(float) for _ in range(num_qubit_pairs)]
    I_target = [declare(float) for _ in range(num_qubit_pairs)]
    Q_target = [declare(float) for _ in range(num_qubit_pairs)]
    I_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    Q_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    I_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    Q_st_target = [declare_stream() for _ in range(num_qubit_pairs)]


    for i, qp in enumerate(qubit_pairs):
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point, qp)
        if reset_coupler_bias:
            qp.coupler.set_dc_offset(0.0)
        wait(1000)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(flux_coupler, fluxes_coupler)):
                with for_(*from_array(flux_qubit, fluxes_qp[qp.name])):
                    with for_(*from_array(frame, frames)):
                        with for_(*from_array(control_initial, [0, 1])):
                            # reset
                            if node.parameters.use_state_discrimination:
                                assign(leakage_control[i], 0)

                            if node.parameters.reset_type == "active":
                                active_reset(qp.qubit_control)
                                active_reset(qp.qubit_target)
                            else:
                                wait(qp.qubit_control.thermalization_time * u.ns)
                                wait(qp.qubit_target.thermalization_time * u.ns)

                            qp.align()

                            if "coupler_qubit_crosstalk" in qp.extras:
                                assign(comp_flux_qubit, flux_qubit + qp.extras["coupler_qubit_crosstalk"] * flux_coupler )
                            else:
                                assign(comp_flux_qubit, flux_qubit)
                            # setting both qubits ot the initial state
                            with if_(control_initial == 1, unsafe = True):
                                qp.qubit_control.xy.play("x180")
                            qp.qubit_target.xy.play("x90")
                            qp.align()

                            qp.qubit_control.z.play("const", amplitude_scale = comp_flux_qubit / qp.qubit_control.z.operations["const"].amplitude, duration = qua_pulse_duration)
                            qp.coupler.play("const", amplitude_scale = flux_coupler / qp.coupler.operations["const"].amplitude, duration = qua_pulse_duration)

                            qp.align()
                            frame_rotation_2pi(frame, qp.qubit_target.xy.name)
                            qp.qubit_target.xy.play("x90")
                            qp.align()
                            # readout
                            if node.parameters.use_state_discrimination:
                                readout_state_gef(qp.qubit_control, state_control[i])
                                readout_state(qp.qubit_target, state_target[i])
                                assign(leakage_control[i], Cast.to_fixed( state_control[i] == 2))
                                save(state_control[i], state_st_control[i])
                                save(state_target[i], state_st_target[i])
                                save(leakage_control[i], leakage_control_st[i])

                            else:
                                qp.qubit_control.resonator.measure("readout", qua_vars=(I_control[i], Q_control[i]))
                                qp.qubit_target.resonator.measure("readout", qua_vars=(I_target[i], Q_target[i]))
                                save(I_control[i], I_st_control[i])
                                save(Q_control[i], Q_st_control[i])
                                save(I_target[i], I_st_target[i])
                                save(Q_target[i], Q_st_target[i])

        align(*([qp.qubit_control.xy.name for qp in qubit_pairs] +
                [qp.qubit_control.z.name for qp in qubit_pairs] +
                [qp.qubit_control.resonator.name for qp in qubit_pairs]))

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            if node.parameters.use_state_discrimination:
                state_st_control[i].buffer(2).buffer(len(frames)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_control{i + 1}")
                state_st_target[i].buffer(2).buffer(len(frames)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_target{i + 1}")
                leakage_control_st[i].buffer(2).buffer(len(frames)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_control_f{i + 1}")
            else:
                I_st_control[i].buffer(2).buffer(len(frames)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"I_control{i + 1}")
                Q_st_control[i].buffer(2).buffer(len(frames)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"Q_control{i + 1}")
                I_st_target[i].buffer(2).buffer(len(frames)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"I_target{i + 1}")
                Q_st_target[i].buffer(2).buffer(len(frames)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"Q_target{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, CPhase_Oscillations, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"control_ax": [0, 1],  "frame": frames, "flux_qubit": fluxes_qubit, "flux_coupler": fluxes_coupler})
        flux_qubit_full = np.array([fluxes_qp[qp.name] for qp in qubit_pairs])
        ds = ds.assign_coords({"flux_qubit_full": (["qubit", "flux_qubit"], flux_qubit_full)})
    else:
        ds, machine, _, qubit_pairs = load_dataset(node.parameters.load_data_id)
        node = node.load_from_id(node.parameters.load_data_id)

    node.results = {"ds": ds}
# %%
detuning_mode = "quadratic" # "cosine" or "quadratic"
if not node.parameters.simulate:
    if reset_coupler_bias:
        flux_coupler_full = np.array([fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs])
    else:
        flux_coupler_full = np.array([fluxes_coupler for qp in qubit_pairs])
    if detuning_mode == "quadratic":
        detuning = np.array([-fluxes_qp[qp.name] ** 2 * qp.qubit_control.freq_vs_flux_01_quad_term  for qp in qubit_pairs])
    elif detuning_mode == "cosine":
        detuning = np.array([oscillation(fluxes_qubit, qp.qubit_control.extras['a'], qp.qubit_control.extras['f'], qp.qubit_control.extras['phi'], qp.qubit_control.extras['offset']) for qp in qubit_pairs])
    ds = ds.assign_coords({"flux_coupler_full": (["qubit", "flux_coupler"], flux_coupler_full)})
    ds = ds.assign_coords({"detuning": (["qubit", "flux_qubit"], detuning)})
    node.results = {"ds": ds}

    fit_data = fit_oscillation(ds.state_target, "frame")
    phase = fix_oscillation_phi_2pi(fit_data)
    phase_diff = phase.diff(dim="control_ax")

    leak = ds.state_control_f.mean(dim = "frame").sel(control_ax = 1)
    # (((phase_diff+0.5 )% 1 -0.5)*360).plot()

    mask = (np.abs((np.abs(phase_diff)-0.5))<0.02)
    leak_mask = leak * mask + (1 - mask)
    min_value = leak_mask.min(dim=["flux_qubit", "flux_coupler","control_ax"])
    min_coords = {}
    for q in phase_diff.qubit.values:

        min_coords[q] = leak_mask.sel(qubit=q).where(leak_mask.sel(qubit=q) == min_value.sel(qubit=q), drop=True)[0]

    node.results["results"] = {}
    for q in min_coords.keys():
        node.results["results"][q] = {}
        node.results["results"][q]["flux_coupler_Cz"] = float(min_coords[q].flux_coupler_full.values)
        node.results["results"][q]["flux_qubit_Cz"] = float(min_coords[q].flux_qubit_full.values)

# %% {What if we mask leakage first?}
# mask = leak < 0.27
# plt.figure()
# (((phase_diff + 0.5) % 1 - 0.5) * 360).plot(cmap=LinearSegmentedColormap.from_list("", ["red", "white", "red"]))
# # (abs((((phase_diff + 0.5) % 1 - 0.5) * 360)) * mask).plot(cmap=LinearSegmentedColormap.from_list("", ["white", "red"]))
# data = abs((((phase_diff + 1.5) % 1 - 0.5) * 360)) * mask
# flat_index = data.argmax()
# indices = np.unravel_index(flat_index, data.shape)
# plt.scatter(data[*indices].flux_qubit, data[*indices].flux_coupler, marker='x', color='k')
# plt.show()
# leak.plot()
# plt.scatter(data[*indices].flux_qubit, data[*indices].flux_coupler, marker='x', color='k')
# plt.show()

# %% {Plotting}
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qp in grid_iter(grid):
        values_to_plot = ds.state_control_f.mean(dim = "frame").sel(control_ax = 1).sel(qubit = qp['qubit'])
        values_to_plot.assign_coords({"flux_qubit_mV": 1e3*values_to_plot.flux_qubit_full, "flux_coupler_mV": 1e3*values_to_plot.flux_coupler_full}).plot(
            ax = ax, cmap=LinearSegmentedColormap.from_list("", ["white", "black"]), x = 'flux_qubit_mV', y = 'flux_coupler_mV'
        )
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
        ax.axhline(1e3*node.results["results"][qp["qubit"]]["flux_coupler_Cz"], color = 'red', lw = 0.5, ls = '--')
        ax.axvline(1e3*node.results["results"][qp["qubit"]]["flux_qubit_Cz"], color = 'red', lw =0.5, ls = '--')

        mask_q = mask.loc[qp['qubit']].sel(control_ax=1)

        # Create 2D arrays of coordinates
        amp_coords = 1e3 * mask_q.flux_qubit_full.values
        rel_coords = 1e3 * mask_q.flux_coupler_full.values
        # Get the indices where mask is True
        true_indices = np.where(mask_q.values)

        # Get the corresponding coordinate values
        valid_amps = amp_coords[true_indices[1]]
        valid_rels = rel_coords[true_indices[0]]

        if len(valid_amps) > 0:
            ax.scatter(valid_amps, valid_rels, alpha=0.5, color="red", s=1)
        # Create a secondary x-axis for detuning
        base_detuning = -machine.qubit_pairs[qp['qubit']].detuning **2 * machine.qubit_pairs[qp['qubit']].qubit_control.freq_vs_flux_01_quad_term
        flux_qubit_data = ds.sel(qubit=qp['qubit']).flux_qubit_full.values*1e3
        detuning_data = (ds.sel(qubit=qp['qubit']).detuning.values -base_detuning) * 1e-6

        def flux_to_detuning(x):
            return np.interp(x, flux_qubit_data, detuning_data)

        def detuning_to_flux(y):
            return np.interp(y, detuning_data, flux_qubit_data)

        sec_ax = ax.secondary_xaxis('top', functions=(flux_to_detuning, detuning_to_flux))
        sec_ax.set_xlabel('Detuning [MHz]')
        ax.set_xlabel('Qubit flux shift [V]')
        ax.set_ylabel('Coupler flux [V]')
    grid.fig.suptitle('Leakage')
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
    node.results['figure_leakage'] = grid.fig

    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qp in grid_iter(grid):

        values_to_plot = (((phase_diff+0.5 )% 1 -0.5)*360).sel(qubit = qp['qubit'])

        values_to_plot.assign_coords({"flux_qubit_mV": 1e3*values_to_plot.flux_qubit_full, "flux_coupler_mV": 1e3*values_to_plot.flux_coupler_full}).plot(
            ax = ax, cmap='twilight_shifted', x = 'flux_qubit_mV', y = 'flux_coupler_mV'
        )
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
        ax.axhline(1e3*node.results["results"][qp["qubit"]]["flux_coupler_Cz"], color = 'k', lw = 0.5, ls = '--')
        ax.axvline(1e3*node.results["results"][qp["qubit"]]["flux_qubit_Cz"], color = 'k', lw =0.5, ls = '--')
        # Create a secondary x-axis for detuning
        base_detuning = -machine.qubit_pairs[qp['qubit']].detuning **2 * machine.qubit_pairs[qp['qubit']].qubit_control.freq_vs_flux_01_quad_term
        flux_qubit_data = ds.sel(qubit=qp['qubit']).flux_qubit_full.values*1e3
        detuning_data = (ds.sel(qubit=qp['qubit']).detuning.values -base_detuning) * 1e-6

        def flux_to_detuning(x):
            return np.interp(x, flux_qubit_data, detuning_data)

        def detuning_to_flux(y):
            return np.interp(y, detuning_data, flux_qubit_data)

        sec_ax = ax.secondary_xaxis('top', functions=(flux_to_detuning, detuning_to_flux))
        sec_ax.set_xlabel('Detuning [MHz]')
        ax.set_xlabel('Qubit flux shift [V]')
        ax.set_ylabel('Coupler flux [V]')
    grid.fig.suptitle('Conditional phase $\phi$')
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
    node.results['figure_target'] = grid.fig


# %% {Update_state}
if not node.parameters.simulate:
    if not node.parameters.simulate:
        with (node.record_state_updates()):
            for qp in qubit_pairs:
                flux_coupler = node.results["results"][qp.name]["flux_coupler_Cz"]
                qp.macros["Cz"].coupler_flux_pulse.amplitude = flux_coupler
                if "coupler_qubit_crosstalk" in qp.extras:
                    qp.macros["Cz"].flux_pulse_control.amplitude = node.results["results"][qp.name]["flux_qubit_Cz"] + qp.extras["coupler_qubit_crosstalk"] * flux_coupler
                else:
                    qp.macros["Cz"].flux_pulse_control.amplitude = node.results["results"][qp.name]["flux_qubit_Cz"]

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()

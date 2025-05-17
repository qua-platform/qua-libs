# %%
"""
Calibration of Single-Qubit Phase Accumulation During CZ Gate Execution

This sequence calibrates the residual phase each qubit accumulates during the execution of the flux pulse enabling the CZ gate. The calibration process involves:

1. Applying a CZ gate flux pulse with varying amplitudes
2. Measuring the resulting phase of each qubit independently
3. Calculating the phase accumulation as a function of pulse amplitude

For each amplitude, we measure:
1. The phase of the control qubit
2. The phase of the target qubit

The calibration process involves:
1. Preparing each qubit in a superposition state
2. Applying the CZ gate flux pulse with varying amplitudes
3. Measuring the resulting phase of each qubit using a Ramsey-like sequence

The outcome of this calibration will be used to:
1. Determine the phase correction needed for each qubit after a CZ gate
2. Understand how the phase accumulation varies with pulse amplitude

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair
- Calibrated readout for both qubits
- Initial estimate of the CZ gate flux pulse parameters

Outcomes:
- Phase accumulation curves for both qubits as a function of flux pulse amplitude
- Optimal phase correction values for each qubit at the nominal CZ gate amplitude
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef, active_reset_gef
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset, save_node
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
from quam_libs.components.gates.two_qubit_gates import CZGate
from quam_libs.lib.pulses import FluxPulse

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = None
    num_averages: int = 1000
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    num_frames: int = 21
    load_data_id: Optional[int] = None
    plot_raw : bool = False
    measure_leak : bool = False


node = QualibrationNode(
    name="33c_Cz_comp_phase_calibration_frame", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

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


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

# Loop parameters
frames = np.arange(0, 1, 1/node.parameters.num_frames+0.01)

with program() as CPhase_Oscillations:
    amp = declare(fixed)   
    frame = declare(fixed)
    control_initial = declare(int)
    n = declare(int)
    n_st = declare_stream()
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    
    for i, qp in enumerate(qubit_pairs):
        # Bring the active qubits to the minimum frequency point
        machine.

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)         
            with for_(*from_array(frame, frames)):
                for qubit, state_q, state_st in [(qp.qubit_control, state_control[i], state_st_control[i]), (qp.qubit_target, state_target[i], state_st_target[i])]:
                    # reset
                    if node.parameters.reset_type == "active":
                            active_reset(qp.qubit_control)
                            qp.align()
                            active_reset(qp.qubit_target)
                            qp.align()
                    else:
                        wait(qp.qubit_control.thermalization_time * u.ns)
                    qp.align()
                    # setting both qubits ot the initial state
                    qubit.xy.play("x90")
                    qp.align()

                    #play the CZ gate
                    qp.gates['Cz'].execute()
                    
                    #rotate the frame
                    frame_rotation_2pi(frame, qubit.xy.name)
                    
                    # return the target qubit before measurement
                    qubit.xy.play("x90")              
                    
                    qp.align()
                    readout_state(qubit, state_q)
                    save(state_q, state_st)
                    qp.align()

        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_st_control[i].buffer(len(frames)).average().save(f"state_control{i + 1}")
            state_st_target[i].buffer(len(frames)).average().save(f"state_target{i + 1}")

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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"frame": frames})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)

        
    node.results = {"ds": ds}
    

# %% Analysis
if not node.parameters.simulate:

    fit_data_target = fit_oscillation(ds.state_target, "frame")
    fit_data_control = fit_oscillation(ds.state_control, "frame")

    ds = ds.assign({'fitted_target': oscillation(ds.frame,
                                        fit_data_target.sel(fit_vals="a"),
                                        fit_data_target.sel(fit_vals="f"),
                                        fit_data_target.sel(fit_vals="phi"),
                                        fit_data_target.sel(fit_vals="offset")),
                    'fitted_control': oscillation(ds.frame,
                                        fit_data_control.sel(fit_vals="a"),
                                        fit_data_control.sel(fit_vals="f"),
                                        fit_data_control.sel(fit_vals="phi"),
                                        fit_data_control.sel(fit_vals="offset"))
                    })

phases_target = {}
phases_control = {}
for qp in qubit_pairs:
    phase_target = float(fix_oscillation_phi_2pi(fit_data_target.sel(qubit=qp.name))) % 1
    phase_control = float(fix_oscillation_phi_2pi(fit_data_control.sel(qubit=qp.name))) % 1
    A_control = float(fit_data_control.sel(qubit=qp.name, fit_vals="a"))
    A_target = float(fit_data_target.sel(qubit=qp.name, fit_vals="a"))
    offset_control = float(fit_data_control.sel(qubit=qp.name, fit_vals="offset"))
    offset_target = float(fit_data_target.sel(qubit=qp.name, fit_vals="offset"))
    
    phases_target[qp.name] = phase_target
    phases_control[qp.name] = phase_control
    
    print(f'measured phase offsets for {qp.name } are target: {phase_target:.3f}, control: {phase_control:.3f}')
    
# %%
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        # ds.to_array().sel(qubit= qubit_pair['qubit']).plot.line(ax =ax, hue="variable", ylim=[0, 1])
        ds.state_target.sel(qubit= qubit_pair['qubit']).plot(ax =ax, marker = '.', lw = 0, label = 'data target',color = 'C0')
        ds.fitted_target.sel(qubit= qubit_pair['qubit']).plot(ax =ax, lw = 0.5, label = 'fit target',color = 'C0')
        ds.state_control.sel(qubit= qubit_pair['qubit']).plot(ax =ax, marker = '.', lw = 0, label = 'data control',color = 'C1')
        ds.fitted_control.sel(qubit= qubit_pair['qubit']).plot(ax =ax, lw = 0.5, label = 'fit control',color = 'C1')
        ax.set_title(qubit_pair['qubit'])
        ax.axvline(x = 1-phases_target[qubit_pair['qubit']], color = 'C0', linestyle = '--')
        ax.axvline(x = 1-phases_control[qubit_pair['qubit']], color = 'C1', linestyle = '--')
        ax.legend()
    plt.suptitle('Cz single qubit phase calibration')
    plt.tight_layout()
    plt.show()
    node.results["figure_phase"] = grid.fig


# %%
phase_target = {}
phase_control = {}
A_control = {}
A_target = {}
offset_control = {}
offset_target = {}
for qp in qubit_pairs:
    phase_target[qp.name] = float(fix_oscillation_phi_2pi(fit_data_target.sel(qubit=qp.name))) % 1
    phase_control[qp.name] = float(fix_oscillation_phi_2pi(fit_data_control.sel(qubit=qp.name))) % 1
    A_control[qp.name] = float(fit_data_control.sel(qubit=qp.name, fit_vals="a"))
    A_target[qp.name] = float(fit_data_target.sel(qubit=qp.name, fit_vals="a"))
    offset_control[qp.name] = float(fit_data_control.sel(qubit=qp.name, fit_vals="offset"))
    offset_target[qp.name] = float(fit_data_target.sel(qubit=qp.name, fit_vals="offset"))
    
    # qp.gates['Cz'].phase_shift_control -= (phase_control / params['cz_num'])
    # qp.gates['Cz'].phase_shift_control = qp.gates['Cz'].phase_shift_control  % (1.0)
    # qp.gates['Cz'].phase_shift_target -= (phase_target/ params['cz_num'])
    # qp.gates['Cz'].phase_shift_target = qp.gates['Cz'].phase_shift_target  % (1.0)
    # qp.gates['Cz'].extras['A_control'] = A_control
    # qp.gates['Cz'].extras['A_target'] = A_target
    # qp.gates['Cz'].extras['offset_control'] = offset_control
    # qp.gates['Cz'].extras['offset_target'] = offset_target
    
    

# %% {Update_state}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qp in qubit_pairs:
                qp.gates['Cz'].phase_shift_control -= (phase_control[qp.name] / 1.0)
                qp.gates['Cz'].phase_shift_control = qp.gates['Cz'].phase_shift_control  % (1.0)
                qp.gates['Cz'].phase_shift_target -= (phase_target[qp.name]/ 1.0)
                qp.gates['Cz'].phase_shift_target = qp.gates['Cz'].phase_shift_target  % (1.0)
                
# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    save_node(node)
        
# %%
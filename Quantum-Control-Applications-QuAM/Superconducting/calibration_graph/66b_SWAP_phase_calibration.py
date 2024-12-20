# %%
"""
Two-Qubit Readout Confusion Matrix Measurement

This sequence measures the readout error when simultaneously measuring the state of two qubits. The process involves:

1. Preparing the two qubits in all possible combinations of computational basis states (|00⟩, |01⟩, |10⟩, |11⟩)
2. Performing simultaneous readout on both qubits
3. Calculating the confusion matrix based on the measurement results

For each prepared state, we measure:
1. The readout result of the first qubit
2. The readout result of the second qubit

The measurement process involves:
1. Initializing both qubits to the ground state
2. Applying single-qubit gates to prepare the desired input state
3. Performing simultaneous readout on both qubits
4. Repeating the process multiple times to gather statistics

The outcome of this measurement will be used to:
1. Quantify the readout fidelity for two-qubit states
2. Identify and characterize crosstalk effects in the readout process
3. Provide data for readout error mitigation in two-qubit experiments

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair
- Calibrated readout for both qubits

Outcomes:
- 4x4 confusion matrix representing the probabilities of measuring each two-qubit state given a prepared input state
- Readout fidelity metrics for simultaneous two-qubit measurement
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef, active_reset_gef
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
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
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from scipy.optimize import curve_fit
from scipy.fft import fft
import xarray as xr
from quam_libs.lib.fit import oscillation, fit_oscillation

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["q0-q1"]
    num_averages: int = 1000
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "pairwise"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    phase_steps : int = 21
    use_state_discrimination: bool = True
    
node = QualibrationNode(
    name="66b_SWAP_phase_calibration", parameters=Parameters()
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

flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise  # 'independent' or 'joint' or 'pairwise'
# Loop parameters
phases = np.linspace(0, 1.0, node.parameters.phase_steps)


# %%
with program() as SWAP_amp_calibration:
    
    n = declare(int)
    phase = declare(float)
    n_st = declare_stream()
    count = declare(int)
    if node.parameters.use_state_discrimination:
        state_control = [declare(int) for _ in range(num_qubit_pairs)]
        state_target = [declare(int) for _ in range(num_qubit_pairs)]
        state = [declare(int) for _ in range(num_qubit_pairs)]
        state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st = [declare_stream() for _ in range(num_qubit_pairs)]
    else:
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
        wait(1000)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st) 
            for temp in range(1):
                with for_(*from_array(phase, phases)):
                    # reset
                    if node.parameters.reset_type == "active":
                            active_reset(qp.qubit_control)
                            active_reset(qp.qubit_target)
                            qp.align()
                    else:
                        wait(qp.qubit_control.thermalization_time * u.ns)
                        wait(qp.qubit_target.thermalization_time * u.ns)
                    align()
                
                    # setting both qubits ot the initial state
                    qp.qubit_target.xy.play("x90")
                                    
                    qp.align()
                    qp.gates["SWAP_Coupler"].execute()
                    qp.align()
                    frame_rotation_2pi(phase, qp.qubit_control.xy.name)
                    qp.align()
                    qp.qubit_control.xy.play("x90")
                    qp.align()
                    # readout
                    if node.parameters.use_state_discrimination:
                        readout_state(qp.qubit_control, state_control[i])
                        readout_state(qp.qubit_target, state_target[i])
                        assign(state[i], state_control[i]*2 + state_target[i])
                        save(state_control[i], state_st_control[i])
                        save(state_target[i], state_st_target[i])
                        save(state[i], state_st[i])
                    else:
                        qp.qubit_control.resonator.measure("readout", qua_vars=(I_control[i], Q_control[i]))
                        qp.qubit_target.resonator.measure("readout", qua_vars=(I_target[i], Q_target[i]))
                        save(I_control[i], I_st_control[i])
                        save(Q_control[i], Q_st_control[i])
                        save(I_target[i], I_st_target[i])
                        save(Q_target[i], Q_st_target[i])
        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            if node.parameters.use_state_discrimination:
                state_st_control[i].buffer(len(phases)).average().save(f"state_control{i + 1}")
                state_st_target[i].buffer(len(phases)).average().save(f"state_target{i + 1}")
                state_st[i].buffer(len(phases)).average().save(f"state{i + 1}")
            else:
                I_st_control[i].buffer(len(phases)).average().save(f"I_control{i + 1}")
                Q_st_control[i].buffer(len(phases)).average().save(f"Q_control{i + 1}")
                I_st_target[i].buffer(len(phases)).average().save(f"I_target{i + 1}")
                Q_st_target[i].buffer(len(phases)).average().save(f"Q_target{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, SWAP_amp_calibration, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(SWAP_amp_calibration)

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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, { "phase": phases})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}

# %%
ds.state_control.plot(label = "control")
ds.state_target.plot(label = "target")
plt.legend()
plt.xlabel('Phase')
plt.ylabel('Population')
# %%
# %% {Data_analysis}
if not node.parameters.simulate:
    
    node.results["results"] = {}
    if N_pi == 1:
        # Fit the power Rabi oscillations
        fit_control = fit_oscillation(ds.data_var_control, "amp")
        fit_evals = oscillation(
            ds.amp,
            fit_control.sel(fit_vals="a"),
            fit_control.sel(fit_vals="f"),
            fit_control.sel(fit_vals="phi"),
            fit_control.sel(fit_vals="offset"),
        )
        ds = ds.assign({"fit_amp_control" : fit_evals})
        
        fit_target = fit_oscillation(ds.data_var_target, "amp")
        fit_evals = oscillation(
            ds.amp,
            fit_target.sel(fit_vals="a"),
            fit_target.sel(fit_vals="f"),
            fit_target.sel(fit_vals="phi"),
            fit_target.sel(fit_vals="offset"),
        )
        ds = ds.assign({"fit_amp_target" : fit_evals})
        
    # Save fitting results
        for qp in qubit_pairs:
            node.results["results"][qp.name] = {}
            f_fit = fit_control.sel(qubit = qp.name).sel(fit_vals="f")
            phi_fit = fit_control.sel(qubit = qp.name).sel(fit_vals="phi")
            phi_fit = phi_fit - np.pi * (phi_fit > np.pi / 2)
            factor = float(1.0 * (np.pi - phi_fit) / (2 * np.pi * f_fit))
            new_pi_amp = qp.gates["SWAP_Coupler"].flux_pulse_control.amplitude * factor
            if new_pi_amp < 0.3:  # TODO: 1 for OPX1000 MW
                print(f"amplitude for Pi pulse is modified by a factor of {factor:.2f}")
                print(
                    f"new amplitude is {1e3 * new_pi_amp:.2f} mV \n"
                )  # TODO: 1 for OPX1000 MW
                node.results["results"][qp.name]["SWAP_amplitude"] = float(new_pi_amp)
            else:
                print(f"Fitted amplitude too high, new amplitude is 300 mV \n")
                node.results["results"][qp.name]["SWAP_amplitude"] = 0.3  # TODO: 1 for OPX1000 MW        
    else:
        data_max_idx = (ds.data_var_target-ds.data_var_control).mean(dim = "N").argmax(dim="amp")
        for qp in qubit_pairs:
            node.results["results"][qp.name] = {}
            node.results["results"][qp.name]["SWAP_amplitude"] = qp.gates["SWAP_Coupler"].flux_pulse_control.amplitude * float(ds.amp[data_max_idx.sel(qubit = qp.name)].values)

# %% {Plotting}
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit in grid_iter(grid):
        if N_pi == 1:
            ds.assign_coords(amp_mV=ds.amp_full * 1e3).loc[qubit].data_var_control.plot(
                ax=ax, x="amp_mV"
            )
            # ax.plot(ds.amp_full.loc[qubit] * 1e3, 1e3 * fit_evals.loc[qubit][0])
            # ax.set_ylabel("Trans. amp. I [mV]")
        elif N_pi > 1:
            ds.assign_coords(amp_mV=ds.amp_full * 1e3).loc[qubit].data_var_control.plot(
                ax=ax, x="amp_mV", y="N"
            )
            ax.axvline(1e3*node.results["results"][qubit['qubit']]["SWAP_amplitude"], color="r", lw = 0.5, ls = '--')
            ax.set_ylabel("num. of pulses")
        ax.set_xlabel("Amplitude [mV]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("SWAP amplitude calibration, control")
    plt.tight_layout()
    plt.show()
    node.results["figure_control"] = grid.fig
    
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit in grid_iter(grid):
        if N_pi == 1:
            ds.assign_coords(amp_mV=ds.amp_full * 1e3).loc[qubit].data_var_target.plot(
                ax=ax, x="amp_mV"
            )
            # ax.plot(ds.amp_full.loc[qubit] * 1e3, 1e3 * fit_evals.loc[qubit][0])
            # ax.set_ylabel("Trans. amp. I [mV]")
        elif N_pi > 1:
            ds.assign_coords(amp_mV=ds.amp_full * 1e3).loc[qubit].data_var_target.plot(
                ax=ax, x="amp_mV", y="N"
            )
            ax.axvline(1e3*node.results["results"][qubit['qubit']]["SWAP_amplitude"], color="r", lw = 0.5, ls = '--')
            ax.set_ylabel("num. of pulses")
        ax.set_xlabel("Amplitude [mV]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("SWAP amplitude calibration, target")
    plt.tight_layout()
    plt.show()
    node.results["figure_target"] = grid.fig        



# %% {Update_state}
if not node.parameters.simulate:
    with node.record_state_updates():
        for qp in qubit_pairs:          
            qp.gates['SWAP_Coupler'].flux_pulse_control.amplitude = node.results["results"][qp.name]["SWAP_amplitude"]

# %% {Save_results}
if not node.parameters.simulate:    
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%

# %%
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["q1"]
    num_averages: int = 600
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 200000
    num_time_steps: int = 200
    flux_point_joint_or_independent_or_arbitrary: Literal['joint', 'independent', 'arbitrary'] = "joint"    
    simulate: bool = False
    timeout: int = 100
    use_state_discrimination: bool = True
    reset_type: Literal['active', 'thermal'] = "thermal"
    drive_pulse_name: str = "x180_Square"

node = QualibrationNode(
    name="07_spinecho",
    parameters=Parameters()
)

scale = 1/1000


from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, multiplexed_readout, node_save, active_reset, readout_state

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_decay_exp, decay_exp





# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == '':
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
idle_times = np.unique(
    np.geomspace(
        node.parameters.min_wait_time_in_ns, node.parameters.max_wait_time_in_ns, node.parameters.num_time_steps
    )
    // 4
).astype(int)


flux_point = node.parameters.flux_point_joint_or_independent_or_arbitrary  # 'independent' or 'joint'
if flux_point == "arbitrary":
    detunings = {q.name : q.arbitrary_intermediate_frequency for q in qubits}
    arb_flux_bias_offset = {q.name: q.z.arbitrary_offset for q in qubits}
else:
    arb_flux_bias_offset = {q.name: 0.0 for q in qubits}
    detunings = {q.name: 0.0 for q in qubits}

with program() as t1:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    t = declare(int)  # QUA variable for the idle time
    if node.parameters.use_state_discrimination:
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            qubit.z.to_independent_idle()
        elif flux_point == "joint" or "arbitrary":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()

        # Wait for the flux bias to settle
        for qb in qubits:
            wait(1000, qb.z.name)

        align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_each_(t, idle_times):
                if node.parameters.reset_type == "active":
                    active_reset(qubit)
                else:
                    qubit.resonator.wait(qubit.thermalization_time * u.ns)
                    qubit.align()
                
                    
                qubit.xy.play("-y90")
                qubit.align()
                # qubit.z.wait(20)
                # qubit.z.play("const", amplitude_scale=arb_flux_bias_offset[qubit.name]/qubit.z.operations["const"].amplitude, duration=t)
                # qubit.z.wait(20)
                
                qubit.xy.play(node.parameters.drive_pulse_name, amplitude_scale=scale, duration = t)
                qubit.align()
                qubit.xy.play("-y90")
                qubit.align()
                
                # Measure the state of the resonators
                if node.parameters.use_state_discrimination:
                    readout_state(qubit, state[i])
                    save(state[i], state_st[i])
                else:
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])

        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            if node.parameters.use_state_discrimination:
                state_st[i].buffer(len(idle_times)).average().save(f"state{i + 1}")
            else:
                I_st[i].buffer(len(idle_times)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(idle_times)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, t1, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(t1)
        # Get results from QUA program
        for i in range(num_qubits):
            print(f"Fetching results for qubit {qubits[i].name}")
            data_list = ["n"]
            results = fetching_tool(job, data_list, mode="live")
            while results.is_processing():
            # Fetch results
                fetched_data = results.fetch_all()
                n = fetched_data[0]

                progress_counter(n, n_avg, start_time=results.start_time)


# %%
# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"idle_time": idle_times})

    ds = ds.assign_coords(idle_time=4*ds.idle_time/1e3)  # convert to usec
    ds.idle_time.attrs = {'long_name': 'idle time', 'units': 'usec'}

# %% {Data_analysis}

if not node.parameters.simulate:
    if node.parameters.use_state_discrimination:
        fit_data = fit_decay_exp(ds.state, 'idle_time')
    else:
        fit_data = fit_decay_exp(ds.I, 'idle_time')
    fit_data.attrs = {'long_name' : 'time', 'units' : 'usec'}
    fitted =  decay_exp(ds.idle_time,
                                                    fit_data.sel(
                                                        fit_vals="a"),
                                                    fit_data.sel(
                                                        fit_vals="offset"),
                                                    fit_data.sel(fit_vals="decay"))


    decay = fit_data.sel(fit_vals = 'decay')
    decay.attrs = {'long_name' : 'decay', 'units' : 'nSec'}

    decay_res = fit_data.sel(fit_vals = 'decay_decay')
    decay_res.attrs = {'long_name' : 'decay', 'units' : 'nSec'}
    
    tau = -1/fit_data.sel(fit_vals='decay')
    tau.attrs = {'long_name' : 'T2*', 'units' : 'uSec'}

    tau_error = -tau * (np.sqrt(decay_res)/decay)
    tau_error.attrs = {'long_name' : 'T2* error', 'units' : 'uSec'}

    node.results = {"ds": ds}
# %% {Plotting}
if not node.parameters.simulate:
    
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            ds.sel(qubit = qubit['qubit']).state.plot(ax = ax)
            
            ax.set_ylabel('State')
        else:
            ds.sel(qubit = qubit['qubit']).I.plot(ax = ax)
            ax.set_ylabel('I (V)')
        ax.plot(ds.idle_time, fitted.loc[qubit], 'r--')
        ax.set_title(qubit['qubit'])
        ax.set_xlabel('Idle_time (uS)')
        ax.text(0.1, 0.9, f'T2e = {tau.sel(qubit = qubit["qubit"]).values:.1f} + {tau_error.sel(qubit = qubit["qubit"]).values:.1f} usec', transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    grid.fig.suptitle('T2 echo')
    plt.tight_layout()
    plt.show()
    node.results['figure_raw'] = grid.fig

# %% {Save_results}
if not node.parameters.simulate:
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%

# %%

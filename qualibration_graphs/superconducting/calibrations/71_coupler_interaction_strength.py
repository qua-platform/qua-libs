# %% {Imports}
import math

from qualibrate import QualibrationNode, NodeParameters

from qualibration_libs.legacy.lib.pulses import FluxPulse
from quam_config.my_quam import Quam

from qualibration_libs.legacy.macros import active_reset, readout_state, active_reset_simple
from qualibration_libs.legacy.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualibration_libs.legacy.lib.fit import extract_dominant_frequencies
from qualibration_libs.legacy.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_builder.architecture.superconducting.custom_gates.cz import CZGate

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
qubit_pair_indexes = [1] 
class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] =  ["q%s-%s"%(i,i+1) for i in qubit_pair_indexes]
    num_averages: int = 880
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None

    # coupler_q1_q2:
    coupler_flux_min : float = -0.1
    coupler_flux_max : float = 0.1
    coupler_flux_step : float = 0.005

    idle_time_min : int = 16
    idle_time_max : int = 500
    idle_time_step : int = 4

    gate_time_in_ns: int = 50

    use_state_discrimination: bool = True
    

node = QualibrationNode(
    name="71_coupler_interaction_strength_calibration", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), \
    "If simulate is True, load_data_id must be None, and vice versa."

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

num_qubit_pairs = len(qubit_pairs)

for qp in qubit_pairs:
    print("control: %s, target: %s" %(qp.qubit_control.name, qp.qubit_target.name))

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise  # 'independent' or 'joint' or 'pairwise'
# Loop parameters
fluxes_coupler = np.arange(node.parameters.coupler_flux_min, node.parameters.coupler_flux_max, node.parameters.coupler_flux_step)
idle_times = np.arange(node.parameters.idle_time_min, node.parameters.idle_time_max, node.parameters.idle_time_step) // 4

with program() as CPhase_Oscillations:
    n = declare(int)
    flux_coupler = declare(float)
    comp_flux_qubit = declare(float)
    idle_time = declare(int)
    n_st = declare_stream()
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
        machine.set_all_fluxes("joint", qp)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)         
            with for_(*from_array(flux_coupler, fluxes_coupler)):
                with for_(*from_array(idle_time, idle_times)):
                    # reset
                    if node.parameters.reset_type == "active":
                            active_reset_simple(qp.qubit_control)
                            active_reset_simple(qp.qubit_target)
                            qp.align()
                    else:
                        wait(qp.qubit_control.thermalization_time * u.ns)
                        wait(qp.qubit_target.thermalization_time * u.ns)
                    

                    if "coupler_qubit_crosstalk" in qp.extras:
                        assign(comp_flux_qubit, qp.detuning  +  qp.extras["coupler_qubit_crosstalk"] * flux_coupler )
                    else:
                        assign(comp_flux_qubit, qp.detuning)
                    qp.align()
                    
                    # setting both qubits ot the initial state
                    qp.qubit_control.xy.play("x180")
                    qp.qubit_control.z.wait(qp.qubit_control.xy.operations["x180"].length >> 2)
                    qp.coupler.wait(qp.qubit_control.xy.operations["x180"].length >> 2)             
                    
                    # Play the flux pulse on the qubit control and coupler
                    qp.qubit_control.z.play("const", amplitude_scale = comp_flux_qubit / qp.qubit_control.z.operations["const"].amplitude, duration = idle_time)
                    qp.coupler.play("const", amplitude_scale = flux_coupler / qp.coupler.operations["const"].amplitude, duration = idle_time)
                    
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
                state_st_control[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"state_control{i + 1}")
                state_st_target[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"state_target{i + 1}")
                state_st[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"state{i + 1}")
            else:
                I_st_control[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"I_control{i + 1}")
                Q_st_control[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"Q_control{i + 1}")
                I_st_target[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"I_target{i + 1}")
                Q_st_target[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"Q_target{i + 1}")

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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {  "idle_time": idle_times, "flux_coupler": fluxes_coupler})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}

# %%
if not node.parameters.simulate:
    ds = ds.assign_coords(idle_time = ds.idle_time * 4)
    if node.parameters.use_state_discrimination:
        ds = ds.assign({"res_sum" : ds.state_control - ds.state_target})
    else:
        ds = ds.assign({"res_sum" : ds.I_control - ds.I_target})
    flux_coupler_full = np.array([fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs])
    ds = ds.assign_coords({"flux_coupler_full": (["qubit", "flux_coupler"], flux_coupler_full)})    
# %%
if not node.parameters.simulate:
    # Add the dominant frequencies to the dataset
    ds['dominant_frequency'] = extract_dominant_frequencies(ds.res_sum)
    ds.dominant_frequency.attrs['units'] = 'GHz'


    # %%
    # Plot the dominant frequencies
    # Find the values of flux_coupler_full for which the dominant frequencies are max and min
    max_frequency = 1 / (2 * node.parameters.gate_time_in_ns)  # in GHz
    interaction_max = (ds.dominant_frequency * (ds.dominant_frequency < max_frequency)).max(dim='flux_coupler')
    coupler_flux_pulse = ds.flux_coupler.isel(flux_coupler=(ds.dominant_frequency * (ds.dominant_frequency < max_frequency)).argmax(dim='flux_coupler'))
    coupler_flux_min = ds.flux_coupler_full.isel(flux_coupler=ds.dominant_frequency.argmin(dim='flux_coupler'))

# %% {Plotting}
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):     
        if node.parameters.use_state_discrimination:
            values_to_plot = ds.state_control.sel(qubit=qp['qubit'])
        else:
            values_to_plot = ds.Q_control.sel(qubit=qp['qubit'])
        values_to_plot.plot(ax = ax, cmap = 'viridis', y = 'idle_time', x = 'flux_coupler')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
    grid.fig.suptitle('I Control')
    plt.tight_layout()
    plt.show()
    node.results['figure_I_control'] = grid.fig
    
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            values_to_plot = ds.state_target.sel(qubit=qp['qubit'])
        else:
            values_to_plot = ds.Q_target.sel(qubit=qp['qubit'])
        values_to_plot.plot(ax = ax, cmap = 'viridis', y = 'idle_time', x = 'flux_coupler')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
    grid.fig.suptitle('I Target')
    plt.tight_layout()
    plt.show()
    node.results['figure_I_target'] = grid.fig
    
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        (1e3*ds.dominant_frequency.sel(qubit=qp['qubit'])).plot(ax = ax, marker = '.', ls = 'None', x = 'flux_coupler')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.axvline(x = qubit_pair.coupler.decouple_offset, color = 'black')
        ax.axvline(x = coupler_flux_pulse.sel(qubit=qp['qubit']), color = 'red', lw = 0.5, ls = '--')
        ax.axvline(x = coupler_flux_min.sel(qubit=qp['qubit']) - qubit_pair.coupler.decouple_offset, color = 'green', lw = 0.5, ls = '--')
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
        ax.set_xlabel('Flux Coupler')
        ax.set_ylabel('Frequency (MHz)')
    grid.fig.suptitle('Dominant Frequency')
    plt.tight_layout()
    plt.show()
    node.results['figure_dominant_frequency'] = grid.fig


for qp in qubit_pairs:
    length = math.ceil(node.parameters.gate_time_in_ns / 4) * 4
    zero_padding = length - node.parameters.gate_time_in_ns

    macro_name = f"Cz_square_pulse_{node.parameters.gate_time_in_ns}"
    qp.macros[macro_name] = CZGate(
        flux_pulse_control=FluxPulse(
            length=length,
            amplitude=0.1,
            zero_padding=zero_padding
        ),
        coupler_flux_pulse=FluxPulse(
            length=length,
            amplitude=0.1,
            zero_padding=zero_padding
        )
    )
    qp.macros["Cz"] = f"#./{macro_name}"

# %% {Update_state}
if not node.parameters.simulate:
    with node.record_state_updates():
        for qp in qubit_pairs:
            # gate_time_ns = int(1/ (2 * interaction_max.sel(qubit = qp.name).values))
            # gate_time_including_zeros = gate_time_ns - gate_time_ns % 4 + 4
            # zero_padding = gate_time_including_zeros - gate_time_ns
            coupler_flux_pulse_amp = float(coupler_flux_pulse.sel(qubit = qp.name).values)
            if "coupler_qubit_crosstalk" in qp.extras:
                qubit_flux_pulse_amp = qp.detuning + qp.extras["coupler_qubit_crosstalk"] * coupler_flux_pulse_amp
            else:
                qubit_flux_pulse_amp = qp.detuning

            # qp.macros["Cz"].flux_pulse_control.amplitude = qubit_flux_pulse_amp
            qp.macros["Cz"].coupler_flux_pulse.amplitude = coupler_flux_pulse_amp

            # # qp.coupler.decouple_offset = float(coupler_flux_min.sel(qubit = qp.name).values)
            # # qp.gates['SWAP_Coupler'] = SWAP_Coupler_Gate(flux_pulse_control = FluxPulse(length = gate_time_including_zeros, amplitude = qubit_flux_pulse_amp, zero_padding = zero_padding, id = 'flux_pulse_control_' + qp.qubit_target.name),
            # #                                              coupler_pulse_control = FluxPulse(length = gate_time_including_zeros, amplitude = coupler_flux_pulse_amp, zero_padding = zero_padding, id = 'coupler_pulse_control_' + qp.qubit_target.name))
            # qp.extras["flux_values"] = ds.flux_coupler.values.tolist()
            # qp.extras["J_vs_flux"] = ds.dominant_frequency.sel(qubit = qp.name).values.tolist()
# %% {Save_results}
if not node.parameters.simulate:    
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%

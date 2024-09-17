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
    - Update the readout frequency (f_opt) in the state.
    - Save the current state by calling machine.save("quam")
"""
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal


class Parameters(NodeParameters):
    qubits: Optional[str] = None
    num_averages: int = 40
    frequency_span_in_mhz: float = 10
    frequency_step_in_mhz: float = 0.05
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False

node = QualibrationNode(
    name="06b_Readout_Frequency_Optimization_G_to_E_to_F",
    parameters_class=Parameters
)

node.parameters = Parameters()


from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, multiplexed_readout, node_save

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray

# matplotlib.use("TKAgg")


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
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
if node.parameters.qubits is None or node.parameters.qubits == '':
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits.replace(' ', '').split(',')]
num_qubits = len(qubits)
for q in qubits:
    # check if an EF_x180 operation exists
    if 'EF_x180' in q.xy.operations:
        operation = 'EF_x180'
    else:
        operation = "x180"

###################
# The QUA program #
###################
n_avg = node.parameters.num_averages  # The number of averages
# The frequency sweep around the resonator resonance frequency f_opt
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span/2, +span/2, step)
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

with program() as ro_freq_opt:
    n = declare(int)
    I_g = [declare(fixed) for _ in range(num_qubits)]
    Q_g = [declare(fixed) for _ in range(num_qubits)]
    I_e = [declare(fixed) for _ in range(num_qubits)]
    Q_e = [declare(fixed) for _ in range(num_qubits)]
    I_f = [declare(fixed) for _ in range(num_qubits)]
    Q_f = [declare(fixed) for _ in range(num_qubits)]
    df = declare(int)
    I_g_st = [declare_stream() for _ in range(num_qubits)]
    Q_g_st = [declare_stream() for _ in range(num_qubits)]
    I_e_st = [declare_stream() for _ in range(num_qubits)]
    Q_e_st = [declare_stream() for _ in range(num_qubits)]
    I_f_st = [declare_stream() for _ in range(num_qubits)]
    Q_f_st = [declare_stream() for _ in range(num_qubits)]
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

        for qb in qubits:
            wait(1000, qb.z.name)
        
        align()
        
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(df, dfs)):
                # Update the resonator frequencies
                update_frequency(qubit.resonator.name, df + qubit.resonator.intermediate_frequency)
                align()
                # Measure the state of the resonators
                qubit.resonator.measure("readout", qua_vars=(I_g[i], Q_g[i]))
                qubit.align()
                # Wait for thermalization again in case of measurement induced transitions
                wait(5*machine.thermalization_time * u.ns)
                save(I_g[i], I_g_st[i])
                save(Q_g[i], Q_g_st[i])
                
                # Play the x180 gate to put the qubits in the excited state
                qubit.xy.play("x180")
                # Align the elements to measure after playing the qubit pulses.
                align()
                # Measure the state of the resonators
                qubit.resonator.measure("readout", qua_vars=(I_e[i], Q_e[i]))
                # wait(1000)
                qubit.align()
                # Wait for thermalization again in case of measurement induced transitions
                wait(5*machine.thermalization_time * u.ns)
                save(I_e[i], I_e_st[i])
                save(Q_e[i], Q_e_st[i])

                # Play the x180 gate and EFx180 gate to put the qubits in the f state
                qubit.xy.play("x180")
                update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency -qubit.anharmonicity)                
                qubit.align()
                qubit.xy.play(operation)
                qubit.align()
                update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency) 
                # Align the elements to measure after playing the qubit pulses.
                qubit.align()
                # Measure the state of the resonators
                qubit.resonator.measure("readout", qua_vars=(I_f[i], Q_f[i]))
                # Wait for the qubits to decay to the ground state
                wait(5*machine.thermalization_time * u.ns)                
                save(I_f[i], I_f_st[i])
                save(Q_f[i], Q_f_st[i])

        align()

                
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].buffer(len(dfs)).average().save(f"I_g{i + 1}")
            Q_g_st[i].buffer(len(dfs)).average().save(f"Q_g{i + 1}")
            I_e_st[i].buffer(len(dfs)).average().save(f"I_e{i + 1}")
            Q_e_st[i].buffer(len(dfs)).average().save(f"Q_e{i + 1}")
            I_f_st[i].buffer(len(dfs)).average().save(f"I_f{i + 1}")
            Q_f_st[i].buffer(len(dfs)).average().save(f"Q_f{i + 1}")
            

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, ro_freq_opt, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
    quit()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_octave_ports(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(ro_freq_opt)
    # Get results from QUA program

    for i in range(num_qubits):
        print(f"Fetching results for qubit {qubits[i].name}")
        data_list = sum([[f"I_g{i + 1}", f"Q_g{i + 1}",f"I_e{i + 1}", f"Q_e{i + 1}"] ], ["n"])
        results = fetching_tool(job, data_list, mode="live")
        while results.is_processing():
            fetched_data = results.fetch_all()
            n = fetched_data[0]
            progress_counter(n, n_avg, start_time=results.start_time)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()


# %%
handles = job.result_handles
ds = fetch_results_as_xarray(handles, qubits, {"freq": dfs})

# %%
def abs_freq(q):
    def foo(freq):
        return freq + q.resonator.intermediate_frequency + q.resonator.LO_frequency
    return foo
ds = ds.assign_coords({'freq_full' : (['qubit','freq'],np.array([abs_freq(q)(dfs) for q in qubits]))})
ds.freq_full.attrs['long_name'] = 'Frequency'
ds.freq_full.attrs['units'] = 'GHz'
ds = ds.assign({'Dge' : np.sqrt((ds.I_g - ds.I_e)**2 + (ds.Q_g - ds.Q_e)**2),
                'Def' : np.sqrt((ds.I_e - ds.I_f)**2 + (ds.Q_e - ds.Q_f)**2),
                'Dgf' : np.sqrt((ds.I_g - ds.I_f)**2 + (ds.Q_g - ds.Q_f)**2),
                'IQ_abs_g' : np.sqrt(ds.I_g**2 + ds.Q_g**2),
                'IQ_abs_e' : np.sqrt(ds.I_e**2 + ds.Q_e**2),
                'IQ_abs_f' : np.sqrt(ds.I_f**2 + ds.Q_f**2)})
ds['D'] = ds[['Dge', 'Def',
                           'Dgf']].to_array().min("variable")

node.results = {}
node.results['ds'] = ds

# %%
detuning = ds.D.rolling({"freq" : 5 }).mean("freq").idxmax('freq')
fit_results = {q.name : {'GEF_detuning' :int(detuning.loc[q.name].values)} for q in qubits}
node.results['fit_results'] = fit_results

for q in qubits:
    print(f"{q.name}: GEF readout frequency is shifted by {fit_results[q.name]['GEF_detuning']/1e3:.0f} KHz from the GE readout frequency \n")

# %%
grid_names = [f'{q.name}_0' for q in qubits]
grid = QubitGrid(ds, grid_names)
for ax, qubit in grid_iter(grid):
    (1e3*ds.assign_coords(freq_MHz=ds.freq / 1e6).Dge.loc[qubit]).plot(ax=ax, x = 'freq_MHz', label = "GE")
    (1e3*ds.assign_coords(freq_MHz=ds.freq / 1e6).Def.loc[qubit]).plot(ax=ax, x = 'freq_MHz', label = "EF")
    (1e3*ds.assign_coords(freq_MHz=ds.freq / 1e6).Dgf.loc[qubit]).plot(ax=ax, x = 'freq_MHz', label = "GF")
    (1e3*ds.assign_coords(freq_MHz=ds.freq / 1e6).D.loc[qubit]).plot(ax=ax, x = 'freq_MHz')
    ax.axvline(fit_results[qubit['qubit']]['GEF_detuning']/1e6, color='red', linestyle='--')
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Distance between IQ blobs [m.v.]")
    ax.legend()
plt.tight_layout()
plt.show()
node.results['figure'] = grid.fig

grid = QubitGrid(ds, [f'q-{i}_0' for i in range(num_qubits)])
for ax, qubit in grid_iter(grid):
    (1e3*ds.assign_coords(freq_MHz=ds.freq / 1e6).IQ_abs_g.loc[qubit]).plot(ax=ax, x = 'freq_MHz', label = "g.s.")
    (1e3*ds.assign_coords(freq_MHz=ds.freq / 1e6).IQ_abs_e.loc[qubit]).plot(ax=ax, x = 'freq_MHz', label = "e.s.")
    (1e3*ds.assign_coords(freq_MHz=ds.freq / 1e6).IQ_abs_f.loc[qubit]).plot(ax=ax, x = 'freq_MHz', label = "f.s.")
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Resonator response [mV]")
    ax.legend()
plt.tight_layout()
plt.show()
node.results['figure2'] = grid.fig

# %%
for q in qubits:
    with node.record_state_updates():
        q.GEF_frequency_shift = int(fit_results[q.name]['GEF_detuning'])

# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()

# %%

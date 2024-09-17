# %%
"""
POWER RABI WITH ERROR AMPLIFICATION
This sequence involves repeatedly executing the qubit pulse (such as x180, square_pi, or similar) 'N' times and
measuring the state of the resonator across different qubit pulse amplitudes and number of pulses.
By doing so, the effect of amplitude inaccuracies is amplified, enabling a more precise measurement of the pi pulse
amplitude. The results are then analyzed to determine the qubit pulse amplitude suitable for the selected duration.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated the IQ mixer connected to the qubit drive line (external mixer or Octave port)
    - Having found the rough qubit frequency and pi pulse duration (rabi_chevron_duration or time_rabi).
    - Set the qubit frequency, desired pi pulse duration and rough pi pulse amplitude in the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the qubit pulse amplitude (pi_amp) in the state.
    - Save the current state by calling machine.save("quam")
"""
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal


class Parameters(NodeParameters):
    qubits: Optional[str] = None
    num_averages: int = 50
    operation_x180_or_any_90: Literal['x180', 'x90', '-x90', 'y90', '-y90'] = "x90"
    min_amp_factor: float = 0.8
    max_amp_factor: float = 1.2
    amp_factor_step: float = 0.005
    max_number_rabi_pulses_per_sweep: int = 100
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    reset_type_thermal_or_active: Literal['thermal', 'active'] = "thermal"
    simulate: bool = False

node = QualibrationNode(
    name="08_Power_Rabi_State",
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
from quam_libs.macros import qua_declaration, multiplexed_readout, node_save, active_reset
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_oscillation, oscillation

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

###################
# The QUA program #
###################

operation = node.parameters.operation_x180_or_any_90  # The qubit operation to play, can be switched to "x180" when the qubits are found.
n_avg = node.parameters.num_averages  # The number of averages
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"

# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
amps = np.arange(node.parameters.min_amp_factor,
                 node.parameters.max_amp_factor,
                 node.parameters.amp_factor_step)

# Number of applied Rabi pulses sweep
N_pi = node.parameters.max_number_rabi_pulses_per_sweep  # Maximum number of qubit pulses

if operation == "x180":
    N_pi_vec = np.arange(1,N_pi,2).astype("int")
elif operation in ["x90", "-x90", "y90", "-y90"]:
    N_pi_vec = np.arange(2,N_pi,4).astype("int")
else:
    raise ValueError(f"Unrecognized operation {operation}.")


with program() as power_rabi:
    I, _, Q, _, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(int) for _ in range(num_qubits)]
    state_stream = [declare_stream() for _ in range(num_qubits)]
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    npi = declare(int)  # QUA variable for the number of qubit pulses
    count = declare(int)  # QUA variable for counting the qubit pulses

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
            with for_(*from_array(npi, N_pi_vec)):
                with for_(*from_array(a, amps)):
                    if reset_type == "active":
                        active_reset(machine, qubit.name)
                    else:
                        wait(5*machine.thermalization_time * u.ns)
                    qubit.align()
                    # Loop for error amplification (perform many qubit pulses)
                    with for_(count, 0, count < npi, count + 1):
                        qubit.xy.play(operation, amplitude_scale=a)
                    align()
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                    save(state[i], state_stream[i])
        
        align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            if operation == "x180":
                state_stream[i].buffer(len(amps)).buffer(np.ceil(N_pi / 2)).average().save(f"state{i + 1}")
            elif operation in ["x90", "-x90", "y90", "-y90"]:
                state_stream[i].buffer(len(amps)).buffer(np.ceil(N_pi / 4)).average().save(f"state{i + 1}")
            else:
                raise ValueError(f"Unrecognized operation {operation}.")

###########################
# Run or Simulate Program #
###########################
simulate = node.parameters.simulate

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, power_rabi, simulation_config)
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
    job = qm.execute(power_rabi)
    # Get results from QUA program
    data_list = ["n"] + sum([[f"state{i + 1}"] for i in range(num_qubits)], [])
    results = fetching_tool(job, data_list, mode="live")
    # Live plotting
    # fig = plt.figure()
    # interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        fetched_data = results.fetch_all()
        n = fetched_data[0]
        progress_counter(n, n_avg, start_time=results.start_time)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

# %%
handles = job.result_handles
ds = fetch_results_as_xarray(handles, qubits, {"amp": amps, "N": N_pi_vec})
# %%
def abs_amp(q):
    def foo(amp):
        return q.xy.operations[operation].amplitude * amp
    return foo

ds = ds.assign_coords({'abs_amp' : (['qubit','amp'],np.array([abs_amp(q)(amps) for q in qubits]))})
node.results = {}
node.results['ds'] = ds

# %%
fit_results = {}

if N_pi == 1:
    fit = fit_oscillation(ds.state, 'amp')
    fit_evals = oscillation(ds.amp,fit.sel(fit_vals = 'a'),fit.sel(fit_vals = 'f'),fit.sel(fit_vals = 'phi'),fit.sel(fit_vals = 'offset'))
    for q in qubits:
        fit_results[q.name] = {}
        f_fit = fit.loc[q.name].sel(fit_vals='f')
        phi_fit = fit.loc[q.name].sel(fit_vals='phi')
        phi_fit = phi_fit - np.pi * (phi_fit > np.pi/2)
        factor = float(1.0 * (np.pi - phi_fit)/ (2* np.pi* f_fit))
        new_pi_amp = q.xy.operations[operation].amplitude * factor
        if new_pi_amp < 0.3:
            print(f"amplitude for Pi pulse is modified by a factor of {factor:.2f}")
            print(f"new amplitude is {1e3 * new_pi_amp:.2f} mV \n")
            fit_results[q.name]['Pi_amplitude'] = float(new_pi_amp)
        else:
            print(f"Fitted amplitude too high, new amplitude is 300 mV \n")
            fit_results[q.name]['Pi_amplitude'] = 0.3
    node.results['fit_results'] = fit_results

elif N_pi > 1:
    I_n=ds.state.mean(dim='N')
    datamaxIndx = I_n.argmax(dim='amp')
    for q in qubits:
        new_pi_amp = ds.abs_amp.sel(qubit = q.name)[datamaxIndx.sel(qubit = q.name)]
        fit_results[q.name] = {}
        if new_pi_amp < 0.3:
            fit_results[q.name]['Pi_amplitude'] = float(new_pi_amp)
            print(f"amplitude for Pi pulse is modified by a factor of {I_n.idxmax(dim='amp').sel(qubit = q.name):.2f}")
            print(f"new amplitude is {1e3 * new_pi_amp:.2f} mV \n")
        else:
            print(f"Fitted amplitude too high, new amplitude is 300 mV \n")
            fit_results[q.name]['Pi_amplitude'] = 0.3
    node.results['fit_results'] = fit_results
# %%
grid_names = [f'{q.name}_0' for q in qubits]
grid = QubitGrid(ds, grid_names)
for ax, qubit in grid_iter(grid):
    if N_pi == 1:
        (ds.assign_coords(amp_mV  = ds.abs_amp *1e3).loc[qubit].state).plot(ax = ax, x = 'amp_mV')
        ax.plot(ds.abs_amp.loc[qubit]*1e3, 1e3*fit_evals.loc[qubit][0])
        ax.set_ylabel('Trans. amp. I [mV]')
    elif N_pi > 1:
        (ds.assign_coords(amp_mV  = ds.abs_amp *1e3).loc[qubit].state).plot(ax = ax, x = 'amp_mV', y = 'N')
        ax.axvline(1e3*ds.abs_amp.loc[qubit][datamaxIndx.loc[qubit]], color = 'r')
        ax.set_ylabel('num. of pulses')
    ax.set_xlabel('Amplitude [mV]')
    ax.set_title(qubit['qubit'])
grid.fig.suptitle('Rabi : I vs. amplitude')
plt.tight_layout()
plt.show()
node.results['figure'] = grid.fig

# %%
with node.record_state_updates():
    for q in qubits:
        q.xy.operations[operation].amplitude = fit_results[q.name]['Pi_amplitude']

# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()

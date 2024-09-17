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

from quam_libs.trackable_object import tracked_updates


class Parameters(NodeParameters):
    qubits: Optional[str] = None
    num_averages: int = 10
    operation: str = "x180"
    min_amp_factor: float = 0.0001
    max_amp_factor: float = 2.0
    amp_factor_step: float = 0.02
    max_number_pulses_per_sweep: int = 40
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    reset_type_thermal_or_active: Literal['thermal', 'active'] = "thermal"
    simulate: bool = False

node = QualibrationNode(
    name="10a_DRAG_Calibration_180_minus_180",
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
operation = node.parameters.operation  # The qubit operation to play

if node.parameters.qubits is None:
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits.replace(' ', '').split(',')]

tracked_qubits = []
for q in qubits:
    with tracked_updates(q, auto_revert=False, dont_assign_to_none=True) as q:
        q.xy.operations[operation].alpha = -1.0
        tracked_qubits.append(q)

       
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

if node.parameters.qubits is None or node.parameters.qubits == '':
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits.replace(' ', '').split(',')]
num_qubits = len(qubits)
# %%
# Print intermediate frequencies of all qubits
print("Intermediate frequencies of qubits:")
for qubit in qubits:
    print(f"{qubit.name}: {qubit.xy.intermediate_frequency / 1e6:.3f} MHz")

# %%
###################
# The QUA program #
###################

n_avg = node.parameters.num_averages  # The number of averages
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"

# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
amps = np.arange(node.parameters.min_amp_factor,
                 node.parameters.max_amp_factor,
                 node.parameters.amp_factor_step)

# Number of applied Rabi pulses sweep
N_pi = node.parameters.max_number_pulses_per_sweep  # Maximum number of qubit pulses
N_pi_vec = np.linspace(1, N_pi, N_pi).astype("int")

with program() as drag_calibration:
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
                        qubit.resonator.wait(machine.thermalization_time * u.ns)
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
                    assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                    save(state[i], state_stream[i])

        align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            state_stream[i].buffer(len(amps)).buffer(N_pi).average().save(f"state{i + 1}")


###########################
# Run or Simulate Program #
###########################
simulate = node.parameters.simulate

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, drag_calibration, simulation_config)
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
    job = qm.execute(drag_calibration)
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
def alpha(q):
    def foo(amp):
        return q.xy.operations[operation].alpha * amp
    return foo

ds = ds.assign_coords({'alpha' : (['qubit','amp'],np.array([alpha(q)(amps) for q in qubits]))})
node.results = {}

node.results = {}
node.results['ds'] = ds

# %%
fit_results = {}

state_n=ds.state.mean(dim='N')

datamaxIndx = state_n.argmin(dim='amp')
alphas = ds.amp[datamaxIndx]
fit_results = {qubit.name : {'alpha': float(alphas.sel(qubit=qubit.name).values*qubit.xy.operations[operation].alpha)} for qubit in qubits}
for q in qubits:
    print(f"DRAG coeff for {q.name} is {fit_results[q.name]['alpha']}")
node.results['fit_results'] = fit_results

# %%
ds.state.plot(col = 'qubit', x = 'alpha', y = 'N', col_wrap = 2)
plt.show()

# %%
grid_names = [f'{q.name}_0' for q in qubits]
grid = QubitGrid(ds, grid_names)
for ax, qubit in grid_iter(grid):

    (ds.loc[qubit].state).plot(ax = ax, x = 'alpha', y = 'N')
    ax.axvline(fit_results[qubit['qubit']]['alpha'], color = 'r')
    ax.set_ylabel('num. of pulses')
    ax.set_xlabel('DRAG coeff')
    ax.set_title(qubit['qubit'])
grid.fig.suptitle('DRAG calibration')
plt.tight_layout()
plt.show()
node.results['figure'] = grid.fig

# %%
for qubit in tracked_qubits:
    qubit.revert_changes()
# %%
with node.record_state_updates():
    for q in qubits:
        q.xy.operations[operation].alpha = fit_results[q.name]['alpha']

# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()

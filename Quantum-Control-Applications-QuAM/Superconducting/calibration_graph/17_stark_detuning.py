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
    num_averages: int = 20
    operation: str = "x180"
    frequency_span_in_mhz: float = 10
    frequency_step_in_mhz: float = 0.02
    max_number_pulses_per_sweep: int = 20
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    reset_type_thermal_or_active: Literal['thermal', 'active'] = "thermal"
    simulate: bool = False

node = QualibrationNode(
    name="09_Stark_Detuning",
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


# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == '':
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits.replace(' ', '').split(',')]
num_qubits = len(qubits)
operation = node.parameters.operation  # The qubit operation to play

tracked_qubits = []
for q in qubits:
    with tracked_updates(q, auto_revert=False) as q:
        q.xy.operations[operation].alpha = None
        q.xy.operations[operation].alpha = 0
        q.xy.operations[operation].detuning = None
        q.xy.operations[operation].detuning = 0
        tracked_qubits.append(q)
    
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()
###################
# The QUA program #
###################

n_avg = node.parameters.num_averages  # The number of averages
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"

# Pulse frequency sweep
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span//2, +span//2, step, dtype=np.int32)

# Number of applied Rabi pulses sweep
N_pi = node.parameters.max_number_pulses_per_sweep  # Maximum number of qubit pulses
N_pi_vec = np.linspace(1, N_pi, N_pi).astype("int")

with program() as stark_detuning:
    n = declare(int)
    n_st = declare_stream()
    I = [declare(fixed) for _ in range(num_qubits)]
    I_st = [declare_stream() for _ in range(num_qubits)]
    Q = [declare(fixed) for _ in range(num_qubits)]
    Q_st = [declare_stream() for _ in range(num_qubits)]
    state = [declare(int) for _ in range(num_qubits)]
    state_stream = [declare_stream() for _ in range(num_qubits)]
    df = declare(int)  # QUA variable for the qubit drive amplitude pre-factor
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
                with for_(*from_array(df, dfs)):
                    if reset_type == "active":
                        active_reset(machine, qubit.name)
                    else:
                        qubit.resonator.wait(machine.thermalization_time * u.ns)
                        qubit.align()
                    # Update the qubit frequency
                    update_frequency(qubit.xy.name, df + qubit.xy.intermediate_frequency)
                    with for_(count, 0, count < npi, count + 1):
                        if operation == "x180":
                            qubit.xy.play(operation)
                            qubit.xy.play(operation, amplitude_scale=-1.0)
                        elif operation == "x90":
                            qubit.xy.play(operation)
                            qubit.xy.play(operation)
                            qubit.xy.play(operation, amplitude_scale=-1.0)
                            qubit.xy.play(operation, amplitude_scale=-1.0)
                    update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency)
                    qubit.align()
                    # reset_phase(qubit.resonator.name)
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                    save(state[i], state_stream[i])
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])

        align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            state_stream[i].buffer(len(dfs)).buffer(N_pi).average().save(f"state{i + 1}")
            I_stream = I_st[i].buffer(len(dfs)).buffer(N_pi).average().save(f"I{i + 1}")
            Q_stream = Q_st[i].buffer(len(dfs)).buffer(N_pi).average().save(f"Q{i + 1}")


###########################
# Run or Simulate Program #
###########################
simulate = node.parameters.simulate

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, stark_detuning, simulation_config)
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
    job = qm.execute(stark_detuning, flags=['auto-element-thread'])
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
ds = fetch_results_as_xarray(handles, qubits, {"freq": dfs, "N": N_pi_vec})

node.results = {}
node.results['ds'] = ds

# %%
fit_results = {}


state_n=ds.state.mean(dim='N')

datamaxIndx = state_n.argmin(dim='freq')
detunings = ds.freq[datamaxIndx]
fit_results = {qubit.name : {'detuning': float(detunings.sel(qubit=qubit.name).values)} for qubit in qubits}
for q in qubits:
    print(f"Detuning for {q.name} is {fit_results[q.name]['detuning']} Hz")
node.results['fit_results'] = fit_results

# %%
grid_names = [f'{q.name}_0' for q in qubits]
grid = QubitGrid(ds, grid_names)
for ax, qubit in grid_iter(grid):
    (ds.assign_coords(freq_MHz  = ds.freq *1e6).loc[qubit].state).plot(ax = ax, x = 'freq_MHz', y = 'N')
    ax.axvline(1e6*fit_results[qubit['qubit']]['detuning'], color = 'r')
    ax.set_ylabel('num. of pulses')
    ax.set_xlabel('detuning [MHz]')
    ax.set_title(qubit['qubit'])
grid.fig.suptitle('Stark detuning')
plt.tight_layout()
plt.show()
node.results['figure'] = grid.fig

# %%
all_operations = True
for qubit in tracked_qubits:
    qubit.revert_changes()
with node.record_state_updates():
    for qubit in qubits:
        qubit.xy.operations[operation].detuning = float(fit_results[qubit.name]['detuning'])
        qubit.xy.operations[operation].alpha = -1.0

# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()

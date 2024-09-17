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
    num_averages: int = 200
    operation: str = "x180"
    min_amp_factor: float = 0.8
    max_amp_factor: float = 1.2
    amp_factor_step: float = 0.005
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False

node = QualibrationNode(
    name="06c_Power_Rabi_E_to_F",
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
from quam.components import pulses
import copy
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

for q in qubits:
    # Check if an optimized GEF frequency exists
    if not hasattr(q, 'GEF_frequency_shift'):
        q.GEF_frequency_shift = 0

###################
# The QUA program #
###################

operation = node.parameters.operation  # The qubit operation to play, can be switched to "x180" when the qubits are found.
n_avg = node.parameters.num_averages  # The number of averages
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
amps = np.arange(node.parameters.min_amp_factor,
                 node.parameters.max_amp_factor,
                 node.parameters.amp_factor_step)

with program() as power_rabi:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
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
            with for_(*from_array(a, amps)):
                update_frequency(qubit.resonator.name, qubit.resonator.intermediate_frequency+ q.GEF_frequency_shift)

                # Loop for error amplification (perform many qubit pulses)
                # Reset the qubit frequency
                update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency)
                # Drive the qubit to the excited state
                qubit.xy.play(operation)
                # Update the qubit frequency to scan around the excepted f_01
                update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency -qubit.anharmonicity)                
                qubit.xy.play(operation, amplitude_scale=a)
                align()
                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])
                qubit.resonator.wait(machine.thermalization_time * u.ns)

        align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            I_st[i].buffer(len(amps)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(amps)).average().save(f"Q{i + 1}")


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
    job = qm.execute(power_rabi, flags=['auto-element-thread'])
    # Get results from QUA program
    data_list = ["n"] + sum([[f"I{i + 1}", f"Q{i + 1}"] for i in range(num_qubits)], [])
    results = fetching_tool(job, data_list, mode="live")
    # Live plotting
    # fig = plt.figure()
    # interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        fetched_data = results.fetch_all()
        n = fetched_data[0]
        I = fetched_data[1::2]
        Q = fetched_data[2::2]
        progress_counter(n, n_avg, start_time=results.start_time)
        # I_volts, Q_volts = [], []
        # # Plot results
        # for i, qubit in enumerate(qubits):
        #     if I[i].shape[0] > 1:
        #         # Convert into volts
        #         I_volts.append(u.demod2volts(I[i], qubit.resonator.operations["readout"].length))
        #         Q_volts.append(u.demod2volts(Q[i], qubit.resonator.operations["readout"].length))
        #         # Plot
        #         plt.suptitle("Power Rabi with error amplification")
        #         plt.subplot(3, num_qubits, i + 1)
        #         plt.cla()
        #         plt.pcolor(
        #             amps * qubit.xy.operations[operation].amplitude,
        #             N_pi_vec,
        #             I_volts[i],
        #         )
        #         plt.title(f"{qubit.name} - I")
        #         plt.subplot(3, num_qubits, i + num_qubits + 1)
        #         plt.cla()
        #         plt.pcolor(
        #             amps * qubit.xy.operations[operation].amplitude,
        #             N_pi_vec,
        #             Q_volts[i],
        #         )
        #         plt.title(f"{qubit.name} - Q")
        #         plt.xlabel("Qubit pulse amplitude [V]")
        #         plt.ylabel("Number of Rabi pulses")
        #         plt.subplot(3, num_qubits, i + 2 * num_qubits + 1)
        #         plt.cla()
        #         plt.plot(
        #             amps * qubit.xy.operations[operation].amplitude,
        #             np.sum(I_volts[i], axis=0),
        #         )
        #         plt.axvline(qubit.xy.operations[operation].amplitude, color="k")
        #         plt.xlabel("Rabi pulse amplitude [V]")
        #         plt.ylabel(r"$\Sigma$ of Rabi pulses")

        #     else:
        #         plt.suptitle("Power Rabi")
        #         plt.subplot(2, num_qubits, i + 1)
        #         plt.cla()
        #         plt.plot(amps * qubit.xy.operations[operation].amplitude, I_volts[i])
        #         plt.title(f"{qubit.name}")
        #         plt.ylabel("I quadrature [V]")
        #         plt.subplot(2, num_qubits, i + num_qubits + 1)
        #         plt.cla()
        #         plt.plot(amps * qubit.xy.operations[operation].amplitude, Q_volts[i])
        #         plt.xlabel("Qubit Pulse Amplitude [V]")
        #         plt.ylabel("Q quadrature [V]")

        # plt.tight_layout()
        # plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
    # data = {}
    # for i, qubit in enumerate(qubits):
    #     data[f"{qubit.name}_amplitude"] = amps * qubit.xy.operations[operation].amplitude
    #     data[f"{qubit.name}_I"] = np.abs(I_volts[i])
    #     data[f"{qubit.name}_Q"] = np.angle(Q_volts[i])

    #     # Get the optimal pi pulse amplitude when doing error amplification
    #     try:
    #         qubit.xy.operations[operation].amplitude = (
    #             amps[np.argmax(np.sum(I_volts[i], axis=0))] * qubit.xy.operations[operation].amplitude
    #         )

    #         data[f"{qubit.name}"] = {
    #             "x180_amplitude": qubit.xy.operations[operation].amplitude,
    #             "successful_fit": True,
    #         }

    #     except (Exception,):
    #         data[f"{qubit.name}"] = {"successful_fit": True}
    #         pass

    # data["figure"] = fig
    # Save data from the node
    plt.show()
    # node_save(machine, "power_rabi", data, additional_files=True)

# %%
handles = job.result_handles
ds = fetch_results_as_xarray(handles, qubits, {"amp": amps})

# %%
def abs_amp(q):
    def foo(amp):
        return q.xy.operations[operation].amplitude * amp
    return foo

ds = ds.assign_coords({'abs_amp' : (['qubit','amp'],np.array([abs_amp(q)(amps) for q in qubits]))})
ds = ds.assign({'IQ_abs' : np.sqrt(ds.I**2 + ds.Q**2)})
node.results = {}
node.results['ds'] = ds

# %%
fit_results = {}

fit = fit_oscillation(ds.IQ_abs, 'amp')
fit_evals = oscillation(ds.amp,fit.sel(fit_vals = 'a'),fit.sel(fit_vals = 'f'),fit.sel(fit_vals = 'phi'),fit.sel(fit_vals = 'offset'))
for q in qubits:
    fit_results[q.name] = {}
    f_fit = fit.loc[q.name].sel(fit_vals='f')
    phi_fit = fit.loc[q.name].sel(fit_vals='phi')
    phi_fit = phi_fit - np.pi * (phi_fit > np.pi/2)
    factor = float(1.0 * (np.pi - phi_fit)/ (2* np.pi* f_fit))
    new_pi_amp = q.xy.operations[operation].amplitude * factor
    if new_pi_amp < 0.3:
        print(f"amplitude for E-F Pi pulse is modified by a factor of {factor:.2f} w.r.t the original pi pulse amplitude")
        print(f"new amplitude is {1e3 * new_pi_amp:.2f} mV \n")
        fit_results[q.name]['Pi_amplitude'] = new_pi_amp
    else:
        print(f"Fitted amplitude too high, new amplitude is 300 mV \n")
        fit_results[q.name]['Pi_amplitude'] = 0.3
node.results['fit_results'] = fit_results

# %%
grid_names = [f'{q.name}_0' for q in qubits]
grid = QubitGrid(ds, grid_names)
for ax, qubit in grid_iter(grid):
    (ds.assign_coords(amp_mV  = ds.abs_amp *1e3).loc[qubit].IQ_abs*1e3).plot(ax = ax, x = 'amp_mV')
    ax.plot(ds.abs_amp.loc[qubit]*1e3, 1e3*fit_evals.loc[qubit])
    ax.set_ylabel('Trans. amp. I [mV]')
    ax.set_xlabel('Amplitude [mV]')
    ax.set_title(qubit['qubit'])
grid.fig.suptitle('Rabi : I vs. amplitude')
plt.tight_layout()
plt.show()
node.results['figure'] = grid.fig

# %%
ef_operation_name = f'EF_{operation}'
for q in qubits:
    if fit_results[q.name]['Pi_amplitude'] > 0.3:
        EF_amp = 0.3
    else:
        EF_amp = fit_results[q.name]['Pi_amplitude']
    if ef_operation_name not in q.xy.operations:
        q.xy.operations[ef_operation_name] = pulses.DragCosinePulse(
            # NOTE: this can lead to unwated behavior if the fits fails, because it can make the amplitude be larger than the permitted values
            amplitude=EF_amp,
            alpha=q.xy.operations[operation].alpha,
            anharmonicity=q.xy.operations[operation].anharmonicity,
            length=q.xy.operations[operation].length,
            axis_angle=0,  # TODO: to check that the rotation does not overwrite y-pulses
            digital_marker=q.xy.operations[operation].digital_marker,
        )
    else:
        with node.record_state_updates():
            # set the new amplitude for the EF operation
            q.xy.operations[ef_operation_name].amplitude = EF_amp

# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()

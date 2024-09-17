# %%
"""
        RESONATOR SPECTROSCOPY VERSUS FLUX
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures. This is done across various readout intermediate dfs and flux biases.
The resonator frequency as a function of flux bias is then extracted and fitted so that the parameters can be stored in the configuration.

This information can then be used to adjust the readout frequency for the maximum and minimum frequency points.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude and duration.
    - Specification of the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Adjust the flux bias to the minimum frequency point, labeled as "max_frequency_point", in the state.
    - Adjust the flux bias to the minimum frequency point, labeled as "min_frequency_point", in the state.
    - Save the current state by calling machine.save("quam")
"""
import warnings

from typing import Literal, Optional
from qualibrate import QualibrationNode, NodeParameters


class Parameters(NodeParameters):
    qubits: Optional[str] = None
    num_averages: int = 10
    min_flux_offset_in_v: float = -0.5
    max_flux_offset_in_v: float = 0.5
    num_flux_points: int = 201
    frequency_span_in_mhz: float = 10
    frequency_step_in_mhz: float = 0.05
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False
    input_line_impedance_in_ohm: float = 50
    line_attenuation_in_db: float = 0
    plot_current_mA : bool = True

node = QualibrationNode(
    name="02b_Resonator_Spectroscopy_vs_Flux",
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
from scipy import signal

import matplotlib
import xarray as xr
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_oscillation

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
if any([q.z is None for q in qubits]):
    warnings.warn("Found qubits without a flux line. Skipping")
qubits = [q for q in qubits if q.z is not None]
resonators = [qubit.resonator for qubit in qubits]
num_qubits = len(qubits)
num_resonators = len(resonators)

###################
# The QUA program #
###################

n_avg = node.parameters.num_averages  # The number of averages
# Flux bias sweep in V
dcs = np.linspace(node.parameters.min_flux_offset_in_v,
                  node.parameters.max_flux_offset_in_v,
                  node.parameters.num_flux_points)
# The frequency sweep around the resonator resonance frequency f_opt
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span/2, +span/2, step)

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
update_flux_min = True  # Update the min flux point

with program() as multi_res_spec_vs_flux:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    dc = declare(fixed)  # QUA variable for the flux bias
    df = declare(int)  # QUA variable for the readout frequency

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

        # resonator of the qubit
        rr = resonators[i]

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            with for_(*from_array(dc, dcs)):
                # Flux sweeping by tuning the OPX dc offset associated with the flux_line element
                qubit.z.set_dc_offset(dc)
                wait(100)  # Wait for the flux to settle

                with for_(*from_array(df, dfs)):
                    # Update the resonator frequencies for resonator
                    update_frequency(rr.name, df + rr.intermediate_frequency)

                    # readout the resonator
                    rr.measure("readout", qua_vars=(I[i], Q[i]))

                    # wait for the resonator to relax
                    rr.wait(machine.depletion_time * u.ns)

                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])

        align(*[rr.name for rr in resonators])

    with stream_processing():
        n_st.save("n")
        for i, rr in enumerate(resonators):
            I_st[i].buffer(len(dfs)).buffer(len(dcs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).buffer(len(dcs)).average().save(f"Q{i + 1}")

# %%
#######################
# Simulate or execute #
#######################
simulate = node.parameters.simulate

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_res_spec_vs_flux, simulation_config)
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
    job = qm.execute(multi_res_spec_vs_flux)
    # Get results from QUA program
    data_list = ["n"] + sum([[f"I{i + 1}", f"Q{i + 1}"] for i in range(num_resonators)], [])
    results = fetching_tool(job, data_list, mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        fetched_data = results.fetch_all()
        n = fetched_data[0]
        I = fetched_data[1::2]
        Q = fetched_data[2::2]

        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)

        plt.suptitle("Resonator spectroscopy vs flux")
        A_data = []
        for i, (qubit, rr) in enumerate(zip(qubits, resonators)):
            s = u.demod2volts(I[i] + 1j * Q[i], rr.operations["readout"].length)
            A = np.abs(s)
            A_data.append(A)
            # Plot
            plt.subplot(1, num_resonators, i + 1)
            plt.cla()
            plt.title(f"{rr.name} (LO: {rr.opx_output.upconverter_frequency / u.MHz} MHz)")
            plt.xlabel("flux [V]")
            plt.ylabel(f"{rr.name} IF [MHz]")
            plt.pcolor(
                dcs,
                (rr.opx_output.upconverter_frequency + rr.intermediate_frequency) / u.MHz + dfs / u.MHz,
                A.T,
            )
            plt.plot(
                qubit.z.min_offset,
                (rr.opx_output.upconverter_frequency + rr.intermediate_frequency) / u.MHz,
                "r*",
            )

        plt.tight_layout()
        plt.pause(0.1)

    plt.show()

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat up
    qm.close()

# %%
handles = job.result_handles
ds = fetch_results_as_xarray(handles, qubits, {"freq": dfs, "flux": dcs})
# %%
ds = ds.assign({'IQ_abs': np.sqrt(ds['I'] ** 2 + ds['Q'] ** 2)})

def abs_freq(q):
    def foo(freq):
        return freq + q.resonator.intermediate_frequency + q.resonator.opx_output.upconverter_frequency
    return foo

ds = ds.assign_coords({'freq_full' : (['qubit','freq'],np.array([abs_freq(q)(dfs) for q in qubits]))})

ds.freq_full.attrs['long_name'] = 'Frequency'
ds.freq_full.attrs['units'] = 'GHz'

ds = ds.assign_coords({'current' : (['qubit','flux'], np.array([ds.flux.values * node.parameters.input_line_impedance_in_ohm for q in qubits]))})

# Calculate current after attenuation
attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
attenuated_current = ds.current * attenuation_factor

# Add attenuated current to dataset
ds = ds.assign_coords({'attenuated_current': (['qubit', 'flux'], attenuated_current.values)})

# Set attributes for the new coordinate
ds.attenuated_current.attrs['long_name'] = 'Attenuated Current'
ds.attenuated_current.attrs['units'] = 'A'


ds.current.attrs['long_name'] = 'Current'
ds.current.attrs['units'] = 'A'

node.results = {}
node.results['ds'] = ds

# %%
peak_freq = ds.IQ_abs.idxmin(dim='freq')
# fit to a cosine using the qiskit function
# a * np.cos(2 * np.pi * f * t + phi) + offset
fit_osc = fit_oscillation(peak_freq.dropna(dim='flux'), 'flux')

# making sure the phase is between -pi and pi
idle_offset = -fit_osc.sel(fit_vals='phi')
idle_offset = np.mod(idle_offset+np.pi, 2*np.pi) - np.pi

# converting the phase phi from radians to voltage
idle_offset = idle_offset/fit_osc.sel(fit_vals='f')/2/np.pi

# finding the location of the minimum frequency flux point
flux_min = idle_offset + ((idle_offset < 0  ) -0.5 )  /fit_osc.sel(fit_vals = 'f')
flux_min = flux_min * (np.abs(flux_min) < 0.5) + 0.5 * (flux_min > 0.5) - 0.5 * (flux_min < -0.5)

# finding the frequency as the sweet spot flux
rel_freq_shift = peak_freq.sel(flux=idle_offset, method='nearest')
abs_freq_shift = rel_freq_shift + \
    np.array([q.resonator.opx_output.upconverter_frequency + q.resonator.intermediate_frequency for q in qubits])
q_IF = {}

for q in qubits:
    q_IF[q.name] = q.resonator.intermediate_frequency

    print(
        f'DC offset for {q.name} is {idle_offset.sel(qubit = q.name).data*1e3:.0f} mV')
    print(
        f"Resonator frequency for {q.name} is {(rel_freq_shift.sel(qubit = q.name).values + q.resonator.intermediate_frequency + q.resonator.opx_output.upconverter_frequency)/1e9:.3f} GHz")
    print(
        f"(shift of {rel_freq_shift.sel(qubit = q.name).values/1e6:.0f} MHz)")
    print()
    #
# %%
grid_names = [f'{q.name}_0' for q in qubits]
grid = QubitGrid(ds, grid_names)

for ax, qubit in grid_iter(grid):
    ax2 = ax.twiny()
    
    # Plot using the attenuated current x-axis
    ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].IQ_abs.plot(ax=ax2, add_colorbar=False,
                                                                         x='attenuated_current', y='freq_GHz', robust=True)
    
    # Move ax2 behind ax
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    
    ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].IQ_abs.plot(ax=ax, add_colorbar=False,
                                                                            x='flux', y='freq_GHz', robust=True)
    
    ax.axvline(idle_offset.loc[qubit], linestyle = 'dashed', linewidth = 2, color = 'r')
    ax.axvline(flux_min.loc[qubit],linestyle = 'dashed', linewidth = 2, color = 'orange')


    ax2.set_xlabel('Current (mA)')
    ax2.set_ylabel('Freq (GHz)')
    ax2.set_xlabel('Flux (V)')
    ax.set_title(qubit['qubit'])
    ax2.set_title('')

grid.fig.suptitle('Resonator spectroscopy vs flux ')

plt.tight_layout()
plt.show()
node.results["figure"] = grid.fig
# %%
fit_results = {}
for q in qubits:
    fit_results[q.name] = {}
    fit_results[q.name]["resonator_frequency"] = rel_freq_shift.sel(qubit = q.name).values + q.resonator.RF_frequency
    fit_results[q.name]['min_offset'] = float(flux_min.sel(qubit=q.name).data)
    fit_results[q.name]['offset'] = float(idle_offset.sel(qubit=q.name).data)
    fit_results[q.name]['dv_phi0'] = 1/fit_osc.sel(fit_vals='f', qubit=q.name).values
    fit_results[q.name]['m_pH'] = (1e12) * (2.068e-15) / ((1/fit_osc.sel(fit_vals='f', qubit=q.name).values) / node.parameters.input_line_impedance_in_ohm * attenuation_factor)
node.results['fit_results'] = fit_results

# %%
with node.record_state_updates():
    for q in qubits:
        if not (np.isnan(float(idle_offset.sel(qubit=q.name).data))):
            if flux_point == 'independent':
                q.z.independent_offset = float(idle_offset.sel(qubit=q.name).data)
            else:
                q.z.joint_offset = float(idle_offset.sel(qubit=q.name).data)
            # q.z.extras['dv_phi0'] = 1/fit_osc.sel(fit_vals='f', qubit=q.name).values

            if update_flux_min:
                q.z.min_offset = float(flux_min.sel(qubit=q.name).data)
        q.resonator.intermediate_frequency += float(rel_freq_shift.sel(qubit=q.name).data)
        q.phi0_voltage = fit_results[q.name]['dv_phi0']
        q.phi0_current = fit_results[q.name]['dv_phi0'] * node.parameters.input_line_impedance_in_ohm * attenuation_factor
# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()

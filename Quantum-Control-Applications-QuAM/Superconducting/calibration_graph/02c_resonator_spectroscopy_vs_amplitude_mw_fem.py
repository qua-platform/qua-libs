# %%
"""
        RESONATOR SPECTROSCOPY VERSUS READOUT AMPLITUDE
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures for all resonators simultaneously.
This is done across various readout intermediate dfs and amplitudes.
Based on the results, one can determine if a qubit is coupled to the resonator by noting the resonator frequency
splitting. This information can then be used to adjust the readout amplitude, choosing a readout amplitude value
just before the observed frequency splitting.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude (the pulse processor will sweep up to twice this value) and duration.
    - Specification of the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the readout frequency, labeled as "f_res" and "f_opt", in the state.
    - Adjust the readout amplitude, labeled as "readout_pulse_amp", in the state.
    - Save the current state by calling machine.save("quam")
"""
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List


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
from quam_libs.lib.qua_datasets import apply_angle, subtract_slope
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.fit import peaks_dips
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.trackable_object import tracked_updates


class Parameters(NodeParameters):
    qubits: Optional[str] = None
    num_averages: int = 100
    frequency_span_in_mhz: float = 10
    frequency_step_in_mhz: float = 0.1
    simulate: bool = False
    forced_flux_bias_v: Optional[float] = None
    max_power_dbm: int = 10
    min_power_dbm: int = -20
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "independent"
    ro_line_attenuation_dB: float = 0
    multiplexed: bool = True

node = QualibrationNode(
    name="02c_Resonator_Spectroscopy_vs_Amplitude_Mw_Fem",
    parameters_class=Parameters
)

node.parameters = Parameters()


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
resonators = [qubit.resonator for qubit in qubits]
prev_amps = [rr.operations["readout"].amplitude for rr in resonators]
num_qubits = len(qubits)
num_resonators = len(resonators)

###################
# The QUA program #
###################


n_avg = node.parameters.num_averages  # The number of averages
# Uncomment this to override the initial readout amplitude for all resonators
# for rr in resonators:
#     rr.operations["readout"].amplitude = 0.25
# NOTE: 0.49 is for OPX+, 0.99 is for OPX1000
max_amp = 0.99

tracked_qubits = []

for i, qubit in enumerate(qubits):
    with tracked_updates(qubit, auto_revert=False, dont_assign_to_none=True) as qubit:
        qubit.resonator.operations["readout"].amplitude = max_amp
        qubit.resonator.opx_output.full_scale_power_dbm = node.parameters.max_power_dbm
        if node.parameters.forced_flux_bias_v is not None:
            qubit.z.joint_offset = node.parameters.forced_flux_bias_v
        tracked_qubits.append(qubit)

config = machine.generate_config()

# The readout amplitude sweep (as a pre-factor of the readout amplitude) - must be within [-2; 2)
# amps = np.arange(0.05, 1.00, 0.02)

amp_max = 10**(-(node.parameters.max_power_dbm - node.parameters.max_power_dbm) / 20)
amp_min = 10**(-(node.parameters.max_power_dbm - node.parameters.min_power_dbm) / 20)

amps = np.geomspace(amp_min, amp_max, 100)  # 100 points from 0.01 to 1.0, logarithmically spaced

# The frequency sweep around the resonator resonance frequencies f_opt
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span//2, +span//2, step)
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

with program() as multi_res_spec_vs_amp:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    a = declare(fixed)  # QUA variable for the readout amplitude pre-factor
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

        # resonator of this qubit
        rr = qubit.resonator

        with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
            save(n, n_st)

            with for_(*from_array(df, dfs)):  # QUA for_ loop for sweeping the frequency
                # Update the resonator frequencies for all resonators
                update_frequency(rr.name, df + rr.intermediate_frequency)
                rr.wait(machine.depletion_time * u.ns)

                with for_(*from_array(a, amps)):  # QUA for_ loop for sweeping the readout amplitude
                    # readout the resonator
                    rr.measure("readout", qua_vars=(I[i], Q[i]), amplitude_scale=a)

                    # wait for the resonator to relax
                    rr.wait(machine.depletion_time * u.ns)
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
        if not node.parameters.multiplexed:
            align(*[rr.name for rr in resonators])

        with stream_processing():
            if not i:
                n_st.save("n")
            # for i in range(num_resonators):
            I_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"Q{i + 1}")

#######################
# Simulate or execute #
#######################
simulate = node.parameters.simulate

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_res_spec_vs_amp, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
    quit()
else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Execute the QUA program
    job = qm.execute(multi_res_spec_vs_amp)
    # Prepare the figures for live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)
    # Tool to easily fetch results from the OPX (results_handle used in it)
    res_list = ["n"] + sum([[f"I{i + 1}", f"Q{i + 1}"] for i in range(num_resonators)], [])
    results = fetching_tool(job, res_list, mode="live")
    # Live plotting
    while results.is_processing():
        # Fetch results
        fetched_data = results.fetch_all()
        n = fetched_data[0]
        I_data = fetched_data[1::2]
        Q_data = fetched_data[2::2]

        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)

        plt.suptitle("Resonator spectroscopy vs amplitude")
        A_data = []
        for i, rr in enumerate(resonators):
            s = u.demod2volts(I_data[i] + 1j * Q_data[i], rr.operations["readout"].length)
            A = np.abs(s)
            # Normalize data
            row_sums = A.sum(axis=0)
            A = A / row_sums[np.newaxis, :]
            A_data.append(A)
            # Plot
            plt.subplot(1, num_resonators, i + 1)
            plt.cla()
            plt.title(f"{rr.name} - f_cent: {int(rr.opx_output.upconverter_frequency / u.MHz)} MHz")
            plt.xlabel("Readout amplitude [V]")
            plt.ylabel("Readout detuning [MHz]")
            plt.pcolor(amps * rr.operations["readout"].amplitude, dfs / u.MHz, A)
            plt.axhline(0, color="k", linestyle="--")
            plt.axvline(prev_amps[i], color="k", linestyle="--")

        plt.tight_layout()
        plt.pause(0.1)

    plt.show()

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    while job.status == 'running':
        pass
    qm.close()

    # resonators[0].operations["readout"].amplitude = 0.008
    # resonators[1].operations["readout"].amplitude = 0.008
    # resonators[2].operations["readout"].amplitude = 0.008
    # resonators[3].operations["readout"].amplitude = 0.008
    # resonators[4].operations["readout"].amplitude = 0.008

    # Save data from the node
    # data = {}
    # for i, rr in enumerate(resonators):
    #     data[f"{rr.name}_amplitude"] = amps * rr.operations["readout"].amplitude
    #     data[f"{rr.name}_frequency"] = dfs + rr.intermediate_frequency
    #     data[f"{rr.name}_R"] = A_data[i]
    #     data[f"{rr.name}_readout_amplitude"] = prev_amps[i]
    # data["figure"] = fig
    # node_save(machine, "resonator_spectroscopy_vs_amplitude", data, additional_files=True)

# %%
handles = job.result_handles
ds = fetch_results_as_xarray(handles, qubits, { "amp": amps,"freq": dfs})

# %%
ds = ds.assign({'IQ_abs': np.sqrt(ds['I'] ** 2 + ds['Q'] ** 2)})

def abs_freq(q):
    def foo(freq):
        return freq + q.resonator.opx_output.upconverter_frequency + q.resonator.intermediate_frequency
    return foo

def abs_amp(q):
    def foo(amp):
        return amp * max_amp
    return foo

ds = ds.assign_coords({'freq_full' : (['qubit','freq'],np.array([abs_freq(q)(dfs) for q in qubits]))})
ds = ds.assign_coords({'abs_amp' : (['qubit','amp'],np.array([abs_amp(q)(amps) for q in qubits]))})
# Convert absolute amplitude in volts to dBm
def volts_to_dbm(voltage, impedance=50):
    power_watts = (voltage ** 2) / impedance
    power_dbm = 10 * np.log10(power_watts * 1000)
    return power_dbm

ds = ds.assign_coords({'power_dbm': (['qubit', 'amp'], np.array([volts_to_dbm(q)-node.parameters.ro_line_attenuation_dB for q in ds.abs_amp.values]))})
ds.power_dbm.attrs['long_name'] = 'Power'
ds.power_dbm.attrs['units'] = 'dBm'

ds.freq_full.attrs['long_name'] = 'Frequency'
ds.freq_full.attrs['units'] = 'GHz'

ds.abs_amp.attrs['long_name'] = 'Amplitude'
ds.abs_amp.attrs['units'] = 'V'

ds = ds.assign({'IQ_abs_norm': ds['IQ_abs']/ds.IQ_abs.mean(dim=['freq'])})

node.results = {}
node.results['ds'] = ds

# %%
res_min_vs_amp = [peaks_dips(ds.IQ_abs_norm.sel(
    amp=amp), dim='freq', prominence_factor=5).position for amp in ds.amp]
res_min_vs_amp = xr.concat(res_min_vs_amp, 'amp')
res_freq_full = ds.freq_full.sel(freq=0, method='nearest') + res_min_vs_amp
res_low_power = res_min_vs_amp.sel(amp=slice(0.001,0.03)).mean(dim='amp')
res_hi_power = res_min_vs_amp.isel(amp=-1)

rr_pwr = xr.where(abs(res_min_vs_amp-res_low_power) < 0.15 *
                  (abs(res_hi_power-res_low_power)), res_min_vs_amp.amp, 0).max(dim='amp')

RO_power_ratio = 0.3
rr_pwr = RO_power_ratio*rr_pwr

# %%
grid_names = [q.grid_location for q in qubits]
grid = QubitGrid(ds, grid_names)

for ax, qubit in grid_iter(grid):
    # Create a secondary y-axis for power in dBm
    ax2 = ax.twinx()
    
    # Plot the data using the secondary y-axis
    ds.loc[qubit].IQ_abs_norm.plot(ax=ax, add_colorbar=False,
                                   x='freq_full', y='power_dbm', robust=True)
    
    
    ds.loc[qubit].IQ_abs_norm.plot(ax=ax2, add_colorbar=False,
                                    x='freq_full', y='abs_amp', robust=True,
                                    yscale = 'log')


    ax2.plot(
        res_freq_full.loc[qubit], ds.abs_amp.loc[qubit], color='orange', linewidth=0.5)
    ax2.axhline(y=abs_amp(machine.qubits[qubit['qubit']])(
            rr_pwr.loc[qubit]).values, color='r', linestyle='--')


    # Set the y-axis label for the secondary axis
    ax.set_ylabel('Power (dBm)')


grid.fig.suptitle('Resonator spectroscopy VS. power at base')
plt.tight_layout()
plt.show()
node.results["figure"] = grid.fig


# %%
fit_results = {}
for q in qubits:
    fit_results[q.name] = {}
    if float(rr_pwr.sel(qubit=q.name)) > 0:
        with node.record_state_updates():
            q.resonator.operations["readout"].amplitude = 0.4*float(rr_pwr.sel(qubit=q.name))
            q.resonator.intermediate_frequency+=int(res_low_power.sel(qubit=q.name).values)
    fit_results[q.name]["RO_amplitude"]=float(rr_pwr.sel(qubit=q.name))
node.results['resonator_frequency'] = fit_results

# %%

for tracked_qubit in tracked_qubits:
    tracked_qubit.revert_changes()

# %%
node.outcomes = {q.name: "successful" for q in qubits}
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()

# %%

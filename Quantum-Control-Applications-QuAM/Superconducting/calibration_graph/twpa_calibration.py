"""
        RESONATOR SPECTROSCOPY VERSUS FLUX
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures. This is done across various readout intermediate dfs and flux biases.
The resonator frequency as a function of flux bias is then extracted and fitted so that the parameters can be stored in the state.

This information can then be used to adjust the readout frequency for the maximum and minimum frequency points.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude and duration.
    - Specification of the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the relevant flux biases in the state.
    - Save the current state
"""

# %% {Imports}
from datetime import datetime, timezone, timedelta
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset, get_node_id, save_node
from quam_libs.lib.fit import fit_oscillation
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


# %% {Node_parameters}
class Parameters(NodeParameters):
    twpas: Optional[List[str]] = ['twpa1']
    num_averages: int = 100
    amp_min: float = 0.1
    amp_max: float = 1.5
    amp_step: float = 0.05
    frequency_span_in_mhz: float = 20
    frequency_step_in_mhz: float = 0.5
    p_frequency_span_in_mhz: float = 700
    p_frequency_step_in_mhz: float = 0.5
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    input_line_impedance_in_ohm: float = 50
    line_attenuation_in_db: float = 0
    update_flux_min: bool = False
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False

node = QualibrationNode(name="twpa_calibration", parameters=Parameters())
node_id = get_node_id()

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
twpas = [machine.twpas[t] for t in node.parameters.twpas]
qubits = [machine.qubits[machine.twpas['twpa1'].qubits[i]] for i in range(len(machine.twpas['twpa1'].qubits))]
resonators = [machine.qubits[machine.twpas['twpa1'].qubits[i]].resonator for i in range(len(machine.twpas['twpa1'].qubits))]

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
    

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
update_flux_min = node.parameters.update_flux_min  # Update the min flux point

amp_max = node.parameters.amp_max
amp_min = node.parameters.amp_min
amp_step = node.parameters.amp_step
amps = np.arange(amp_min, amp_max, amp_step)

span_p = node.parameters.p_frequency_span_in_mhz * u.MHz
step_p = node.parameters.p_frequency_step_in_mhz * u.MHz
dps = np.arange(-span_p / 2, +span_p / 2, step_p)

I_list, I_st_list = [], []
Q_list, Q_st_list = [], []
n_list, n_st_list = [], []

with program() as twpa_calibration:
    
    # declare qua variable
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=len(qubits))
    df = declare(int)  # QUA variable for the readout frequency
    dp = declare(int)  # QUA variable for the pump frequency
    amp = declare(float) 
    machine.apply_all_flux_to_min()

    # measure readout responses around readout resonators without pump
    # for i,twpa in enumerate(twpas):
    with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(df, dfs)):
                for i, rr in enumerate(resonators):
                    # Update the resonator frequencies for all resonators
                    update_frequency(rr.name, df + rr.intermediate_frequency)
                    # Measure the resonator
                    rr.measure("readout", qua_vars=(I[i], Q[i]))
                    # wait for the resonator to relax
                    rr.wait(rr.depletion_time * u.ns)
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
    
    with stream_processing():
        n_st.save("n")
        for i in range(len(qubits)):
            I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")
    
        # turn on twpa pump     
    with for_(*from_array(dp, dps)):
            with for_(*from_array(amp, amps)):                
                I_p, I_st_p, Q_p, Q_st_p, n_p, n_st_p = qua_declaration(num_qubits=len(qubits))
              
                update_frequency(twpas[0].pump.name, dp + twpas[0].pump.intermediate_frequency)
                twpas[0].pump.play('pump', amplitude_scale=amp) ######## how can i make it on until the readout measurement is finished
        # measure amplified readout responses around readout resonators with pump
                with for_(n, 0, n < n_avg, n + 1):
                    save(n_p, n_st_p)
                    with for_(*from_array(df, dfs)):
                        for i, rr in enumerate(resonators):
                            # Update the resonator frequencies for all resonators
                            update_frequency(rr.name, df + rr.intermediate_frequency)
                            # Measure the resonator
                            rr.measure("readout", qua_vars=(I_p[i], Q_p[i]))
                            # wait for the resonator to relax
                            rr.wait(rr.depletion_time * u.ns)
                            # save data
                            save(I_p[i], I_st_p[i])
                            save(Q_p[i], Q_st_p[i])
    if not node.parameters.multiplexed:
        align()
    with stream_processing():
        n_st_p.save("n_p")
        for i in range(len(qubits)):
            I_st_p[i].buffer(len(dfs)).average().buffer(len(amp)).buffer(len(dp)).save(f"I_p{i + 1}")
            Q_st_p[i].buffer(len(dfs)).average().buffer(len(amp)).buffer(len(dp)).save(f"Q_p{i + 1}")
              
# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, twpa_calibration, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(twpa_calibration)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    if node.parameters.load_data_id is not None:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    else:
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs, "flux": dcs})
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        RF_freq = np.array([dfs + q.resonator.RF_frequency for q in qubits])
        ds = ds.assign_coords({"freq_full": (["qubit", "freq"], RF_freq)})
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
        # Add the current axis of each qubit to the dataset coordinates for plotting
        current = np.array([ds.flux.values / node.parameters.input_line_impedance_in_ohm for q in qubits])
        ds = ds.assign_coords({"current": (["qubit", "flux"], current)})
        ds.current.attrs["long_name"] = "Current"
        ds.current.attrs["units"] = "A"
        # Add attenuated current to dataset
        attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
        attenuated_current = ds.current * attenuation_factor
        ds = ds.assign_coords({"attenuated_current": (["qubit", "flux"], attenuated_current.values)})
        ds.attenuated_current.attrs["long_name"] = "Attenuated Current"
        ds.attenuated_current.attrs["units"] = "A"
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    # Find the minimum of each frequency line to follow the resonance vs flux
    peak_freq = ds.IQ_abs.idxmin(dim="freq")
    peak_freq_full = {q.name : peak_freq.loc[q.name].values + q.resonator.RF_frequency for q in qubits}

    # Fit to a cosine using the qiskit function: a * np.cos(2 * np.pi * f * t + phi) + offset
    fit_osc = fit_oscillation(peak_freq.dropna(dim="flux"), "flux")
    # Ensure that the phase is between -pi and pi
    idle_offset = -fit_osc.sel(fit_vals="phi")
    idle_offset = np.mod(idle_offset + np.pi, 2 * np.pi) - np.pi
    # converting the phase phi from radians to voltage
    idle_offset = idle_offset / fit_osc.sel(fit_vals="f") / 2 / np.pi
    # finding the location of the minimum frequency flux point
    flux_min = idle_offset + ((idle_offset < 0) - 0.5) / fit_osc.sel(fit_vals="f")
    flux_min = flux_min * (np.abs(flux_min) < 0.5) + 0.5 * (flux_min > 0.5) - 0.5 * (flux_min < -0.5)
    # finding the frequency as the sweet spot flux
    rel_freq_shift = peak_freq.sel(flux=idle_offset, method="nearest")
    abs_freqs = ds.sel(flux=idle_offset, method="nearest").sel(freq = rel_freq_shift).freq_full
    # Save fitting results
    fit_results = {}
    for q in qubits:
        fit_results[q.name] = {}
        fit_results[q.name]["resonator_frequency"] = abs_freqs.sel(qubit=q.name).values
        fit_results[q.name]["min_offset"] = float(flux_min.sel(qubit=q.name).data)
        fit_results[q.name]["offset"] = float(idle_offset.sel(qubit=q.name).data)
        fit_results[q.name]["dv_phi0"] = 1 / fit_osc.sel(fit_vals="f", qubit=q.name).values
        attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
        fit_results[q.name]["m_pH"] = (
            1e12
            * 2.068e-15
            / (
                (1 / fit_osc.sel(fit_vals="f", qubit=q.name).values)
                / node.parameters.input_line_impedance_in_ohm
                * attenuation_factor
            )
        )
        print(f"DC offset for {q.name} is {fit_results[q.name]['offset'] * 1e3:.0f} mV")
        print(f"Resonator frequency for {q.name} is {fit_results[q.name]['resonator_frequency'] / 1e9:.3f} GHz")
        print(f"(shift of {rel_freq_shift.sel(qubit = q.name).values / 1e6:.0f} MHz)")

    node.results["fit_results"] = fit_results

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ax2 = ax.twiny()
        # Plot using the attenuated current x-axis
        ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].IQ_abs.plot(
            ax=ax2,
            add_colorbar=False,
            x="attenuated_current",
            y="freq_GHz",
            robust=True,
        )
        ax2.set_xlabel("Current (A)")
        ax2.set_ylabel("Freq (GHz)")
        ax2.set_title("")
        # Move ax2 behind ax
        ax2.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)
        # Plot using the flux x-axis
        ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].IQ_abs.plot(
            ax=ax, add_colorbar=False, x="flux", y="freq_GHz", robust=True
        )
        ax.plot(ds.flux, peak_freq_full[qubit["qubit"]]/1e9, linewidth=1, color = "purple")
        ax.axvline(
            idle_offset.loc[qubit],
            linestyle="dashed",
            linewidth=2,
            color="r",
            label="idle offset",
        )
        ax.axvline(
            flux_min.loc[qubit],
            linestyle="dashed",
            linewidth=2,
            color="orange",
            label="min offset",
        )
        # Location of the current resonator frequency
        ax.plot(
            idle_offset.loc[qubit].values,
            abs_freqs.sel(qubit=qubit["qubit"]).values
            * 1e-9,
            "r*",
            markersize=10,
        )
        ax.set_title(qubit["qubit"])
        ax.set_xlabel("Flux (V)")

    grid.fig.suptitle(f"Resonator spectroscopy vs flux \n {date_time} GMT+3 #{node_id}")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Update_state}
    if not node.parameters.load_data_id:
        with node.record_state_updates():
            for q in qubits:
                if not (np.isnan(float(idle_offset.sel(qubit=q.name).data))):
                    if flux_point == "independent":
                        q.z.independent_offset = float(idle_offset.sel(qubit=q.name).data)
                    else:
                        q.z.joint_offset = float(idle_offset.sel(qubit=q.name).data)

                    if update_flux_min:
                        q.z.min_offset = float(flux_min.sel(qubit=q.name).data)
                q.resonator.intermediate_frequency += float(rel_freq_shift.sel(qubit=q.name).data)
                q.phi0_voltage = fit_results[q.name]["dv_phi0"]
                q.phi0_current = (
                    fit_results[q.name]["dv_phi0"] * node.parameters.input_line_impedance_in_ohm * attenuation_factor
                )

        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        save_node(node)



# %%

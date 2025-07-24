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
from quam_libs.lib.fit_utils import fit_resonator
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset, get_node_id, save_node
from quam_libs.lib.twpa_utils import  * 
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
class Parameters(NodeParameters):
    twpas: Optional[List[str]] = ['twpa1']
    num_averages: int = 100
    amp_min: float = 0.1
    amp_max: float = 1.5
    amp_step: float = 0.5
    frequency_span_in_mhz: float = 20
    frequency_step_in_mhz: float = 5#0.5
    p_frequency_span_in_mhz: float = 700
    p_frequency_step_in_mhz: float = 300 #0.5
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    input_line_impedance_in_ohm: float = 50
    line_attenuation_in_db: float = 0
    update_flux_min: bool = False
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None

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
daps = np.arange(amp_min, amp_max, amp_step)
daps = np.insert(daps,0,0)

span_p = node.parameters.p_frequency_span_in_mhz * u.MHz
step_p = node.parameters.p_frequency_step_in_mhz * u.MHz
dfps = np.arange(-span_p / 2, +span_p / 2, step_p)


with program() as twpa_calibration:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=len(qubits))
    dp = declare(int)  # QUA variable for the pump frequency
    da = declare(float)# QUA variable for the pump amplitude
    df = declare(int)  # QUA variable for the readout frequency

    # turn on twpa pump     
    with for_(*from_array(dp, dfps)):
            # with for_(*from_array(da, daps)):  
            with for_each_(da, daps):     
                update_frequency(twpas[0].pump.name, dp + twpas[0].pump.intermediate_frequency)
                twpas[0].pump.play('pump', amplitude_scale=da) ######## how can i make it on until the readout measurement is finished
    
    # measure amplified readout responses around readout resonators with pump
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
            I_st[i].buffer(len(dfs)).average().buffer(len(daps)).buffer(len(dfps)).save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).average().buffer(len(daps)).buffer(len(dfps)).save(f"Q{i + 1}")
                    
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
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs, "pump_amp": daps, "pump_freq" : dfps})
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        ds = ds.assign({"IQ_abs": 1e3*np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        # get pump off - resonator spec, signal, snr
        pumpoff_resspec = pumpoff_res_spec_per_qubit(ds.IQ_abs, qubits, dfs, dfps)
        pumpoff_signal_snr = pumpzero_signal_snr(ds.IQ_abs, dfs, qubits, dfps, daps)
        # get pump on - resonator spec, signal, snr
        pumpon_resspec_maxG = pumpoon_maxgain_res_spec(ds.IQ_abs, qubits,  dfps, daps)
        pumpon_resspec_maxDsnr = pumpoon_maxdsnr_res_spec(ds.IQ_abs, qubits,  dfps, daps)
        pumpon_signal_snr = pump_signal_snr(ds.IQ_abs, qubits, dfps, daps)
        # get gain & snr improvement
        gain_dsnr = pumpon_signal_snr-pumpoff_signal_snr

        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        RF_freq = np.array([dfs + q.resonator.RF_frequency for q in qubits])
        # ds = ds.assign_coords({"freq_full": (["qubit", "freq"], RF_freq)})
        # Add the pump amp axis of each qubit to the dataset coordinates for plotting
        pump_amp= np.array([ds.pump_amp.values  for q in qubits])
        pump_freq= np.array([ds.pump_freq.values  for q in qubits])
    node.results = {"ds": ds}

    # %% {Data_analysis}
    p_lo=twpas[0].pump.LO_frequency
    p_if=twpas[0].pump.intermediate_frequency
    # get pump point of max G
    pumpATmaxG=pump_maxgain(pumpon_signal_snr, dfps, daps)
    print(f'max Avg Gain at fp={np.round((p_lo+p_if+pumpATmaxG[0])*1e-9,3)}GHz,Pp={-50-10+pumpATmaxG[1]}')
    # get pump point of max dSNR
    pumpATmaxDSNR=pump_maxdsnr(pumpon_signal_snr,dfps, daps)
    print(f'max Avg dSNR at fp={np.round((p_lo+p_if+pumpATmaxDSNR[0])*1e-9,3)}GHz,Pp={-50-10+pumpATmaxDSNR[1]}')
    
    operation_point={'fp':np.round((p_lo+p_if+pumpATmaxDSNR[0]),3), 'Pp': -50-10+pumpATmaxDSNR[1]}
    node.results["pumping point"] = operation_point

    # %% {Plotting}
    # plot spectroscopy results of pumpoff/on@maxG/on@maxDsnr
    fig, axs = plt.subplots(1, len(qubits), figsize=(25,5))
    for i in range(len(qubits)):
        axs[i].plot(RF_freq[i], pumpoff_resspec[i],label='pumpoff')
        axs[i].plot(RF_freq[i], pumpon_resspec_maxG[i],label='pump @ maxG')
        axs[i].plot(RF_freq[i], pumpon_resspec_maxDsnr[i],label='pump @ maxDsnr')
        axs[i].set_title(f'{qubits[i].name}', fontsize=20)
        axs[i].set_xlabel('Res.freq[GHz]', fontsize=20)
        axs[i].set_ylabel('Trans.amp.[mV]', fontsize=20)
        axs[i].legend()
    plt.tight_layout(pad=2.0) 
    plt.show()
    ##    
    gain_dsnr_avg=np.mean(gain_dsnr,axis=0)
    data_gain = gain_dsnr_avg[:, :, 0] 
    data_dSNR = gain_dsnr_avg[:, :, 1]  
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # plot gain vs pump
    im0 = axs[0].imshow(data_gain, origin='lower', aspect='auto',
                        extent=[0, len(daps)-1, 0, len(dfps)-1])
    axs[0].set_xticks(np.arange(len(daps)))
    axs[0].set_xticklabels(daps)
    axs[0].set_yticks(np.arange(len(dfps)))
    axs[0].set_yticklabels(np.round((dfps*1e-9),3))
    axs[0].set_title('pump vs Gain', fontsize=20)
    axs[0].set_xlabel('pump amplitude', fontsize=20)
    axs[0].set_ylabel('pump frequency[GHz]', fontsize=20)
    cbar0 = fig.colorbar(im0, ax=axs[0])
    cbar0.set_label('Avg Gain [dB]', fontsize=14)

    # plot dSNR vs pump
    im1 = axs[1].imshow(data_dSNR, origin='lower', aspect='auto',
                        extent=[0, len(daps)-1, 0, len(dfps)-1])
    axs[1].set_xticks(np.arange(len(daps)))
    axs[1].set_xticklabels(daps)
    axs[1].set_yticks(np.arange(len(dfps)))
    axs[1].set_yticklabels(np.round((dfps*1e-9),3))
    axs[1].set_title('pump vs dSNR', fontsize=20)
    axs[1].set_xlabel('pump amplitude', fontsize=20)
    axs[1].set_ylabel('pump frequency[GHz]', fontsize=20)
    cbar1 = fig.colorbar(im1, ax=axs[1])
    cbar1.set_label('Avg dSNR [dB]', fontsize=14)

    plt.tight_layout()
    plt.show()

    # %% {Update_state}
    if not node.parameters.load_data_id:
        with node.record_state_updates():
           
            machine.twpas['twpa1'].pump.operations.pump.amplitude=pumpamp # need to find out how to update in the state file
            machine.twpas['twpa1'].pump.intermediate_frequency=pumpfreq

        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        save_node(node)



# %%

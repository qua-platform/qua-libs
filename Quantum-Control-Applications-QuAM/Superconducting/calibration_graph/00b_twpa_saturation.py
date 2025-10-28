
""" Jeongwon Kim IQCC, 251022
do spectroscopy around the bandwidth ~(7GHz ~ 7.6GHz : our usual readout bandwidth)
for various readout power ~(-120dBm~-95dBm)
with the pump off and on(pumping condition is given through node 001) and get the Gain.
For each signal frequency, get the Gain as a function of readout power
and get the input power at which gain is compressed 1dB(P1dB)

Prerequisites:
    - Having calibrated the optimal twpa pumping point (nodes 001). All the gain compression 
      is measured under the given pumping point which is obatained through node001

* Gain is defined as the increase in the signal level.
    - twpa pump off : measure the signal response within 600MHz around the readout bandwidth
      singal_off= signal[dB] 
    - twpa pump on :  measure signal response within 600MHz around the readoutbandwidth
      singal_on= signal[dB]
    => gain=signal_on-signal_off
* P1dB : measure the gain as a function of readout amplitude and find the point where
        gain starts to get compressed 1dB
"""

# %% {Imports}
from datetime import datetime, timezone, timedelta
from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode, NodeParameters
from iqcc_calibration_tools.quam_config.components import Quam
from iqcc_calibration_tools.quam_config.macros import qua_declaration
from iqcc_calibration_tools.quam_config.lib.qua_datasets import convert_IQ_to_V
from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray
from iqcc_calibration_tools.analysis.twpa_utils import  * 
from qualang_tools.results import  fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import math

# %% {Node_parameters}
class Parameters(NodeParameters):
    twpas: Optional[List[str]] = ['twpa3']
    num_averages: int = 100
    frequency_span_in_mhz: float = 600
    frequency_step_in_mhz: float = 1
    amp_min : float = 0.02
    amp_max : float = 1.1
    points : float = 60
    simulate: bool = False
    simulation_duration_ns: int = 4000
    timeout: int = 300
    load_data_id: Optional[int] = None
    pumpline_attenuation: int = -50-5 #(-50: fridge atten(-30)+directional coupler(-20)/ room temp line(4m)~-5,)  #-5: fridge line # exclude for now
    signalline_attenuation : int = -60-6-5 #-60dB : fridge atten, -6dB : cryogenic wiring, -5dB : room temp wiring # galil arbel
node = QualibrationNode(name="00b_twpa_saturation", parameters=Parameters())
# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = Quam.load()

# Get the relevant QuAM components
twpas = [machine.twpas[t] for t in node.parameters.twpas]
qubits = [machine.qubits[machine.twpas['twpa3'].qubits[1]]]
resonators = [machine.qubits[machine.twpas['twpa3'].qubits[1]].resonator]
spectroscopy = [twpas[0].spectroscopy]

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
#%% # readout pulse information
from iqcc_calibration_tools.quam_config.lib.qua_datasets import dBm, mV
readout_power=[np.round(dBm(qubits[i].resonator.opx_output.full_scale_power_dbm,qubits[i].resonator.operations["readout"].amplitude)+node.parameters.signalline_attenuation,2) for i in range(len(qubits))]
readout_length=[qubits[i].resonator.operations["readout"].length for i in range(len(qubits))]
readout_voltage=[np.round(mV(qubits[i].resonator.opx_output.full_scale_power_dbm,qubits[i].resonator.operations["readout"].amplitude),2) for i in range(len(qubits))]
for i in range(len(readout_power)):
    print(f"{qubits[i].name}: readout power @ resonator={readout_power[i]}dBm, opx output={readout_voltage[i]}mV, readout length={readout_length[i]}")
# %% {QUA_program}
n_avg = node.parameters.num_averages  
# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)

# pump amp, frequency sweep
full_scale_power_dbm=twpas[0].pump.opx_output.full_scale_power_dbm
daps = np.logspace(np.log10(node.parameters.amp_min), np.log10(node.parameters.amp_max), node.parameters.points)
f_p = twpas[0].pump_frequency
p_p = twpas[0].pump_amplitude
# pump duration should be able to cover the resonator spectroscopy which takes #(dfs) (as we are multiplexing qubit number doesnt matter) 
pump_duration = (10*len(daps)*len(dfs)*(machine.qubits[twpas[0].qubits[0]].resonator.operations["readout"].length+machine.qubits[twpas[0].qubits[0]].resonator.depletion_time))/4#(n_avg*len(dfs)*(machine.qubits[twpas[0].qubits[0]].resonator.operations["readout"].length+machine.qubits[twpas[0].qubits[0]].resonator.depletion_time))/4
with program() as twpa_pump_off:    
    I, I_st, Q, Q_st,n,n_st = qua_declaration(num_qubits=len(qubits))
    da = declare(float)# QUA variable for the readout amplitude
    df = declare(int)  # QUA variable for the readout frequency
# TWPA off
    with for_(n, 0, n < n_avg, n + 1):  
        save(n, n_st)
# measure readout responses around readout resonators without pump
        with for_each_(da, daps): 
            with for_(*from_array(df, dfs)): 
                for i, spec in enumerate(spectroscopy):
                    # Update the resonator frequencies for all resonators
                    update_frequency(spec.name, df+spec.intermediate_frequency)
                    # Measure the resonator
                    spec.measure("readout", amplitude_scale=da, qua_vars=(I[i], Q[i]))
                    # wait for the resonator to relax
                    spec.wait(spec.depletion_time * u.ns)
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
    align()  
    with stream_processing():
        n_st.save("n")
        for i in range(len(qubits)):
            I_st[i].buffer(len(dfs)).buffer(len(daps)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).buffer(len(daps)).average().save(f"Q{i + 1}")
        
with program() as twpa_pump_on:    
    I, I_st, Q, Q_st,n,n_st = qua_declaration(num_qubits=len(qubits))
    da = declare(float)# QUA variable for the readout amplitude
    df = declare(int)  # QUA variable for the readout frequency
# TWPA on
    with for_(n, 0, n < n_avg, n + 1):  
        save(n, n_st)
        update_frequency(twpas[0].pump.name, f_p+twpas[0].pump.intermediate_frequency)
        twpas[0].pump.play('pump', amplitude_scale=p_p, duration=pump_duration)#+250)
        wait(250) #1000/4 wait 1us for pump to settle before readout
        with for_each_(da, daps): 
            with for_(*from_array(df, dfs)):
                for i, spec in enumerate(spectroscopy):
                    # Update the resonator frequencies for all resonators
                    update_frequency(spec.name, df+spec.intermediate_frequency)
                    # Measure the resonator
                    spec.measure("readout", amplitude_scale=da, qua_vars=(I[i], Q[i]))
                    # wait for the resonator to relax
                    spec.wait(spec.depletion_time * u.ns)
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
        align()  
    with stream_processing():
        n_st.save("n")
        for i in range(len(qubits)):
            I_st[i].buffer(len(dfs)).buffer(len(daps)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).buffer(len(daps)).average().save(f"Q{i + 1}")
# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, twpa_pump_off, simulation_config)
    
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(f'{con}-Pump, Readout pulse simulation')
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(twpa_pump_off)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job_ = qm.execute(twpa_pump_on)
        results_ = fetching_tool(job_, ["n"], mode="live")
        while results_.is_processing():
            n_ = results_.fetch_all()[0]
# %% {Data_fetching_and_dataset_creation}
#data for pump off
ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs,"ro_amp": daps})
ds_ = fetch_results_as_xarray(job_.result_handles, qubits, {"freq": dfs,"ro_amp": daps})
# Convert IQ data into volts
ds = convert_IQ_to_V(ds, qubits)
ds_ = convert_IQ_to_V(ds_, qubits)
# Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
ds = ds.assign({"IQ_abs": 1e3*np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
ds_ = ds_.assign({"IQ_abs": 1e3*np.sqrt(ds_["I"] ** 2 + ds_["Q"] ** 2)})
# %% {Data Analysis}
# Gain & P 1dB point (saturation power)
Gain = mvTOdbm(ds_.IQ_abs.values[0])-mvTOdbm(ds.IQ_abs.values[0])
ps=np.round(dBm(twpas[0].spectroscopy.opx_output.full_scale_power_dbm,
                daps*twpas[0].spectroscopy.operations["readout"].amplitude) # resonator amp should be 0.1 to make ps from -120~-94
                +node.parameters.signalline_attenuation,2)

# %% {Plotting}
print(f"calibration time = {np.round(1e-9*len(ps)*len(dfs)*n_avg*(machine.qubits[twpas[0].qubits[0]].resonator.operations['readout'].length + machine.qubits[twpas[0].qubits[0]].resonator.depletion_time), 3)} sec")
# fs VS Gain : Gain compression along the bandwidth 
fs=np.array([dfs+q.resonator.RF_frequency for q in qubits])[0]
plt.figure(figsize=(4,3))
for i in range(1,len(daps),10):
    plt.plot(fs,Gain[i], label=f'Ps={ps[i]}dBm')
    plt.xlabel('fs[GHz]')
    plt.ylabel('Gain[dB]')
    plt.title(f'{twpas[0].id} Gain Compression')
plt.legend(loc='upper right', bbox_to_anchor=(1.7, 1))
gain_profile=plt.gcf()
# ps VS Gain / fs
plt.figure(figsize=(4,3))
for i in range(0, Gain.shape[1], 90):   # take every 5th column
    plt.plot(ps, Gain[:, i], label=f'{np.round(fs[i]*1e-9,3)}GHz')
# ps VS AvgGain(of all fs)
plt.plot(ps, np.mean(Gain,axis=1),linewidth=3, color='red',label='Avg Gain')
# get compression point
avg_gain_of_all_fs=np.mean(Gain,axis=1)
linear_gain = np.mean(avg_gain_of_all_fs[:10])
gain_1db_compression = linear_gain - 1
p1db = np.argmax(avg_gain_of_all_fs < gain_1db_compression)
plt.scatter(ps[p1db], gain_1db_compression, label=f'P1dB={ps[p1db]}dBm', color='black', marker='x',s=35, zorder=10)
plt.xlabel('Ps[dBm]')
plt.ylabel('Gain[dB]')
plt.title(f"{twpas[0].id}: Gain Compression ")
plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1), fontsize=7)
gain_compression=plt.gcf()

# %% {Update state}
p1db = ps[p1db]
node.results = {"p_saturation":p1db,
                "fp":f_p+twpas[0].pump.intermediate_frequency+twpas[0].pump.LO_frequency,
                 "pp":p_p }
node.results["figures"]={"gain_profile": gain_profile,
                         "gain_compression": gain_compression}
if not node.parameters.load_data_id:
    with node.record_state_updates():        
        machine.twpas['twpa3'].p_saturation=p1db
    # %% {Save_results}
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%

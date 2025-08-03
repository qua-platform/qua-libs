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
    num_averages: int = 1
    amp_min: float =  0.1
    amp_max: float =  0.4
    amp_step: float = 0.1 # 0.01
    frequency_span_in_mhz: float = 7
    frequency_step_in_mhz: float = 7
    p_frequency_span_in_mhz: float = 100
    p_frequency_step_in_mhz: float = 30
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = True
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    pumpline_attenuation: int = -50-6 #(-50: fridge atten, directional coupler, -6: room temp line, -5: fridge line)

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

amp_max = node.parameters.amp_max
amp_min = node.parameters.amp_min
amp_step = node.parameters.amp_step
daps = np.arange(amp_min, amp_max, amp_step)
# daps = np.insert(daps,0,0)

span_p = node.parameters.p_frequency_span_in_mhz * u.MHz
step_p = node.parameters.p_frequency_step_in_mhz * u.MHz
dfps = np.arange(-span_p / 2, +span_p / 2, step_p)
dfps = np.array([-3e6,0])
# pump duration should be able to cover the resonator spectroscopy which takes #(dfs)*n_avg (as we are multiplexing qubit number doesnt matter) 
pump_duration = (n_avg*len(dfs)*(machine.qubits["qB1"].resonator.operations["readout"].length+machine.qubits["qB1"].resonator.depletion_time))/4

with program() as twpa_calibration:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=len(qubits))
    dp = declare(int)  # QUA variable for the pump frequency
    da = declare(float)# QUA variable for the pump amplitude
    df = declare(int)  # QUA variable for the readout frequency

# #### test for checking pump with SA
#     with infinite_loop_():
#         update_frequency(twpas[0].pump.name,  -3e6+ twpas[0].pump.intermediate_frequency)
#         twpas[0].pump.play('pump', amplitude_scale=0.2, duration=pump_duration)
# ####
    # turn on twpa pump     
    with for_(*from_array(dp, dfps)):  
            with for_each_(da, daps):     
                update_frequency(twpas[0].pump.name, dp + twpas[0].pump.intermediate_frequency)
                twpas[0].pump.play('pump', amplitude_scale=da, duration=pump_duration+2500)#250/4) 
                wait(2500) #1000/4 wait 1us for pump to settle 
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
                align()    

    with stream_processing():
        n_st.save("n")
        for i in range(len(qubits)):
            I_st[i].buffer(len(dfs)).average().buffer(len(daps)).buffer(len(dfps)).save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).average().buffer(len(daps)).buffer(len(dfps)).save(f"Q{i + 1}")

    
# ## kill
# qm=qmm.open_qm(config,close_other_machines=True)
# job=qm.execute(twpa_calibration)        
# # #qm.close()            
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
        plt.title(f'{con}-Pump, Readout pulse simulation')
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
#%%
res = job.get_simulated_samples().con1
t_min, t_max = 0,700     # ns

def t_axis(n_samples, sr):
    return np.arange(n_samples) / sr * 1e9       # ns
def label_from(port, typ):
    port = str(port)
    addr = port.split("-")
    if len(addr) == 2:
        return f"FEM{addr[0]}-{typ}O{addr[1]}"
    if len(addr) == 3:
        return f"FEM{addr[0]}-{typ}O{addr[1]}-UP{addr[2]}"
    return port
# --- analog ---
for port, samples in res.analog.items():
    sr     = res._analog_sampling_rate[str(port)]
    t      = t_axis(len(samples), sr)
    window = (t >= t_min) & (t <= t_max)

    if np.iscomplexobj(samples):
        I = samples.real
        Q = samples.imag
        if np.any(I[window]) or np.any(Q[window]):
            plt.plot(t[window], I[window], label=f"{label_from(port,'A')} I")
            plt.plot(t[window], Q[window], label=f"{label_from(port,'A')} Q")
    else:
        if np.any(samples[window]):
            plt.plot(t[window], samples[window], label=label_from(port,'A'))
# --- digital (optional, rarely useful in this window) ---
for port, dig in res.digital.items():
    if not np.any(dig):
        continue
    t  = t_axis(len(dig), 1e9)   # digital always at 1 GS/s
    window = (t >= t_min) & (t <= t_max)
    plt.plot(t[window], dig[window], label=label_from(port,'D'))
plt.xlabel("Time [ns]")
plt.ylabel("Output")
# plt.legend()
plt.xlim(t_min, t_max)          # keeps ticks sensible
plt.tight_layout()
plt.show()
# %%

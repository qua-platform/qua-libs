# %%
"""
RAMSEY WITH VIRTUAL Z ROTATIONS
The program consists in playing a Ramsey sequence (x90 - idle_time - x90 - measurement) for different idle times.
Instead of detuning the qubit gates, the frame of the second x90 pulse is rotated (de-phased) to mimic an accumulated
phase acquired for a given detuning after the idle time.
This method has the advantage of playing resonant gates.

From the results, one can fit the Ramsey oscillations and precisely measure the qubit resonance frequency and T2*.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Update the qubits frequency (f_01) in the state.
    - Save the current state by calling machine.save("quam")
"""
from turtledemo.penrose import start

from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal
from qualang_tools.multi_user import qm_session

import time


class Parameters(NodeParameters):
    qubits: Optional[str] = ["qubitC1"]
    num_averages: int = 200
    num_repetition: int = 10
    frequency_detuning_in_mhz: float = 1.0
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 1000
    wait_time_step_in_ns: int = 50
    flux_point_joint_or_independent_or_arbitraryPulse_or_arbitraryDC: Literal['joint', 'independent', 'arbitraryPulse', 'arbitraryDC'] = "joint" 
    simulate: bool = False
    use_state_discrimination: bool = True
    timeout: int = 100

node = QualibrationNode(
    name="05c_Ramsey_EF_repetitive", parameters=Parameters()
)


from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, multiplexed_readout, node_save, readout_state

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp

# matplotlib.use("TKAgg")


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handle units and conversions.
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
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)
# %%
###################
# The QUA program #
###################
n_avg = node.parameters.num_averages  # The number of averages

# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
idle_times = np.arange(
    node.parameters.min_wait_time_in_ns // 4,
    (node.parameters.max_wait_time_in_ns+node.parameters.wait_time_step_in_ns/2) // 4,
    node.parameters.wait_time_step_in_ns // 4,
)

# Detuning converted into virtual Z-rotations to observe Ramsey oscillation and get the qubit frequency
detuning = int(1e6 * node.parameters.frequency_detuning_in_mhz)
flux_point = node.parameters.flux_point_joint_or_independent_or_arbitraryPulse_or_arbitraryDC  # 'independent' or 'joint'
if flux_point == "arbitraryDC" or "arbitraryPulse":
    # arb_detunings = {q.name : q.xy.intermediate_frequency - q.arbitrary_intermediate_frequency for q in qubits}
    arb_detunings = {q.name: 0.0 for q in qubits}
    arb_flux_bias_offset = {q.name: q.z.arbitrary_offset for q in qubits}
else:
    arb_flux_bias_offset = {q.name: 0.0 for q in qubits}
    arb_detunings = {q.name: 0.0 for q in qubits}

n_rep = node.parameters.num_repetition
ms = np.arange(n_rep)
    
with program() as ramsey:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    m = declare(int)
    m_st = declare_stream()
    t = declare(int)  # QUA variable for the idle time
    sign = declare(int)  # QUA variable to change the sign of the detuning
    phi = declare(fixed)  # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)
    if node.parameters.use_state_discrimination:
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        
    for i, q in enumerate(qubits):

        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_zero()
            q.z.to_independent_idle()
        elif flux_point == "joint" or "arbitraryPulse":
            machine.apply_all_flux_to_joint_idle()
        elif flux_point == "arbitraryDC":
            machine.apply_all_flux_to_joint_idle()
            q.z.set_dc_offset(q.z.arbitrary_offset+q.z.joint_offset)
            q.xy.update_frequency(q.arbitrary_intermediate_frequency)
        else:
            machine.apply_all_flux_to_zero()

        for qb in qubits:
            wait(1000, qb.z.name)
        
        align()
        with for_(m, 0, m < n_rep, m + 1):
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(t, idle_times)):

                    assign(phi, Cast.mul_fixed_by_int((-arb_detunings[q.name] - detuning) * 1e-9, 4 * t))

                    align()
                    # # Strict_timing ensures that the sequence will be played without gaps
                    # with strict_timing_():
                    q.xy.update_frequency(q.xy.intermediate_frequency)
                    q.xy.play("x180",amplitude_scale=0)
                    q.xy.play("x180")
                    q.xy.update_frequency(q.xy.intermediate_frequency - q.anharmonicity)
                    q.xy.play("x180",amplitude_scale=0)
                    q.xy.play("EF_x90")
                    q.xy.frame_rotation_2pi(phi)

                    q.align()

                    # if reading out at the detuned frequency, no need to z-pulse during wait
                    if flux_point != "arbitraryPulse":
                        wait(t)
                    # if reading out at the USS, z-pulse to arbitrary detuning during idle
                    else:
                        q.z.wait(20)
                        q.z.play("const", amplitude_scale=arb_flux_bias_offset[q.name]/q.z.operations["const"].amplitude, duration=t)
                        q.z.wait(20)

                    q.align()
                    q.xy.play("EF_x90")
                    q.xy.update_frequency(q.xy.intermediate_frequency)
                    q.xy.play("x180",amplitude_scale=0)
                    q.xy.play('x180')
                    q.xy.update_frequency(q.xy.intermediate_frequency - q.anharmonicity)
                    q.xy.play("x180",amplitude_scale=0)
                    q.xy.play("EF_x180")

                    # Align the elements to measure after playing the qubit pulse.
                    align()
                    # Measure the state of the resonators

                    # save data
                    if node.parameters.use_state_discrimination:
                        readout_state(q, state[i])
                        save(state[i], state_st[i])

                    else:
                        q.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])

                    # Wait for the qubits to decay to the ground state
                    q.resonator.wait(machine.thermalization_time * u.ns)

                    # Reset the frame of the qubits in order not to accumulate rotations
                    reset_frame(q.xy.name)

        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            if node.parameters.use_state_discrimination:
                state_st[i].buffer(len(idle_times)).buffer(n_avg).map(FUNCTIONS.average(0)).buffer(n_rep).save(f"state{i + 1}")
            else:
                I_st[i].buffer(len(idle_times)).buffer(n_avg).map(FUNCTIONS.average(0)).buffer(n_rep).save(f"I{i + 1}")
                Q_st[i].buffer(len(idle_times)).buffer(n_avg).map(FUNCTIONS.average(0)).buffer(n_rep).save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, ramsey, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(ramsey)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)


# %%
if node.parameters.use_state_discrimination:
    # results = fetching_tool(job, data_list=["state"])
    _,state = results.fetch_all()
idle_time_ns  = 4*idle_times
reps = ms

elapsed_time = np.round(end_time - start_time,decimals=1)
real_time_s = np.linspace(0,elapsed_time,n_rep)

fig = plt.figure()
xpl = idle_time_ns*1e-3
ypl = real_time_s
plt.pcolormesh(xpl,ypl,state)
plt.xlabel('Idle time (us)')
plt.ylabel('Time (s)')
plt.colorbar(label='state')
plt.show()
node.results['figure1'] = fig

node.results['elapsed_time'] = elapsed_time


node.results['state'] = state
node.results['idle_time_ns'] = idle_time_ns
node.results['real_time_s'] = real_time_s

def calc_fft(state,sampling_interval=200e-9,dc_cutoff=5):
    signal = state-np.mean(state)
    fft_result = np.fft.rfft(signal)
    fft_freq = np.fft.rfftfreq(len(signal), d=sampling_interval)
    power_spectrum = np.abs(fft_result) ** 2
    return fft_freq[dc_cutoff:],power_spectrum[dc_cutoff:]



sampling_interval = (idle_time_ns[1]-idle_time_ns[0])*1e-9
fft_freq,_ = calc_fft(state[0,:],sampling_interval=sampling_interval)
power_spectrum_s = np.array([calc_fft(state[_i,:],sampling_interval=sampling_interval)[1] for _i in range(len(real_time_s))])

from matplotlib.colors import LogNorm

xpl = real_time_s
ypl = fft_freq*1e-6
zpl = power_spectrum_s.T
fig2 = plt.figure()
plt.pcolormesh(xpl,ypl,zpl,norm=LogNorm(vmin=zpl.min(), vmax=zpl.max()))
plt.ylabel('Frequency (MHz)')
plt.xlabel('Time (s)')
# pl.colorbar(label='State')
margin = np.max(ypl) - node.parameters.frequency_detuning_in_mhz
plt.ylim([node.parameters.frequency_detuning_in_mhz-margin,node.parameters.frequency_detuning_in_mhz+margin])
node.results['figure2'] = fig2

# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()

# %%

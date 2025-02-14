"""
        POWER RABI WITH ERROR AMPLIFICATION
This sequence involves repeatedly executing the qubit pulse (such as x180) 'N' times and
measuring the state of the resonator across different qubit pulse amplitudes and number of pulses.
By doing so, the effect of amplitude inaccuracies is amplified, enabling a more precise measurement of the pi pulse
amplitude. The results are then analyzed to determine the qubit pulse amplitude suitable for the selected duration.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated the IQ mixer connected to the qubit drive line (external mixer or Octave port)
    - Having found the rough qubit frequency and set the desired pi pulse duration (qubit spectroscopy).
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the qubit pulse amplitude in the state.
    - Save the current state
"""


# %% {Imports}

from qualibrate import QualibrationNode
from quam_libs.components import QuAM
from quam_libs.experiments.power_rabi.parameters import Parameters, get_number_of_rabi_pulses
from quam_libs.experiments.power_rabi.plotting import plot_rabi_oscillations
from quam_libs.experiments.simulation import simulate_and_plot
from quam_libs.experiments.power_rabi.analysis import fetch_dataset, fit_pi_amplitude
from quam_libs.macros import qua_declaration, active_reset

from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit

from qm.qua import program, declare, for_, assign, save, stream_processing, align, declare_stream, fixed

import numpy as np


# %% {Node_parameters}
node = QualibrationNode(
    name="04_Power_Rabi",
    parameters=Parameters(
       qubits = None,
       num_averages = 50,
       operation_x180_or_any_90 = "x180",
       min_amp_factor = 0.001,
       max_amp_factor = 1.99,
       amp_factor_step = 0.005,
       max_number_rabi_pulses_per_sweep = 1,
       flux_point_joint_or_independent = "independent",
       reset_type_thermal_or_active = "thermal",
       state_discrimination = False,
       update_x90 = True,
       simulate = False,
       simulation_duration_ns = 2500,
       timeout = 100,
       load_data_id = None,
       multiplexed = True
    ),
)


# %% {Initialize_QuAM_and_QOP}


u = unit(coerce_to_integer=True)

machine = QuAM.load()

qubits = machine.get_qubits_used_in_node(node.parameters)
resonators = machine.get_resonators_used_in_node(node.parameters)

config = machine.generate_config()

num_qubits = len(qubits)
if node.parameters.load_data_id is None:
    qmm = machine.connect()    


# %% {QUA_program}
n_avg = node.parameters.num_averages  
N_pi = node.parameters.max_number_rabi_pulses_per_sweep  
flux_point = node.parameters.flux_point_joint_or_independent  
reset_type = node.parameters.reset_type_thermal_or_active  
state_discrimination = node.parameters.state_discrimination
operation = node.parameters.operation_x180_or_any_90  
amps = np.arange(
                        node.parameters.min_amp_factor,
                        node.parameters.max_amp_factor,
                        node.parameters.amp_factor_step,
                    )
N_rabi_pulses = get_number_of_rabi_pulses(node.parameters)

with program() as power_rabi:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    if state_discrimination:
        state = [declare(bool) for _ in range(num_qubits)]
        state_stream = [declare_stream() for _ in range(num_qubits)]
    a = declare(fixed)  
    npi = declare(int)  
    count = declare(int) 
 
    for multiplexed_qubits in qubits.batch():
        
        q0 = list(multiplexed_qubits.values())[0]
        
        machine.set_all_fluxes(flux_point, target=q0)
            
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(npi, N_rabi_pulses)):
                with for_(*from_array(a, amps)):
                    
                    for qubit in multiplexed_qubits.values():
                        # Initialize the qubits
                        if reset_type == "active":
                            active_reset(qubit, "readout")
                        else:
                            qubit.wait(qubit.thermalization_time * u.ns)
                    
                    align()
                    
                    for qubit in multiplexed_qubits.values():

                        # Loop for error amplification (perform many qubit pulses)
                        with for_(count, 0, count < npi, count + 1):
                            qubit.xy.play(operation, amplitude_scale=a)
                    
                    align()
                    
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        if state_discrimination:
                            assign(state[i], I[i] > qubit.resonator.operations["readout"].threshold)
                            save(state[i], state_stream[i])
                        else:
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                                

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            if operation == "x180":
                if state_discrimination:
                    state_stream[i].boolean_to_int().buffer(len(amps)).buffer(np.ceil(N_pi / 2)).average().save(
                        f"state{i + 1}"
                    )
                else:
                    I_st[i].buffer(len(amps)).buffer(np.ceil(N_pi / 2)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(amps)).buffer(np.ceil(N_pi / 2)).average().save(f"Q{i + 1}")

            elif operation in ["x90", "-x90", "y90", "-y90"]:
                if state_discrimination:
                    state_stream[i].boolean_to_int().buffer(len(amps)).buffer(np.ceil(N_pi / 4)).average().save(
                        f"state{i + 1}"
                    )
                else:
                    I_st[i].buffer(len(amps)).buffer(np.ceil(N_pi / 4)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(amps)).buffer(np.ceil(N_pi / 4)).average().save(f"Q{i + 1}")
            else:
                raise ValueError(f"Unrecognized operation {operation}.")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    fig, samples = simulate_and_plot(qmm, config, power_rabi, node.parameters)
    node.results = {"figure": fig, "samples": samples}
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(power_rabi)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds = fetch_dataset(job, qubits, parameters=node.parameters, state_discrimination=state_discrimination)                           
    else:
        load_data_id = node.parameters.load_data_id
        node = node.load_from_id(load_data_id)
        node.parameters.load_data_id = load_data_id
        ds = node.results["ds"] 
    
    node.results = {"ds": ds}

# %% {Data_analysis}
fit_results = fit_pi_amplitude(ds, N_pi, state_discrimination, qubits, operation, N_rabi_pulses)
node.results["fit_results"] = fit_results 
# %% {Plotting}
fig = plot_rabi_oscillations(ds, qubits, fit_results, N_pi, state_discrimination)
node.results["figure"] = fig
# %% {Update_state}
if node.parameters.load_data_id is None:
    with node.record_state_updates():
        for q in qubits:
            q.xy.operations[operation].amplitude = fit_results["pi_amp_fit"][q.name]["Pi_amplitude"]
            if operation == "x180" and node.parameters.update_x90:
                q.xy.operations["x90"].amplitude = fit_results["pi_amp_fit"][q.name]["Pi_amplitude"] / 2

# %% {Save_results}
node.outcomes = {q.name: "successful" for q in qubits}
node.results["initial_parameters"] = node.parameters.model_dump()
node.machine = machine
node.save()

# %%

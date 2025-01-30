"""
        T1 MEASUREMENT
The sequence consists in putting the qubit in the excited stated by playing the x180 pulse and measuring the resonator
after a varying time. The qubit T1 is extracted by fitting the exponential decay of the measured quadratures.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit spectroscopy, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update the qubit T1 in the state.
"""

# %% {Imports}
from qualibrate import QualibrationNode
from quam_libs.components import QuAM
from quam_libs.experiments.T1.parameters import Parameters, get_arb_flux_offset_for_each_qubit, get_idle_times_in_clock_cycles
from quam_libs.experiments.T1.analysis import fetch_dataset, fit_exponential_decay
from quam_libs.experiments.T1.plotting import plot
from quam_libs.experiments.simulation import simulate_and_plot
from quam_libs.macros import qua_declaration, active_reset, readout_state
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit

from qm.qua import program, declare, for_, save, stream_processing, align, declare_stream, strict_timing_, reset_frame

import numpy as np


# %% {Node_parameters}
node = QualibrationNode(
    name="05_T1", 
    parameters=Parameters(
        qubits = None,
        num_averages = 100,
        min_wait_time_in_ns = 16,
        max_wait_time_in_ns = 100000,
        wait_time_step_in_ns = 600,
        flux_point_joint_or_independent_or_arbitrary = "independent",
        reset_type = "thermal",
        use_state_discrimination = False,
        simulate = False,
        simulation_duration_ns = 2500,
        timeout = 100,
        load_data_id = None,
        multiplexed = False
        )
    )


# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)

machine = QuAM.load()

qubits = machine.get_qubits_used_in_node(node.parameters)
num_qubits = len(qubits)

config = machine.generate_config()
if node.parameters.load_data_id is None:
    qmm = machine.connect()
    
# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
idle_times = get_idle_times_in_clock_cycles(node.parameters)

flux_point = node.parameters.flux_point_joint_or_independent_or_arbitrary  # 'independent' or 'joint'
arb_flux_offset = get_arb_flux_offset_for_each_qubit(qubits, flux_point)

with program() as t1:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    t = declare(int)  # QUA variable for the idle time
    if node.parameters.use_state_discrimination:
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        
    for multiplexed_qubits in qubits.batch():
        
        for qubit in multiplexed_qubits.values():
             # Bring the active qubits to the desired frequency point
            machine.set_all_fluxes(flux_point, target=qubit)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(t, idle_times)):
                
                for qubit in multiplexed_qubits.values():
                    if node.parameters.reset_type == "active":
                        active_reset(qubit, "readout")
                    else:
                        qubit.resonator.wait(qubit.thermalization_time * u.ns)
                        qubit.align() # TODO : is this necessary?
                align()
                
                for qubit in multiplexed_qubits.values():
                    with strict_timing_(): # TODO : is this necessary?
                        qubit.xy.play("x180")
                        qubit.align()
                        qubit.z.wait(20)
                        qubit.z.play(
                            "const",
                            amplitude_scale=arb_flux_offset[qubit.name] / qubit.z.operations["const"].amplitude,
                            duration=t,
                        )
                        qubit.z.wait(20)
                        qubit.align()
                
                align()

                for i, qubit in multiplexed_qubits.items():
                    # Measure the state of the resonators
                    if node.parameters.use_state_discrimination:
                        readout_state(qubit, state[i])
                        save(state[i], state_st[i])
                    else:
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        
                align()
                
                for i, qubit in multiplexed_qubits.items(): # TODO : Added this. Is this necessary?
                        qubit.resonator.wait(qubit.thermalization_time * u.ns)
                        reset_frame(qubit.xy.name)
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            if node.parameters.use_state_discrimination:
                state_st[i].buffer(len(idle_times)).average().save(f"state{i + 1}")
            else:
                I_st[i].buffer(len(idle_times)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(idle_times)).average().save(f"Q{i + 1}")


# with program() as t1:
#     I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
#     t = declare(int)  # QUA variable for the idle time
#     if node.parameters.use_state_discrimination:
#         state = [declare(int) for _ in range(num_qubits)]
#         state_st = [declare_stream() for _ in range(num_qubits)]
#     for i, qubit in enumerate(qubits):

#         # Bring the active qubits to the desired frequency point
#         machine.set_all_fluxes(flux_point=flux_point, target=qubit)

#         with for_(n, 0, n < n_avg, n + 1):
#             save(n, n_st)
#             with for_(*from_array(t, idle_times)):
#                 if node.parameters.reset_type == "active":
#                     active_reset(qubit, "readout")
#                 else:
#                     qubit.resonator.wait(qubit.thermalization_time * u.ns)
#                     qubit.align()
                
#                 qubit.xy.play("x180")
#                 qubit.align()
#                 qubit.z.wait(20)
#                 qubit.z.play(
#                     "const",
#                     amplitude_scale=arb_flux_offset[qubit.name] / qubit.z.operations["const"].amplitude,
#                     duration=t,
#                 )
#                 qubit.z.wait(20)
#                 qubit.align()

#                 # Measure the state of the resonators
#                 if node.parameters.use_state_discrimination:
#                     readout_state(qubit, state[i])
#                     save(state[i], state_st[i])
#                 else:
#                     qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
#                     # save data
#                     save(I[i], I_st[i])
#                     save(Q[i], Q_st[i])
#         # Measure sequentially
#         if not node.parameters.multiplexed:
#             align()

#     with stream_processing():
#         n_st.save("n")
#         for i in range(num_qubits):
#             if node.parameters.use_state_discrimination:
#                 state_st[i].buffer(len(idle_times)).average().save(f"state{i + 1}")
#             else:
#                 I_st[i].buffer(len(idle_times)).average().save(f"I{i + 1}")
#                 Q_st[i].buffer(len(idle_times)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    fig = simulate_and_plot(qmm, config, t1, node.parameters)
    node.results = {"figure": fig}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(t1)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds = fetch_dataset(job, qubits, idle_times=idle_times, unit=u)
        node.results = {"ds": ds}
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    fit_results = fit_exponential_decay(ds, node.parameters)
    # %% {Plotting}
    fig = plot(ds, qubits, fit_results, node)
    node.results["figure_raw"] = fig    
    # %% {Update_state}
    tau = fit_results["tau"]
    tau_error = fit_results["tau_error"]
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for index, q in enumerate(qubits):
                
                t1_is_positive  = float(tau.sel(qubit=q.name).values) > 0
                t1_error_is_smaller_than_t1 = tau_error.sel(qubit=q.name).values / float(tau.sel(qubit=q.name).values) < 1
                
                if (t1_is_positive and t1_error_is_smaller_than_t1):
                    q.T1 = float(tau.sel(qubit=q.name).values) * 1e-6
        # %% {Save_results}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()


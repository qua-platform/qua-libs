"""
        RESONATOR SPECTROSCOPY MULTIPLEXED
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to extract the
'I' and 'Q' quadratures across varying readout intermediate frequencies for all resonators simultaneously.
The data is then post-processed to determine the resonator resonance frequency.
This frequency can be used to update the readout frequency in the state.

Prerequisites:
    - Ensure calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibrate the IQ mixer connected to the readout line (whether it's an external mixer or an Octave port).
    - Define the desired readout pulse amplitude and duration in the state.
    - Specify the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the readout frequency, in the state for all resonators.
    - Save the current state
"""

# %% {Imports}
from qualibrate import QualibrationNode
from quam_libs.components import QuAM
from quam_libs.experiments.simulation import simulate_and_plot
from quam_libs.experiments.resonator_spectroscopy.analysis import fetch_dataset, fit_resonators
from quam_libs.experiments.resonator_spectroscopy.plotting import plot_raw_amplitude, plot_raw_phase, plot_fit_amplitude
from quam_libs.experiments.resonator_spectroscopy.parameters import Parameters
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm.qua import *
import numpy as np

from quam_libs.macros import qua_declaration
from qualang_tools.loops import from_array


# %% {Node_parameters}
node = QualibrationNode(
    name="02a_Resonator_Spectroscopy",
    parameters=Parameters(
        qubits=None,
        num_averages=100,
        frequency_span_in_mhz=30.0,
        frequency_step_in_mhz=0.1,
        simulate=False,
        simulation_duration_ns=2500,
        timeout=100,
        load_data_id=None,
        multiplexed=True
    )
)

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Get the relevant QuAM components
qubits = machine.get_qubits_used_in_node(node)
resonators = machine.get_resonators_used_in_node(node)
num_qubits = len(qubits)
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# %% {QUA_program}
n_avg = node.parameters.num_averages
# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)

with program() as multi_res_spec:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    df = declare(int)  # QUA variable for the readout frequency

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    for _resonators in resonators.batch():
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(df, dfs)):
                for i, rr in enumerate(_resonators):
                    # Update the resonator frequencies for all resonators
                    update_frequency(rr.name, df + rr.intermediate_frequency)
                    # Measure the resonator
                    rr.measure("readout", qua_vars=(I[i], Q[i]))
                    # wait for the resonator to relax
                    rr.wait(machine.depletion_time * u.ns)
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])

                    align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    fig, samples = simulate_and_plot(qmm, config, multi_res_spec, node.parameters)
    node.results = {"figure": fig, "samples": samples}

else:
    if node.parameters.load_data_id is None:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            # Send the QUA program to the OPX, which compiles and executes it
            job = qm.execute(multi_res_spec)
            # Creates a result handle to fetch data from the OPX
            res_handles = job.result_handles
            # Waits (blocks the Python console) until all results have been acquired
            res_handles.wait_for_all_values()

    # %% {Data_fetching_and_dataset_creation}
    if node.parameters.load_data_id is None:
        ds = fetch_dataset(job, qubits, frequencies = dfs)
        node.results = {"ds": ds}
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]

    # %% {Data_analysis}
    ds, fit_results = fit_resonators(ds, qubits)
    node.results["fit_results"] = fit_results
    
    # %% {Plotting}
    fig = plot_raw_amplitude(ds, qubits)
    node.results["raw_amplitude"] = fig    
    
    fig = plot_raw_phase(ds, qubits)
    node.results["raw_phase"] = fig
    
    fig = plot_fit_amplitude(ds, qubits)
    node.results["fit_amplitude"] = fig

    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for index, q in enumerate(qubits):
                q.resonator.intermediate_frequency += int(fit_results[q.name]["resonator_freq_detuning"])

        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()
        print("Results saved")


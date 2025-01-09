"""
        TIME OF FLIGHT
This sequence involves sending a readout pulse and capturing the raw ADC traces.
The data undergoes post-processing to calibrate three distinct parameters:
    - Time of Flight: This represents the internal processing time and the propagation delay of the readout pulse.
    Its value can be adjusted in the configuration under "time_of_flight".
    This value is utilized to offset the acquisition window relative to when the readout pulse is dispatched.

    - Analog Inputs Offset: Due to minor impedance mismatches, the signals captured by the OPX might exhibit slight offsets.

    - Analog Inputs Gain: If a signal is constrained by digitization or if it saturates the ADC,
    the variable gain of the OPX analog input, ranging from -12 dB to 20 dB, can be modified to fit the signal within the ADC range of +/-0.5V.
"""

from qualibrate import QualibrationNode

from CS_installations.quam_libs.experiments.two_qubit_rb.test.configuration import time_of_flight
from quam_libs.components import QuAM
from quam_libs.experiments.simulation import simulate_and_plot
from quam_libs.experiments.time_of_flight.analysis import fetch_dataset, analyze_pulse_arrival_times
from quam_libs.experiments.time_of_flight.node import patch_readout_pulse_params
from quam_libs.experiments.time_of_flight.plotting import plot_adc_single_runs, plot_adc_averaged_runs
from quam_libs.experiments.time_of_flight.parameters import Parameters
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm.qua import *
import numpy as np


# %% {Node_parameters}
node = QualibrationNode(
    name="01b_Time_of_Flight",
    parameters=Parameters(
        qubits=None,
        num_averages=100,
        time_of_flight_in_ns=24,
        intermediate_frequency_in_mhz=50,
        readout_amplitude_in_v=0.1,
        readout_length_in_ns=None,
        timeout=100,
        simulate=True
        # simulation_duration=10_000
    )
)


# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)

machine = QuAM.load()

qubits = machine.get_qubits_used_in_node(node.parameters)
resonators = machine.get_resonators_used_in_node(node.parameters)
patched_resonators = patch_readout_pulse_params(resonators, node.parameters)

config = machine.generate_config()
qmm = machine.connect()


# %% {QUA_program}
with program() as raw_trace_prog:
    n = declare(int)  # QUA variable for the averaging loop
    adc_st = [declare_stream(adc_trace=True) for _ in range(len(resonators))]  # The stream to store the raw ADC trace

    for i, rr in enumerate(resonators):
        with for_(n, 0, n < node.parameters.num_averages, n + 1):
            # Reset the phase of the digital oscillator associated to the resonator element. Needed to average the cosine signal.
            reset_phase(rr.name)
            # Measure the resonator (send a readout pulse and record the raw ADC trace)
            rr.measure("readout", stream=adc_st[i])
            # Wait for the resonator to deplete
            rr.wait(machine.depletion_time * u.ns)
        # Measure sequentially
        align(*[rr.name for rr in resonators])

    with stream_processing():
        for i in range(len(qubits)):
            # Will save average:
            adc_st[i].input1().average().save(f"adcI{i + 1}")
            adc_st[i].input2().average().save(f"adcQ{i + 1}")
            # Will save only last run:
            adc_st[i].input1().save(f"adc_single_runI{i + 1}")
            adc_st[i].input2().save(f"adc_single_runQ{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    fig = simulate_and_plot(qmm, config, time_of_flight, node.parameters)
    node.results = {"figure": fig}

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(raw_trace_prog)
        # Creates a result handle to fetch data from the OPX
        res_handles = job.result_handles
        # Waits (blocks the Python console) until all results have been acquired
        res_handles.wait_for_all_values()

    # %% {Data_fetching_and_dataset_creation}
    ds = fetch_dataset(job, qubits, resonators)
    node.results = {"ds": ds}

    # %% {Data_analysis}
    ds, fit_results = analyze_pulse_arrival_times(ds, qubits)
    node.results["fit_results"] = fit_results

    # %% {Plotting}
    fig = plot_adc_single_runs(ds, qubits)
    node.results["adc_single_run"] = fig

    fig = plot_adc_averaged_runs(ds, qubits)
    node.results["adc_averaged"] = fig

    # %% {Update_state}
    # Update the time of flight
    with node.record_state_updates():
        for q in qubits:
            delay = int(ds.sel(qubit=q.name).delays)
            if node.parameters.time_of_flight_in_ns is not None:
                q.resonator.time_of_flight = node.parameters.time_of_flight_in_ns + delay
            else:
                q.resonator.time_of_flight += delay

    # Update the offsets per controller for each qubit
    for con in np.unique(ds.con.values):
        for i, q in enumerate(ds.where(ds.con == con).qubit.values):
            resonator = machine.qubits[q].resonator
            ds_at_controller = ds.where(ds.con == con)
            # Only add the offsets once,
            if i == 0:
                if resonator.opx_input_I.offset is not None:
                    resonator.opx_input_I.offset += float(ds_at_controller.offsets_I.mean(dim="qubit").values)
                else:
                    resonator.opx_input_I.offset = float(ds_at_controller.offsets_I.mean(dim="qubit").values)

                if resonator.opx_input_Q.offset is not None:
                    resonator.opx_input_Q.offset += float(ds_at_controller.offsets_Q.mean(dim="qubit").values)
                else:
                    resonator.opx_input_Q.offset = float(ds_at_controller.offsets_Q.mean(dim="qubit").values)

            # else copy the values from the updated qubit
            else:
                resonator.opx_input_I.offset = machine.qubits[ds.where(ds.con == con).qubit.values[0]].resonator.opx_input_I.offset
                resonator.opx_input_Q.offset = machine.qubits[ds.where(ds.con == con).qubit.values[0]].resonator.opx_input_Q.offset

    # Revert the change done at the beginning of the node
    for resonator in patched_resonators:
        resonator.revert_changes()

    # %% {Save_results}
    node.outcomes = {rr.name: "successful" for rr in resonators}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.results["ds"] = ds
    node.machine = machine
    node.save()


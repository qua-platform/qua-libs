"""
"""

from datetime import datetime
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset, get_node_id
from quam_libs.trackable_object import tracked_updates
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import pickle


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_averages: int = 100
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100


node = QualibrationNode(name="000_readout_raw_adc_01", parameters=Parameters())
node_id = get_node_id()

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
resonators = [qubit.resonator for qubit in qubits]
num_qubits = len(qubits)

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()


# %% {QUA_program}
with program() as raw_trace_prog:
    n = declare(int)  # QUA variable for the averaging loop
    
    I = [declare(fixed) for _ in range(num_qubits)]
    Q = [declare(fixed) for _ in range(num_qubits)]
    I_st = [declare_stream() for _ in range(num_qubits)]
    Q_st = [declare_stream() for _ in range(num_qubits)]
    
    
    adc_st = [declare_stream(adc_trace=True) for _ in range(len(resonators))]  # The stream to store the raw ADC trace

    for i, qubit in enumerate(qubits):
        # Wait for the qubits to decay to the ground state
        with for_(n, 0, n < node.parameters.num_averages, n + 1):
            
            qubit.wait(4* qubit.thermalization_time * u.ns)
            qubit.align()
            # Reset the phase of the digital oscillator associated to the resonator element. Needed to average the cosine signal.
            reset_phase(qubit.resonator.name)
            # Measure the resonator (send a readout pulse and record the raw ADC trace)
            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]), stream=adc_st[i])
            # save data
            save(I[i], I_st[i])
            save(Q[i], Q_st[i])
            # |0> -> |1>
            qubit.align()
            qubit.wait(qubit.thermalization_time * u.ns)
            qubit.align()
            qubit.xy.play("x180")
            qubit.align()
            # Reset the phase of the digital oscillator associated to the resonator element. Needed to average the cosine signal.
            reset_phase(qubit.resonator.name)
            # Measure the resonator (send a readout pulse and record the raw ADC trace)
            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]), stream=adc_st[i])
            # save data
            save(I[i], I_st[i])
            save(Q[i], Q_st[i])
        # Measure sequentially
        align(*[qubit.resonator.name for qubit in qubits])
        

    with stream_processing():
        for i in range(num_qubits):
            # Will save average:
            adc_st[i].input1().real().buffer(2).buffer(node.parameters.num_averages).save(f"adcI{i + 1}")
            adc_st[i].input1().image().buffer(2).buffer(node.parameters.num_averages).save(f"adcQ{i + 1}")
            I_st[i].buffer(2).buffer(node.parameters.num_averages).save(f"I{i + 1}")
            Q_st[i].buffer(2).buffer(node.parameters.num_averages).save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, raw_trace_prog, simulation_config)
    # Plot the simulated samples
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # save the figure
    node.results = {"figure": plt.gcf()}
else:
    
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(raw_trace_prog)
        # Creates a result handle to fetch data from the OPX
        res_handles = job.result_handles
        # Waits (blocks the Python console) until all results have been acquired
        res_handles.wait_for_all_values()

    # %% {Data_fetching_and_dataset_creation}
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ts = 1
    time_axis = ts * np.arange(0,resonators[0].operations["readout"].length)
    ds_adc = fetch_results_as_xarray(job.result_handles, qubits, {"time": time_axis, "qubit_init_state": range(2), "shot_index": range(node.parameters.num_averages)}, 
                                 keys = ["adcI1", "adcQ1", "adcI2", "adcQ2"])
    # Convert raw ADC traces into volts
    ds_adc = ds_adc.assign({key: -ds_adc[key] / 2**12 for key in ("adcI", "adcQ")})
    ds_adc = ds_adc.assign({"IQ_abs": np.sqrt(ds_adc["adcI"] ** 2 + ds_adc["adcQ"] ** 2)})
    # Add the dataset to the node
    node.results = {"ds_adc": ds_adc}
    # %%
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds_IQ = fetch_results_as_xarray(job.result_handles, qubits, {"qubit_init_state": range(2), "shot_index": range(node.parameters.num_averages)}, 
                                 keys = ["I1", "Q1", "I2", "Q2"])
    # Add the dataset to the node
    node.results = {"ds_IQ": ds_IQ}

    # %% {Data_analysis}

# %% {Save_results}
    # Save the xarray dataset to a pickle file
    with open(f"raw_adc_readout_01_{date_time.replace(":","_")}.pkl", 'wb') as f:
        pickle.dump(ds_adc, f)
    with open(f"readout_01_IQ_{date_time.replace(":","_")}.pkl", 'wb') as f:
        pickle.dump(ds_IQ, f)

    # # %% {Save_results}
    
    # node.outcomes = {rr.name: "successful" for rr in resonators}
    # node.results["ds"] = ds
    # node.results["initial_parameters"] = node.parameters.model_dump()
    # node.machine = machine
    # node.save()


# %%

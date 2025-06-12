# %% {Imports}
from datetime import datetime
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset, get_node_id, save_node
from quam_libs.trackable_object import tracked_updates
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
    qubits: Optional[List[str]] = ["qC1"]
    num_averages: int = 10  
    initial_num_readout: int = 2
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True
    seed: int = 345324



node = QualibrationNode(name="measurment_mitigation", parameters=Parameters())
node_id = get_node_id()

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
flux_point = "joint"
initial_num_readout = node.parameters.initial_num_readout


def generate_pauli_pair_int():
    
    rand = Random(seed=node.parameters.seed)
    pauli_pair_int = declare(int)
    assign(pauli_pair_int, rand.rand_int(8))
    
    return pauli_pair_int

def measure_with_rc(pauli_pair_int):
    with switch_(pauli_pair_int):
        with case_(0): # II
            qubit.align()
            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
        with case_(1): # IZ
            qubit.align()
            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
            qubit.align()
            qubit.xy.frame_rotation(np.pi)
        with case_(2): # ZI
            qubit.xy.frame_rotation(np.pi)
            qubit.align()
            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
        with case_(3): # ZZ
            qubit.xy.frame_rotation(np.pi)
            qubit.align()
            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
            qubit.align()
            qubit.xy.frame_rotation(np.pi)
        with case_(4): # XY
            qubit.xy.play("x180")
            qubit.align()
            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
            qubit.align() # align()
            qubit.xy.play("y180")
        with case_(5): # YX
            qubit.xy.play("y180")
            qubit.align()
            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
            qubit.align() # align()
            qubit.xy.play("x180")
        with case_(6): # XX
            qubit.xy.play("x180")
            qubit.align()
            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
            qubit.align() # align()
            qubit.xy.play("x180")
        with case_(7): # YY
            qubit.xy.play("y180")
            qubit.align()
            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
            qubit.align() # align()
            qubit.xy.play("y180")
            
            


with program() as drag_calibration:
    I, _, Q, _, n, n_st = qua_declaration(num_qubits=num_qubits)
    j = declare(int)
    state = [declare(bool, size=initial_num_readout) for _ in range(num_qubits)]
    state_stream = [declare_stream() for _ in range(num_qubits)]
    paulis_stream = [declare_stream() for _ in range(num_qubits)]

    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            qubit.wait(qubit.thermalization_time * u.ns)
            with for_(j, 0, j < initial_num_readout, j + 1):
                
                index = generate_pauli_pair_int()
                measure_with_rc(index)
                
                assign(state[i][j], I[i] > qubit.resonator.operations["readout"].threshold)
                save(state[i][j], state_stream[i])
                save(index, paulis_stream[i])
                wait(qubit.resonator.depletion_time // 4)
            
            #qubit.align()
            #qubit.xy.play("x180") # X gate
            # qubit.xy.play("x180") # Y gate
            # qubit.xy.play("x90") # X90 gate
            # qubit.z.play(...) # Z gate
            # qubit.xy.frame_rotation_2pi(phi) # virtual Z gate
            
            
        # Measure sequentially
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            # get all shots
            state_stream[i].boolean_to_int().buffer(initial_num_readout).buffer(n_avg).save(f"state{i + 1}")
            # # avergee over the shots 
            # state_stream[i].boolean_to_int().buffer(initial_num_readout).average().save(f"state{i + 1}")
            paulis_stream[i].buffer(initial_num_readout).buffer(n_avg).save(f"pauli{i + 1}") 


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, drag_calibration, simulation_config)
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
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(drag_calibration)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # get all shots
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"readout_number": range(initial_num_readout), "n": range(n_avg)})
        # # average over the shots
        # # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        # ds = fetch_results_as_xarray(job.result_handles, qubits, {"readout_number": range(initial_num_readout)})
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    
    # %% {Save_results}
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()


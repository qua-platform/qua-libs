# %%

# %load_ext autoreload
# %autoreload 2

import numpy as np
from qm.qua import *
from qm import SimulationConfig
from qualang_tools.multi_user import qm_session

from qm import generate_qua_script

from qualang_tools.results import progress_counter, fetching_tool

from quam_libs.experiments.rb.parameters import BaseNode, Parameters
from quam_libs.experiments.rb.qua_utils import reset_qubits
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.macros import readout_state

# from qualibrate import QualibrationNode

from quam_libs.components import QuAM
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.experiments.rb.rb_utils import StandardRB, quantum_circuit_to_integer_and_angle_list

import matplotlib.pyplot as plt



# %% {Node_parameters}

basis_gates = ['rz', 'x90',  'x180', 'cz'] # , 'y90', 'y180']

node = BaseNode(
    name="2Q_RB",
    parameters=Parameters(
        qubit_pair_name="qC2-qC1",
        circuit_lengths=[4], # [1,2,4,8,16,32],
        num_circuits_per_length = 1,
        num_averages = 1,
        basis_gates = basis_gates, # gates are as written in state
        seed = 0,
        reset_type = "thermal",
        simulate = True,
        timeout = 100
    )
)

x90_id = 2
x180_id = 4
rz_id = 8
cz_id = 16
measure_id = 32
barrier_id = 64

gate_to_int = dict(zip(basis_gates, range(len(basis_gates))))

gate_to_int['measure'] = measure_id
gate_to_int['barrier'] = barrier_id
gate_to_int['cz'] = cz_id
gate_to_int['rz'] = rz_id

int_to_gate = {v: k for k, v in gate_to_int.items()}
# %% {Initialize_QuAM_and_QOP}
num_qubits = 2
u = unit(coerce_to_integer=True)
machine = QuAM.load()

# Qubits and resonators
qubit_pair = [pair for pair in machine.active_qubit_pairs if pair.id == node.parameters.qubit_pair_name][0]
qubit_control = qubit_pair.qubit_control
qubit_target = qubit_pair.qubit_target
qubits = [qubit_control, qubit_target]


# cz gate
cz = qubit_pair.gates['Cz']


# thermalization time
thermalization_time = qubit_control.thermalization_time if qubit_control.thermalization_time > qubit_target.thermalization_time else qubit_target.thermalization_time

config = machine.generate_config()

qmm = machine.connect()

qmm.close_all_quantum_machines()

# %% {Random circuit generation}

standardRB = StandardRB(
    circuit_lengths=node.parameters.circuit_lengths,
    num_circuits_per_length=node.parameters.num_circuits_per_length,
    basis_gates=node.parameters.basis_gates,
    num_qubits=2,
    seed=node.parameters.seed
)

transpiled_circuits = standardRB.transpiled_circuits
# %% {the QUA program}

int_to_xy = {k : v for k,v in int_to_gate.items() if v not in ['measure', 'barrier', 'rz', 'cz']}

n_avg = node.parameters.num_averages  # The number of averages

circuits_as_int = []
circuit_gate_angles = []
qubit_indices = []
circuits_start_index = [0]

gate_to_int_qiskit_naming = {standardRB.map_to_known_basis_gates.get(k, k) : v for k, v in gate_to_int.items()}
for l, circuits in transpiled_circuits.items():
    for qc in circuits:
        gate_int, gate_angle, qubit_int = quantum_circuit_to_integer_and_angle_list(qc, gate_to_int_qiskit_naming)
        qubit_indices.extend(qubit_int)
        circuits_as_int.extend(gate_int)
        circuit_gate_angles.extend(gate_angle)


circuits_as_int = [2,2,2,2,2][:2]
qubit_indices = [0,0,0,0,0][:2]
circuit_gate_angles = [0.,0.,0.,0.,0.][:2]

with program() as rb:
    
    # n_st = declare_stream()
    state_control = declare(int)
    state_target = declare(int)
    state = declare(int)
    # state_st_control = declare_stream()
    # state_st_target = declare_stream()
    # state_st = declare_stream()

    # machine.set_all_fluxes('joint', qubit_control)
    
    gate = declare(int)
    qubit_index = declare(int)
    angle = declare(fixed)
    n = declare(int)
    
    num_lengths = len(node.parameters.circuit_lengths)
    num_circuits_per_length = node.parameters.num_circuits_per_length
    
    x90_id_qua = declare(int, value=x90_id)
    x180_id_qua = declare(int, value=x180_id)
    cz_id_qua = declare(int, value=cz_id)
    rz_id_qua = declare(int, value=rz_id)
    measure_id_qua = declare(int, value=measure_id)
    barrier_id_qua = declare(int, value=barrier_id)
   
    with for_(n, 0, n < n_avg, n + 1):
        # save(n, n_st)
        
        # qubit_control.xy.play('x90')
        # reset_qubits(node, qubit_control, qubit_target, thermalization_time * u.ns)
            
        align() # align(qubit_control.name, qubit_target.name)
        
        with for_each_((gate, qubit_index, angle), ([circuits_as_int, qubit_indices, circuit_gate_angles])):   
            
            with switch_(gate, unsafe=True):
                with case_(cz_id_qua):
                    cz.execute()
                with case_(barrier_id_qua):
                    align() 
                with case_(measure_id_qua):
                    # readout
                    readout_state(qubit_control, state_control)
                    readout_state(qubit_target, state_target)
                    # assign(state, state_control*2 + state_target)
                    assign(state, (Cast.to_int(state_control) << 1) + Cast.to_int(state_target))
                    # save(state, state_st)
                    
                    # reset qubits
                    reset_qubits(node, qubit_control, qubit_target, thermalization_time)
                with case_(rz_id_qua):
                    with if_(qubit_index == 0): # unfavourable becuase expected to genarate gaps from the case_ and the if_
                        qubit_control.xy.frame_rotation(angle)
                    with else_():
                        qubit_target.xy.frame_rotation(angle)
                with case_(x90_id_qua):
                    with if_(qubit_index == 0):
                        qubit_control.xy.play('x90')
                    with elif_(qubit_index == 1):
                        qubit_target.xy.play('x90')
                        
                with case_(x180_id_qua):
                    with if_(qubit_index == 0): # unfavourable becuase expected to genarate gaps from the case_ and the if_
                        qubit_control.xy.play('x180')
                    with else_():
                        qubit_target.xy.play('x180')

            

    # with stream_processing():
    #     n_st.save("n")
    #     for i in range(1): # num qubit pairs
    #         # state_st_control.buffer(num_circuits_per_length).buffer(num_lengths).buffer(n_avg).save(f"state_control{i + 1}")
    #         # state_st_target.buffer(num_circuits_per_length).buffer(num_lengths).buffer(n_avg).save(f"state_target{i + 1}")
    #         state_st.buffer(num_circuits_per_length).buffer(num_lengths).buffer(n_avg).save(f"state{i + 1}")

# %% 
###########################
# Run or Simulate Program #
###########################

if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=1_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, rb, simulation_config)
    job.get_simulated_samples()
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    for i, con in enumerate(samples.keys()):
        plt.plot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
        plt.show()
    
else: #node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(rb)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %%
measurement_axis = {"num_circuits_per_length": np.linspace(1, standardRB.num_circuits_per_length, standardRB.num_circuits_per_length), 
                    "lengths": standardRB.circuit_lengths, "N": np.linspace(1, n_avg, n_avg)}
ds = fetch_results_as_xarray(job.result_handles, [qubit_pair], measurement_axis)
# %% {analysis}

standardRB.plot_with_fidelity(data=ds, num_averages=node.parameters.num_averages)
# %%

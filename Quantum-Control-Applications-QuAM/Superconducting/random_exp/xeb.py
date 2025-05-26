#%%
import numpy as np
from quam_libs.components import QuAM, TransmonPair
from quam_libs.experiments.xeb import (
    XEBConfig,
    XEB,
    backend as fake_backend,
    QUAGate,
)

machine = QuAM.load()
qubits = machine.active_qubits
# Get the relevant QuAM components
readout_qubit_indices = [0,1,2,3,4]  # Indices of the target qubits
readout_qubits = [qubits[i] for i in readout_qubit_indices]
target_qubit_indices = [0,1]  # Indices of the target qubits
target_qubits = [qubits[i] for i in target_qubit_indices]
target_qubit_pairs = [
    qubit_pair
    for qubit_pair in machine.active_qubit_pairs
    if qubit_pair.qubit_control in target_qubits and qubit_pair.qubit_target in target_qubits

]

from qm.qua import frame_rotation_2pi, align, wait
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

def cz_gate(qubit_pair: TransmonPair):
    """
    CZ gate QUA macro: Add your own QUA code here implementing your CZ gate for any given qubit pair
    :param qubit_pair: TransmonPair instance on which to apply the gate
    :return: None
    """
    qubit_pair.align()

    # qubit_pair.qubit_target.xy.play("y90")
    # qubit_pair.gates['Cz'].execute()
    # qubit_pair.qubit_control.xy.play("-y90")

    qp = qubit_pair
    # qp.qubit_target.xy.play("y90")
    # qp.qubit_target.xy.play("x180")
    wait(120 * u.ns)
    qp.gates['Cz'].execute()
    wait(120 * u.ns)
    # qp.qubit_target.xy.play("y90")
    # qp.qubit_target.xy.play("x180")

    qubit_pair.align()


cz_qua = QUAGate("cz", cz_gate)
thermalization_factor = 10

xeb_config = XEBConfig(
    seqs=24, #128, #81,
    # depths=np.arange(1, 600, 24),
    depths=np.arange(1, 32, 2),
    # depths=list(np.arange(1, 91, 4)),
    n_shots=32, #1000, NOTE: The limit is around 64 ??
    readout_qubits=readout_qubits, 
    qubits=target_qubits,
    qubit_pairs=target_qubit_pairs,
    baseline_gate_name="x90",
    gate_set_choice="sw",
    two_qb_gate=cz_qua, #cz_qua, None
    save_dir="xeb_data/QCage_5q4c",
    should_save_data=False, #True,
    generate_new_data=True,
    disjoint_processing=False, #False,
    # reset_method="active",
    # reset_kwargs={"max_tries": 3, "pi_pulse": "x180"},
    reset_method="cooldown", #"active",
    reset_kwargs={"cooldown_time": thermalization_factor * 100000, "max_tries": 3, "pi_pulse": "x180"},
)

print("target_qubits: %s" %[q.name for q in target_qubits]) 
# print("qubit_control: %s" %(qubits[0]@qubits[1]).qubit_control)

print("Number of points: %s" % (xeb_config.seqs * len(xeb_config.depths) * xeb_config.n_shots)) # 2048 is the limit?? 
xeb_runtime = xeb_config.seqs * len(xeb_config.depths) * xeb_config.n_shots / (150 * 27 * 700) * 19.35
print("time required: %s min" % (thermalization_factor * xeb_runtime))

#%%
simulate = False  # Set to True to simulate the experiment with Qiskit Aer instead of running it on the QPU
xeb = XEB(xeb_config, machine=machine)
if simulate:
    job = xeb.simulate(backend=fake_backend)
else:
    job = xeb.run(simulate=False)  # If simulate is False, job is run on the QPU, else pulse output is simulated

#%%
for qubit_pair in target_qubit_pairs: 
    print("qubit_control: %s" %qubit_pair.qubit_control)
    print("qubit_target: %s" %qubit_pair.qubit_target)
    print(qubit_pair.qubit_target.id)

print("Qubits: %s" % ", ".join([q.name for q in target_qubits]))
print("sequences: %s" % xeb_config.seqs)
print("depths: %s" % xeb_config.depths)
print("shots: %s" % xeb_config.n_shots)
print("CZ: %s" %(xeb_config.two_qb_gate))

xeb_config.should_save_data = True #False, True
if xeb_config.should_save_data: print("saving data under: %s" %xeb_config.save_dir)

#%%
job.circuits[15][3].draw("mpl") # job.circuits[seq][depth] 
#%%
job.circuits[7][1].draw("mpl") # job.circuits[seq][depth]
#%%
# 1. Extracting Outputs from QPU (QUA, Measured) 
# 2. Extracting recorded Circuits and perform ideal simulation accordingly on CPU (Expected): 
# 3. Saving data
result = job.result() 
result.plot_state_heatmap()


#%%

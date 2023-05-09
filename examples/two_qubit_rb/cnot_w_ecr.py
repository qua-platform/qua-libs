from qiskit import Aer, QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import numpy as np
from qiskit import *

def add_cnot(circuit, c, t, method=1):

  if method==1:

    # reference CNOT
    circuit.cnot(c, t)

  elif method==2:

    # Use phasedxz unitaries:
    circuit.rz(-np.pi/2, c)
    circuit.rx(np.pi, c)
    circuit.rz(+np.pi/2, c)
    circuit.rx(+np.pi/2, t)
    circuit.ecr(c, t)
    circuit.rz(-np.pi/2, c)

  else:
    print("incorrect CNOT method")

  return circuit


c = 1; t = 0

# Use Aer's AerSimulator
backend = Aer.get_backend('unitary_simulator')

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)

# Add operations to circuit:
circuit = add_cnot(circuit, c, t, method=1)

job = execute(circuit, backend)

# Grab results from the job
result = job.result()

print(result.get_unitary(circuit, 2))

print(circuit)


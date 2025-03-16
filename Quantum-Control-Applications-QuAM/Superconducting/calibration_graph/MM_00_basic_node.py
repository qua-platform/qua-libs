# %% {Imports}
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from qualibrate import QualibrationNode, NodeParameters


# %% Define parameters
class Parameters(NodeParameters):
    qubits: List[str] = ["q1", "q2"]
    noise_factor: float = 0.2


# %% Instantiate node
node = QualibrationNode("basic_node", parameters=Parameters())


# %% Perform actions
frequencies = np.linspace(-1e6, 1e6, 1000)
signal = np.sin(frequencies * 2 * np.pi / 1e6)
signal += node.parameters.noise_factor * np.random.randn(len(frequencies))
fig = plt.figure()
plt.plot(frequencies, signal)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Signal")
plt.title("Basic Node Signal")


# %% Record results
node.results = {"figure": fig, "signal": signal}

# %% Save node results
node.save()

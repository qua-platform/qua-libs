## Running QUAlibrate Nodes

### Node structure

> **_NOTE:_** For the most detailed and up-to-date documentation on calibration nodes, visit the QUAlibrate [documentation](https://qua-platform.github.io/qualibrate/calibration_nodes/).
>
> Qualibrate provides a framework to convert any old calibration script into a calibration **node** to be used within
> a calibration graph, whilst maintaining its ability to be run standalone. The core elements of this framework are as
> follows:

#### Core features

```python
from qualibrate import NodeParameters, QualibrationNode

# 1. Define the set of input parameters relevant for calibration
class Parameters(NodeParameters):
    span: float = 20
    num_points: int = 101

# 2. Instantiate a QualibrationNode with a unique name
node = QualibrationNode(name="my_calibration_node")

# Run your regular calibration code here
...

# 3. Record any relevant output from the calibration
node.results = {...}  # a dictionary with any result data you like (including figures)!

# 4. Save the results
node.save()
```

After executing the node, results will be saved at the `<path_to_your_data_folder>`, as well as being viewable on the
web app.

#### Additional Feature: Interactive calibration

Naturally as part of a calibration node, one would like to _update their QUAM parameters_ according to calibration
results. When using QUAlibrate, you can define **interactive** state-updates to a QUAM as follows:

```python
with node.record_state_updates():
    # Modify the resonance frequency of a qubit
    machine.qubits["q0"].f_01 = 5.1e9
```

This will simply update the values if the script is executed normally. However, if the node is executed through the
QUAlibrate Web App, any changes will be presented as a proposed state update to the user, allowing them to interactively accept or decline the changes based on the measurement outcomes.

### Execution

#### As standalone python scripts

Simply run the script in your favourite IDE!

#### Within Qualibrate

1. Activate your conda environment if you haven't already:

```shell
conda activate qm
```

2. Start the QUAlibrate web-app in the command-line within your conda environment, e.g.,

```shell
qualibrate start
```

3. Open http://localhost:8001/ on your browser:
   ![browser window](../.img/qualibrate_1.png)
4. Select the node you would like to run:
   ![select node](../.img/qualibrate_2.png)
5. Change the input parameters to your liking:
   ![change parameters](../.img/qualibrate_3.png)
6. Press "Run":
   ![change parameters](../.img/qualibrate_4.png)



## Calibration Nodes

The scripts within the `calibration_graph` directory are the building blocks for automated calibration routines. Each script typically performs a specific measurement (e.g., Resonator Spectroscopy, Rabi Oscillations, T1 measurement). They are designed to be run via the QUAlibrate framework, either individually or as part of a larger calibration sequence (graph).

Refer to the `calibration_graph/README.md` for detailed information on the structure and conventions used for these nodes.
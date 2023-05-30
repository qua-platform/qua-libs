# Two-Qubit Randomized Benchmarking

Author: Maximilian Zanner

*Important note: The code in this folder was used for running the experiment on a*
*specifically tailored setup and software environment. When adapting the code to run on your device, make sure to adjust the relevant functions and parameters and contact QM Customer Success!*

## Introduction
Two-Qubit Randomized Benchmarking has become a popular protocol that allows to experimentally quantify the performance of a quantum processor by applying sequences of randomly sampled Clifford gates and measuring the average error rate. Due to its universality it has been implemented in various qubit platforms such as trapped-ions [^1], NMR [^2], spin [^3] and superconducting qubits [^4]. In this use-case example we introduce a possible implementation on the OPX+ using the current version (2023, June) of the generic *TwoQubitRb* class. An updated version can be found int the *py-qua-tools* repository.

[^1]: Knill et al (2008 Phys. Rev. A 77 012307)
[^2]: C A Ryan et al 2009 New J. Phys. 11 013034
[^3]: X. Xue et al Phys. Rev. X 9, 021011
[^4]: A. D. Córcoles et al Phys. Rev. A 87, 030301(R)

## Experimental Setup
<img align="right" width="400" src="https://github.com/maximilianqm/max-dev/blob/main/Quantum-Control-Applications/Superconducting/Multiple%20Flux%20Tunable%20Transmons/Use%20Case%202%20-%20Two%20Qubit%20Randomized%20Benchmarking/setup.png">

The use-case in this example is tailored for a superconducting quantum processor using flux-tunable transmon qubits, where we focus on a subset of two qubits that are capacitively coupled to each other. Single qubit operations are controlled by sending microwave pulses through a xy-line that is capacitively coupled to the individual qubits. The two-qubit gate is implemented by a controlled-Z (CZ) gate utilizing the fast-flux lines to rapidly change the qubit frequencies and the capacitive coupling between both qubits. Part of the optimization protocol for tuning up a CZ gate can be found in the use-case Two-Qubit Gate Optimization.

## Prerequisites
```diff
- OPX+, QOP Version > 20???, QUA Version ???, py-qua-tools > ???}
````
- Calibrated Single Qubit Gates
- Calibrated CZ Gate
- Calibrated Measurement Protocol for 2-State Discrimination for both Qubits

## Implementation in QUA
The following procedure implements Two-Qubit Randomized Benchmarking with the described setup and the *TwoQubitRb* class. The decomposition of the two-qubit unitaries into CZ and single qubit gates is given in Ref. [^5]. 
```diff
- The circuit generation}$$ is done using the *baking* tool from the *py-qua-tools* library. Randomization is done prior to the execution using tableau calculation, also to find the inverse operation. The sequences are passed to the OPX using *input stream*.
```

[^5]: Barends, R. et al. Nature 508, 500–503 (2014)

### The TwoQubitRB Class
```python
rb = TwoQubitRb(config, single_qubit_gate_generator, two_qubit_gate_generators, prep_func, measure_func, verify_generation=True)
```

- **TwoQubitRb**: The class for generating the configuration and running two-qubit randomized benchmarking experiments with the OPX
- **config**: dict – Standard configuration “config” containing the relevant experimental details (e.g. what analog outputs are connected to the xy drive, z flux line, etc.).
- **single_qubit_gate_generator**: A callable used to generate a generic (baked) single qubit gate using a signature similar to phasedXZ
- **two_qubit_gate_generators**: Mapping two qubit gate names to callables used to generate the (baked) gates (needs at least one two-qubit gate). Can contain all two-qubit gates implemented by the user.
- **prep_func**: Callable used to reset the qubits to the |00> state. This function does not use the baking object, and is a proper QUA code macro (e.g. wait() statement or active reset protocol).
- **measure_func**: A callable used to measure the qubits. This function does not use the baking object, and is a proper QUA code macro. Returns a tuple containing the measured values of the two qubits as QUA expressions. The expression must evaluate to a Boolean value. False means |0>, True means |1>. The most significant bit (MSB) is the first qubit. 
verify_generation: bool = False

### Run the 2 QB RB 
The experiment is run by calling the run method of the previously generated program rb.

```python
qmm = QuantumMachinesManager('127.0.0.1',8080)
res = rb.run(qmm, circuit_depths=[1, 2, 3, 4, 5], num_circuits_per_depth=50, num_shots_per_circuit=1000)
```

For running the experiment the user has to specify the following arguments:
- **qmm**: The quantum machine manager instance, on which the 2-Qubit-RB will be executed on.
- **circuit_depths**: Number of consecutive clifford gates (layers) per sequence (not including the inverse, more info on depth: https://qiskit.org/documentation/apidoc/circuit.html).
- **num_circuits_per_depth**: The amount of different circuit randomizations (combination of Cliffords) in each sequence. 
- **num_shots_per_circuit**: The number of repetitions of the same circuit of a depth, e.g. used for averaging.

### Gate Definition
Gate generation is performed using the *baking* class. This class adds to QUA the ability to generate arbitrary waveforms ("baked waveforms") using syntax similar to QUA. 
#### single_qubit_gate_generator
```python
def bake_phased_xz(baker: Baking, q, x, z, a):
    element = f"qubit{q}_xy"
    baker.frame_rotation_2pi(-a, element)
    baker.play("x$drag", element, amp=x)
    baker.frame_rotation_2pi(a + z, element)
```
single_qubit_gate_generator: A callable used to generate a single qubit gate using a signature similar to `phasedXZ`.
Callable arguments:  
- **baking**: The baking object. 
- **qubit**: The qubit number. 
- **x**: The x rotation exponent. 
- **z**: The z rotation exponent. 
- **a**: the axis phase exponent. 
 
#### two_qubit_gate_generators
```python
qubit1_frame_update = 0.23  # example, should be taken from QPU parameters
qubit2_frame_update = 0.12  # example, should be taken from QPU parameters
def bake_cz(baker: Baking, q1, q2):
    q1_xy_element = f"qubit{q1}_xy"
    q2_xy_element = f"qubit{q2}_xy"
    q2_z_element = f"qubit{q2}_z"
   
    baker.play("cz_qubit1_qubit0$rect", q2_z_element)
    baker.align()
    baker.frame_rotation_2pi(qubit1_frame_update, q2_xy_element)
    baker.frame_rotation_2pi(qubit2_frame_update, q1_xy_element)
    baker.align()
```
Mapping one or more two qubit gate names to callables used to generate those gates.
Callable arguments: 
- **baking**: The baking object. 
- **qubit1**: The first qubit number. 
- **qubit2**: The second qubit number. 

### State Preparation
```python
def prep():
    wait(10000)  # thermal preparation
    align()
```
This example of the state preparation simply uses a thermal reset using the QUA wait() command. It is executed between sequences to ensure that the initial state of the qubit pair is |00>. More advanced preparation protocols (e.g. active reset) can be implemented in this macro.


### Measurement

```python
def meas():
    rr0_name = f"qubit0_rr"
    rr1_name = f"qubit1_rr"
    Iq0 = declare(fixed)
    Qq0 = declare(fixed)


    Iq1 = declare(fixed)
    Qq1 = declare(fixed)
   
    measure("readout$rect$rotation", rr0_name, None,
            dual_demod.full("w1", "out1", "w2", "out2", Iq0),
            dual_demod.full("w3", "out1", "w1", "out2", Qq0)
            )
    measure("readout$rect$rotation", rr1_name, None,
            dual_demod.full("w1", "out1", "w2", "out2", Iq1),
            dual_demod.full("w3", "out1", "w1", "out2", Qq1)
            )

    return Iq0 > 0, Iq1 > 0  # example, should be taken from QPU parameters
```    
**measure_func**: A callable used to measure the qubits. This function does not use the baking object, and is a proper QUA code macro. 
Returns a tuple containing the measured values of the two qubits as Qua expressions. The expression must evaluate to a Boolean value. False means |0>, True means |1>. The MSB is the first qubit. 

### Results
<img align="right" width="400" src="https://github.com/maximilianqm/max-dev/blob/main/Quantum-Control-Applications/Superconducting/Multiple%20Flux%20Tunable%20Transmons/Use%20Case%202%20-%20Two%20Qubit%20Randomized%20Benchmarking/results.png">
Plot fidelity as function of depth and plot histograms for each depth

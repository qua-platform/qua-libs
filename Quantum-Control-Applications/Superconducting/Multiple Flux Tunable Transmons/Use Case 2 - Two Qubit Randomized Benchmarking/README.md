# Two-Qubit Randomized Benchmarking

Author: Maximilian Zanner

*Important note: The code in this folder was used for running the experiment on a*
*specifically tailored setup and software environment. When adapting the code to run on your device, make sure to adjust the relevant functions and parameters and contact QM Customer Success!*

## Introduction
Two-Qubit Randomized Benchmarking (RB) has become a popular protocol that allows to experimentally quantify the performance of a quantum processor by applying sequences of randomly sampled Clifford gates and measuring the average error rate. Due to its universality it has been implemented in various qubit platforms such as trapped-ions [^1], NMR [^2], spin [^3] and superconducting qubits [^4]. Two-Qubit RB can be challenging to implement with state-of-the-art control electronics because of the necessity to sample from a large amount of Clifford gates corresponding to the Two-Qubit Clifford group. This Clifford group consists of 11520 operations [^4] and contains the single qubit Clifford operations (576), the CNOT-like class (5184), the iSWAP-like class (5184) and the SWAP-like class (576). In this use-case example we introduce a possible implementation on the OPX+ using the current version (2023, June) of the generic *TwoQubitRb* class. The implementation exploits the [*baking tool*](https://github.com/qua-platform/py-qua-tools/blob/main/qualang_tools/bakery/README.md) in order to generate and invert the sequence before running the QUA program. The QUA program then uses the [*Input Stream*](https://docs.quantum-machines.co/0.1/qm-qua-sdk/docs/Guides/features/?h=declare_input_stream#input-streams) feature to randomly load and execute the pre-compiled sequences. An updated version of the *TwoQubitRb* class can be found in the [*py-qua-tools*](https://github.com/qua-platform/py-qua-tools) repository.

[^1]: Knill et al (2008 Phys. Rev. A 77 012307)
[^2]: C A Ryan et al 2009 New J. Phys. 11 013034
[^3]: X. Xue et al Phys. Rev. X 9, 021011
[^4]: A. D. Córcoles et al Phys. Rev. A 87, 030301(R)

## Experimental Setup
<img align="right" width="400" src="https://github.com/maximilianqm/max-dev/blob/main/Quantum-Control-Applications/Superconducting/Multiple%20Flux%20Tunable%20Transmons/Use%20Case%202%20-%20Two%20Qubit%20Randomized%20Benchmarking/setup.png">

The use-case in this example is tailored for a superconducting quantum processor using flux-tunable transmon qubits, where we focus on a subset of two qubits that are capacitively coupled to each other. Single qubit operations are controlled by sending microwave pulses through a xy-line that is capacitively coupled to the individual qubits. The two-qubit gate is implemented by a controlled-Z (CZ) gate utilizing the fast-flux lines to rapidly change the qubit frequencies and the capacitive coupling between both qubits. Part of the optimization protocol for tuning up a CZ gate can be found in the use-case [Two-Qubit Gate Optimization](https://github.com/qua-platform/qua-libs/tree/2qb-RB-usecase/Quantum-Control-Applications/Superconducting/Multiple%20Flux%20Tunable%20Transmons/Use%20Case%201%20-%20Two%20qubit%20gate%20optimization%20with%20cryoscope).

## Prerequisites
- Calibrated Single Qubit Gates
- Calibrated CZ Gate
- Calibrated Measurement Protocol for 2-State Discrimination for both Qubits

## Quick User Guide
For a quick implementation just clone the repository and edit the file [*two_qubit_rb_example.py*](https://github.com/qua-platform/qua-libs/blob/2qb-RB-usecase/Quantum-Control-Applications/Superconducting/Multiple%20Flux%20Tunable%20Transmons/Use%20Case%202%20-%20Two%20Qubit%20Randomized%20Benchmarking/two_qubit_rb_example.py).

### Single Qubit Gates
The function for the single qubit gates requires that the operation "x" was previously calibrated by the user and corresponds to a pi-pulse on the target qubits. The *amp=x* condition inside the *baker.play* statement allows to scale the amplitude of the pulse. Together with the first *baker.frame_rotation_2pi* it allows the *baker.play* statement to act as X and Y gates by shifting the frame of the control signal, thus realizing rotations around the x- and y-axis. The second *baker.frame_rotation_2pi* resets the frame and additionally allows for rotations around the z-axis.
```python
def bake_phased_xz(baker: Baking, q, x, z, a):
    element = f"qubit{q}_xy"
    baker.frame_rotation_2pi(-a, element)
    baker.play("x", element, amp=x)
    baker.frame_rotation_2pi(a + z, element)
```

### CZ Gate
The use-case is designed for flux-tunable transmon qubits where the qubit-qubit interaction is realized with a direct capacitive coupling. Utilizing this architecture it is possible to realize a flux-tuned |11>-|02> phase gate. An applied flux pulse that tunes the qubits in and out of the |11>−|02> avoided-crossing leads to a conditional phase accumulation. Leaving the system at the avoided-crossing for a specific time maps the state |11〉back into itself but acquires a minus sign in the process. As the computational states are far from being resonant with other transitions their phases evolve trivially and can be corrected using single qubit phase corrections and thus realize the CZ gate. The *baker.play* statement therefore contains a flux pulse that frequency-tunes transmon *q1* in and out of the avoided crossing |11>-|02> , while the *baker.frame_rotation_2pi* statements correct the single qubit phases.

```python
def bake_cz(baker: Baking, q0, q1):
    q0_xy_element = f"qubit{q0}_xy"
    q1_xy_element = f"qubit{q1}_xy"
    q1_z_element = f"qubit{q1}_z"
    baker.play("cz", q1_z_element)
    baker.align()
    baker.frame_rotation_2pi(qubit0_frame_update, q0_xy_element)
    baker.frame_rotation_2pi(qubit1_frame_update, q1_xy_element)
    baker.align()
```

### Initialization
Before each circuit, it is important to implement a initialization protocol to reset the qubits to the ground state. In the example the *prep* function contains a single QUA command *wait* and is called before each circuit execution to assure that the initial state is set to |00>. The time inside the *wait* statement is chosen to be a multiple of the characteristic decay time of the qubits *T1* to leave enough time for the qubit to relax after it has been excited to the excited state |1>. If single shot readout is implemented, it is possible to use active feedback to reset the qubit to the ground state |0> by sending a pi-pulse if the qubit was measured in the excited state |1>.   

```python
T1 = 10000
def prep():
    wait(10*T1)  # thermal preparation
    align()
```

### Measurement
Finally, the measurement is performed at the end of the circuit 
```python
def meas():
    rr0_name = f"qubit{q0}_rr"
    rr1_name = f"qubit{q1}_rr"
    Iq0 = declare(fixed)
    Qq0 = declare(fixed)
    Iq1 = declare(fixed)
    Qq1 = declare(fixed)
    measure("readout", rr0_name, None,
            dual_demod.full("w1", "out1", "w2", "out2", Iq0),
            dual_demod.full("w3", "out1", "w1", "out2", Qq0)
            )
    measure("readout", rr1_name, None,
            dual_demod.full("w1", "out1", "w2", "out2", Iq1),
            dual_demod.full("w3", "out1", "w1", "out2", Qq1)
            )
return Iq0 > 0, Iq1 > 0  # example, should be taken from QPU parameters
```
![image](https://github.com/qua-platform/qua-libs/assets/117653449/1f9e119d-aef3-4593-b2a3-c0ddf3646a66)


----------------------------------------------------------------------------------------------------------------------------------------------------------

## Implementation in QUA
The python program *two_qubit_rb_example.py* implements Two-Qubit Randomized Benchmarking with the described setup and the *TwoQubitRb* class. The decomposition of the two-qubit unitaries into CZ and single qubit gates is given in Ref. [^5]. The circuit generation is done using the *baking* tool from the *py-qua-tools* library. Randomization is done prior to the execution using tableau calculation, also to find the inverse operation. The sequences are passed to the OPX using *input stream*.

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
- **circuit_depths**: Number of consecutive clifford gates (layers) per sequence (not including the inverse.
- **num_circuits_per_depth**: The amount of different circuit randomizations (combination of Cliffords) in each sequence. 
- **num_shots_per_circuit**: The number of repetitions of the same circuit of a depth, e.g. used for averaging.

### Gate Definition
Gate generation is performed using the *baking* class. This class adds to QUA the ability to generate arbitrary waveforms ("baked waveforms") using syntax similar to QUA. 
#### single_qubit_gate_generator
```python
def bake_phased_xz(baker: Baking, q, x, z, a):
    element = f"qubit{q}_xy"
    baker.frame_rotation_2pi(-a, element)
    baker.play("x", element, amp=x)
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
   
    baker.play("cz", q2_z_element)
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
   
    measure("readout", rr0_name, None,
            dual_demod.full("w1", "out1", "w2", "out2", Iq0),
            dual_demod.full("w3", "out1", "w1", "out2", Qq0)
            )
    measure("readout", rr1_name, None,
            dual_demod.full("w1", "out1", "w2", "out2", Iq1),
            dual_demod.full("w3", "out1", "w1", "out2", Qq1)
            )

    return Iq0 > 0, Iq1 > 0  # example, should be taken from QPU parameters
```    
**measure_func**: A callable used to measure the qubits. This function does not use the baking object, and is a proper QUA code macro. 
Returns a tuple containing the measured values of the two qubits as Qua expressions. The expression must evaluate to a Boolean value. False means |0>, True means |1>. The MSB is the first qubit. 

### Results
<img align="center" width="400" src="https://github.com/qua-platform/qua-libs/blob/2qb-RB-usecase/Quantum-Control-Applications/Superconducting/Multiple%20Flux%20Tunable%20Transmons/Use%20Case%202%20-%20Two%20Qubit%20Randomized%20Benchmarking/results.png">

The result object *res* , that is created from the *rb* class by the running the experiment contains an xarray dataset (res.data) with the result data and the specified parameters. It has two implemented functions that easily visualize the fidelity decay with increasing circuit depth

```python
res.plot_fidelity()
```

as well as the resulting histograms for each of the corresponding circuit depths.

```python
res.plot_hist()
```


---
id: index
title: Multiplexed Readout of Multi-state Qubits Using a Neural Network
sidebar_label: Multilevel discriminator with NN
slug: ./
---

# Overview
This program allows for a multiplexed readout of up to 5 qubits using 2 OPXs (**a multiplexed readout of 10 qubits with 2 
OPXs is available from QUA version 0.8 thanks to internal resources optimization**). 
The basic idea is to use a neural network per each resonator to learn the optimal weights 
for state classification of the corresponding qubit.
The program is divided into 3 main stages: 
- Training data generation 
    - prepare the qubits in a combination state, 
      i.e. for 3 qubits a state could be [1,0,2] meaning qubits 1-3 are in states 'e','g','f', respectively.-
    - measure the state of the resonator, in this case a waveguide to which all qubits are coupled.
    - record the incoming raw ADC data
    - repeat the process many times
- Training
    - use the recorded raw ADC data as the input to the neural network
    - each neural network has a structure that effectively performs demodulation
    - learn the optimal weights such that the demodulation leads to good classification
- Measurement
  - at this point we have the optimal weights for demodulation
  - to classify we need to transform the values of 2 demodulations into one of 3 states 
  - we use a small 2 by 3 matrix which is the final layer in our neural network and 
    perform a multiplication in QUA to implement that layer for a final state classification
    
## Configuration

The configuration consists of 4 main parts: the readout resonators elements, the qubits elements, 
the readout pulses, and the preparation pulses.

- Readout resonators: 
    - One needs to define the quantum elements that correspond to the readout resonators
    - In our case all RR elements will be controlled using 'con1' - the measurement 
    operation will be done through 'con1'
      - **For 10 RRs and qubits, the configuration uses 'con1' for the first 5 RRs and 'con2' for the last 5 RRs**
    - **ATTENTION** : All elements **MUST** have the *outputs* section defined as follows:
```python
      'outputs': {  
                'out1': ('con1', 1),  
                'out2': ('con1', 2)  
            }
 ```
    
- Furthermore, each RR needs to define an operation, and a pulse which correspond to the readout
- **ATTENTION** : All elements **MUST** have the same *readout pulse length*, and the same name for the *operation*.
- **ATTENTION** : one must make sure that the component that has a phase $\pi/2$ ahead of the other, 
  will be directed towards 'out1' (ADC 1 on con1). 
  Otherwise, there's a need to change TimeDiffCalibrator (which will be discussed below).    
- Qubits:
  - One need to define the quantum elements that correspond to the qubits
  - In our case all qubits will be controlled using 'con2'
    - **For 10 qubits the configuration defines the first 5 on 'con1' and the last 5 on 'con2'**
  - All elements need to define operations and pulses that correspond to the preparation of
    all 3 states - g,e,f
- Both the RRs and the Qubits need to be **mixed input** since both require IQ components. 
- **From QUA 0.8 each resonator and its corresponding qubit will be put in the same element group
  (assigned the same pulsers) since they are not controlled or measured at overlapping times**

- Readout pulses:
    - The readout pulses should have the same length for all resonators
    - All pulses should be associated to an operation which has the same name for all resonators,
    i.e. if both "rr1" and "rr2" have an operation "readout_op" it will look something as follows:  
```python      
"rr1":{  
      "operations":{   
      "readout_op" : "readout_pulse_1"  
      }}  
       "rr2":{  
      "operations":{   
      "readout_op" : "readout_pulse_2"  
      }} 
``` 

where "readout_pulse_1/2" are the calibrated pulses for a readout.

- Preparation pulses:
    - The preparation pulses should define the calibrated pulses used
    for preparing the different qubits in the different states
    - All qubits need to define the operation something as follows,
      let "qb1" be the qubit 1 quantum element, then:  
```python
      "qb1":{  
      "operations":{  
      "prepare0" : "prepare_pulse_qb1_0"  
      "prepare1" : "prepare_pulse_qb1_1"  
      "prepare2" : "prepare_pulse_qb1_2"  
      }}  
```
where there could be different pulses associated with different qubits. Therefore, applying "prepare1" to "qb1" 
will prepare qubit 1 in the 'e' state.
      

## Program
### NNStateDiscriminator class
The program begins with creating an instance of the class NNStateDiscriminator. 
This creates a discriminator object that eventually will be used for the state estimation.
To create a new discriminator we need the following components:  
- a QuantumMachinesManager instance
- a configuration dictionary (of the structure described above)
- a list with the names of the readout resonators
- a list with the names of the qubits
- a list with the names of the quantum elements used for calibration (further details below)
- a path to a folder where all the data and parameters will be stored

### Generating training data
To be able to estimate the states of the qubits we need to learn the system, 
and for that we need first to generate labeled data. 
To do that we will use the NNStateDiscriminator.generate_training_data() function. 
This function receives as arguments the following components:  
- prepare_qubits - a function which prepares the qubits in the desired state
- readout_op - the name of the readout operation for the readout resonators
- n_avg - the number of times to repeat the preparation and measurement of each state  
    - NOTE: the more noise there's in system the number will need to be larger  
- states - a numpy.array or a list with the states to prepare for the training.
In other words, each row in the array or each element in the list will contain some state of every qubit,
  i.e. [0,2,1,1] means that qubits 1-4 are in states ['g','f','e','e'], respectively.
- wait_time: the time to wait between the preparation and measurement of a state to the next one.
    - NOTE: the time is in multiples of 4ns. It should be long enough to let the system relax to a known state
      (such as the ground state), so the preparation of the next state will be correct.  

The generated data is saved into the given folder as HDF5 files, and depending on the number of states it might be 
broken into several files.

### Training
The training stage is as simple as calling the NNStateDiscriminator.train() function. The training program starts from calibrating 
the time difference(which will be discussed below). Then it loads the data from the files in the path given to the 
discriminator object. Then it trains.  
The training can also have 3 main arguments:
- data_files_idx - describes the indexes of data files to use for the training (from the given path)
- epochs - the number of training epochs. One might need to increase it for a better state estimation
- kernel_initializer - the weights from which to start the training, i.e. random or constant. Some may be better in 
different scenarios. One needs to check which works the best.  

Another possible arguments is to whether to calibrate the time difference - if the setup has changed (i.e. wires of
different length) a new calibration is needed.  
After the training is done the optimal demodulation weights are put into the config and save to a file named 
"optimal_params.pkl". The file contains the configuration with the demodulation weights, and a matrix which describes 
the final layer of the neural network for state classification.

### Measurement
After we have the discriminator with the optimal weights we can use the QUA macro NNStateDiscriminator.measure_state()
to measure the state of all qubits. The estimated state of all qubits is saved into one stream in the order defined in
the list that contains the names of the qubits (NNStateDiscriminator.qubits).  
The measurement function gets a few arguments:
- The readout operation name
- Name of the result stream
- And some qua variables used for the state estimation (can be found in the docstring).  

The measure_state command could be used in any QUA program when a multiplexed readout of all qubits is desired.

### Calibrations

There are two calibrations throughout the program which are done automatically: time calibration and DC offset 
calibration. All calibrations need to be given quantum elements by which to calibrate (calibrate_with parameter). 
For example, given "rr0" the 0'th readout resonator the corresponding controller ('con1') will be calibrated for DC
offset and time difference and the configuration will be updated accordingly. It is recommended to calibrate with a 
group of elements which cover all the controllers, i.e. I would give "rr0" and "rr10" if those use 'con1' and 'con2'.
- NOTE: elements with no defined outputs cannot be calibrated. Also, if more than one element is associated with the 
  same controller the DC offset calibration will happen only for the first one.
#### Time difference calibration
It's used for the training stage in which we want to accurately mimic the OPX's demodulation using a neural network. 
The timings of the demodulation depend on the specific setup, therefore there's a need to calibrate the time difference
between the physical system and the computationl model. This stage is done automatically and the value is stored in the 
NNStateDiscriminator object. 
-  NOTE: one needs to pay attention how the IQ compenets from the mixer after downconversion connect to the ADCs of the 
OPX. The demodulation done in the TimeDiffCalibrator class must match the demodulation done throughout the program
   (with respect to 'out1/2' and IQ components). 
- The way the time calibrator is written now assumes that the component with the phase ahead goes into 'out1'.
  
#### DC offset calibration
There's a DC component in different setups on the ADCs. We need to take that into account in our programs. The DCoffsetCalibrator
class does that automatically. That means the DC offset on the analog inputs is measured automatically for each given 
controller, and the configuration instance is updated accordingly. One has the option to choose whether to calibrate the 
offset when running different parts of the program.
- NOTE: the calibrator assumes that when nothing (zero amplitude pulse) is played the analog inputs (ADC values) should 
  also be zero (up to noise)
#Multiplexed Readout of Multi-state Qubits Using a Neural Network

#Overview
This program allows for a multiplexed readout of up to 5 qubits using 2 OPXs. 
The basic idea is to use a neural network per resonator to learn the optimal weights 
for state classification of the corresponding qubit.
The program is divided into 3 main stages: 
- Training data generation 
    - prepare the qubits in a combination state, 
      i.e for 3 qubits a state could be [1,0,2] meaning qubits 1-3 are in states 'e','g','f', respectively.-
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
    
##Configuration

The configuration consists of 4 main parts: the readout resonators elements, the qubits elements, 
the readout pulses, and the preparation pulses.

- Readout resonators: 
    - One needs to define the quantum elements that correspond to the readout resonators
    - In our case all RR elements will be controlled using 'con1' - the measurement 
    operation will be done through 'con1'
    - **ATTENTION** : All elements **MUST** have the *outputs* section defined as follows:
        - 'outputs': {  
                'out1': ('con1', 1),  
                'out2': ('con1', 2)  
            }
          
    - Furthermore, each RR needs to define an operation, and a pulse which correspond to the readout
        - **ATTENTION** : All elements **MUST** have the same *readout pulse length*, and the same name for the *operation*.
- **ATTENTION** : one must make sure that the component that has a phase $\pi/2$ ahead of the other, 
  will be directed towards 'out1' (ADC 1 on con1). 
  Otherwise, there's a need to change TimeDiffCalibrator (which will be discussed below).    
- Qubits:
  - One need to define the quantum elements that correspond to the qubits
  - In our case all qubits will be controlled using 'con2'
  - All elements need to define operations and pulses that correspond to the preparation of
    all 3 states - g,e,f
- Both the RRs and the Qubits need to be **mixed input** since both require IQ components.  

- Readout pulses:
    - The readout pulses should have the same length for all resonators
    - All pulses should be associated to an operation which has the same name for all resonators,
    i.e if both "rr1" and "rr2" have an operation "readout_op" it will look something as follows:  
      "rr1":{  
      "operations":{   
      "readout_op" : "readout_pulse_1"  
      }}  
       "rr2":{  
      "operations":{   
      "readout_op" : "readout_pulse_2"  
      }}  
      where "readout_pulse_1/2" are the calibrated pulses for readout.

- Preparation pulses:
    - The preparation pulses should define the calibrated pulses used
    for preparing the different qubits in the different states
    - All qubits need to define the operation something as follows,
      let "qb1" be the qubit 1 quantum element, then:  
      "qb1":{  
      "operations":{  
      "prepare0" : "prepare_pulse_qb1_0"  
      "prepare1" : "prepare_pulse_qb1_1"  
      "prepare2" : "prepare_pulse_qb1_2"  
      }}  
      where there could be different pulses associated with different qubits.
      

##Program

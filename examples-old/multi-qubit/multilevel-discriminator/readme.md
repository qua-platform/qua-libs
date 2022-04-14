---
id: index
title: Multilevel discriminator
sidebar_label: Multilevel discriminator
slug: ./
---
A basic requirement for Quantum computation is the ability to tell the state of the qubit.
The state of the qubit can be determined by a measurement of the readout resonator reponse, since the response is dependant on the state of the qubit.
To be able to distinguish between the qubit's state one needs to gain information about the resonator system.
Here we assume that we already know how to prepare the qubit in different states and also the readout resonance frequency.
Having that information we proceed to construct a state discriminator based on the Maximum likelihood estimation method.
The code is in principle general and can be used for any number of states, up to physical limitations.

## Config
The configuration defines two elements `rr1a` the readout resonator and `qb1a` the associated qubit/qudit.
The OPX is connected to a mixer via two analog output channels of the OPX, 
numbered 1 and 2. We also specify the LO frequency received by the mixer using the `lo_frequency` field of the 
`mixInputs` dictionary, and a mixer correction matrix using the `mixer` field.

The `qb1a` qubit element defines 3 operations, one for each of the multilevel qubit states. 
Each operation prepares the corresponding qudit state. 
These operations are set by the user and require knowledge about how to manipulate the qubit.

The `rr1a` readout resonator element defines 4 operations, the `readout_pulse` to measure the pulse 
reflected from the readout resonator.
The other 3 operations are auxiliary and are used to simulate the response using the loopback interface.
We also define both I and Q components used to measure the reflected microwave signal.
Note that the `time_of_flight` and `smearing` parameters must be defined to perform a measurement.

## Program
The program is divided into two phases: the training and the testing.
In the training phase we try to find the response of the resonator in each of the qubit's states,
and create a discriminator object to be able to distinguish between the states. In the testing phase we check how well 
discriminator performs.

There are two nested loops, the outer one loops over the states and the inner one repeats the same measurment multiple times.
Inside the nested loops of `training_program` each cycle consists of a `play` command that prepares the qubit in the desired state 
and a `measure` command that measure the readout response, and demodulates the signal 4 times, twice for each OPX input,
corresponding to the I and Q components.
```python

def prepare_state(state,qe):
    if state==0:
        pass # do nothing
    if state==1:
        play("pi",qe)
        
for state in states:
    prepare_state(state,'qubit')
    measure("readout", "rr", "adc", 
          demod.full("integW_cos", I1, "out1"),
          demod.full("integW_sin", Q1, "out1"),
          demod.full("integW_cos", I2, "out2"),
          demod.full("integW_sin", Q2, "out2"))
    assign(I, I1 + Q2)
    assign(Q, -Q1 + I2)
    save(I, 'I')
    save(Q, 'Q')
```
The `training_program` results are processed by the `train` function of the `StateDiscriminator` class.
There we downconvert the reflected signal, extract the waveform and average it.
Using that we update the integration weights to be used in future measurments, according to the maximum likelihood principle.

## Post Processing
After we have trained the discriminator weights, we measure the state of the qubit using the `test_program`.
Again we loop over the states and repeat each measurement multiple times and save the results. 

The results from `discriminator.measure_state` are already inside an array of integers (0-2 for 3 states) 
that represent the measured state. To check how well the discriminator does we count how many of the measured states
are the same as the prepared states. Finally, we represent the data in a confusion matrix. 

## Script

[download script](multilevel_discriminator.py)

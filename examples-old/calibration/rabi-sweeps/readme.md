---
id: index
title: Rabi sweeps
sidebar_label: Rabi sweeps
slug: ./
---

One- and two-dimensional Rabi sweeps are some of the most basic and important qubit characterization protocols. 
The examples included here show how QUA is used to generate time-Rabi and power-Rabi pulse sequences. 
A two-dimensional (time and power) scan can just as easily be performed and an example is included. 
These examples show several features of the QUA language: looping constructs, dynamical stretching of pulse duration 
and dynamical stretching of pulse amplitude. By *dynamical*, we mean that the factors by 
which stretching is performed are QUA variables which are executed in hardware and can therefore 
be calculated in script run-time, included with feedback from the system (though this is not the case here). 

#  Description of experiments

## Introduction

The OPX device is conceived to ensure the user has an optimal control over the quantum system he wants to work on. A prior and critical step for this control is to perform a set of calibration procedures, which ensure that the control sequence sent by the OPX on the quantum system, e.g a superconducting qubit coupled to a readout resonator, does modify its state according to what is originally desired. 

The Power Rabi experiment is one type of those calibration procedures. The idea is to find the right parameter tuning (controllable by the user) leading to the execution of a particular single qubit gate. 

In this example, we want to find the required amplitude of the pulse that will produce a $$\pi$$-rotation around the x-axis, i.e a NOT gate.

## Review of few theoretical elements
A single qubit gate can be visualized as a rotation of a defined angle $$\theta$$  around an arbitrary axis on the Bloch-Sphere. On the hardware, this single qubit gate translates itself in a particular pulse sequence involving specific parameters, such as the pulse envelope (i.e its shape), its amplitude, its frequency and its duration. 

We know that an arbitrary qubit rotation can be decomposed into a subset of rotations around the main axes. One is interested here into figuring out which parameters we must use in order to achieve elementary rotation such as $$\pi$$-pulse around the x-axis. This pulse maps the state $$|0\rangle$$ to the state $$|1\rangle$$ and vice versa. 

We can write the pulse signal sent to the qubit as follows :
$$s(t)=A(t)\cos(\omega_dt+\phi)$$
where $$A(t)$$ is the amplitude of the pulse, which is time dependent since it is directly related to the pulse shape. When the frequency of the drive $$\omega_d$$ is set to the resonant frequency of the qubit $$\omega_q$$, we remove the contribution related to the detuning ($$\omega_d-\omega_q$$) that allows rotations along the z-axis, leaving us with qubit rotations in the x-y plane (the rotation vector will be defined accordingly to the value of $$\phi$$). 

To perform a rotation around the x-axis, we set the phase to be $$0$$, and the rotation angle is determined by the following equation :
$$\theta=\displaystyle\int_{t_0}^{t_0+\tau}A(t)dt$$.

## Power-Rabi experiment
In a typical Power Rabi Oscillations experiment, the shape and duration of the pulse $$A(t)$$ are fixed (e.g. a 20-nanosecond gaussian pulse) and only its amplitude is varied in order to get different rotation angles θ. The experiment is performed by repeating the following basic sequence:
1. Initialize the qubit to the ground state, $$|0\rangle$$.

2. Apply a pulse with amplitude $$a$$ (e.g. $$A(\tau)$$ is a gaussian shaped pulse with peak amplitude $$a$$ and fixed duration $$\tau$$), which rotates the qubit by $$\theta= a\times \tau\times e^{-\tau^2/2\sigma^2}$$ so that the qubit is in the state
$$|\psi\rangle=\cos(\theta_a)|0\rangle+\sin(\theta_a)e^{i\phi}|1\rangle$$
3. Apply a resonant pulse to the readout resonator coupled to the qubit, and from the phase of the reflected pulse, deduce the state of the qubit.

This basic sequence is repeated in the program for a series of amplitudes (i.e., many values $$a$$), where for each amplitude, we do a sampling of N identical sequences. This measurement sampling is required due to the state collapse, inducing the need to obtain statistics representing the probability of measuring the state $$|1\rangle$$ which can be written as : 
$$P_{|1\rangle(a)}=|\sin^2{\theta_a}|$$

```python
with for_(Nrep, 0, Nrep < N_max, Nrep + 1):  # Do 10 times the experiment
    with for_(
        a, 0.00, a < a_max - da / 2, a + da
    ):  # Sweep from 0 to 0.7 V the amplitude
        play(
            "gauss_pulse" * amp(a), "qubit"
        )  # Modulate the Gaussian pulse with the varying amplitude a
        align("qubit", "RR")
        measure("meas_pulse", "RR", None, ("integW1", I), ("integW2", Q))
```

Once we obtain the results, we can plot the probability above in terms of the amplitude $$a$$, and find out for which amplitude this probability reaches one. This amplitude is the one allowing us to finally configure our NOT gate.

## Time-Rabi experiment
This experiment is very similar to the previous one, but instead of sweeping the amplitudes to find the desired rotation angle, we set it at a fixed value and proceed a sweeping of the pulse duration.

```python
with for_(
        Nrep, 0, Nrep < N_max, Nrep + 1
    ):  # Do a 100 times the experiment to obtain statistics
        with for_(
            t, t_min, t <= t_max, t + dt
        ):  # Sweep from 0 to 100 *4 ns the pulse duration
            play("gauss_pulse", "qubit", duration=t)
            align("qubit", "RR")
            measure("meas_pulse", "RR", None, ("integW1", I), ("integW2", Q))
```

We hence do reproduce the same experiment as before, but looking for the maximum probability of measuring the $$|1\rangle$$ in terms of the pulse duration $$\tau$$.

## Describing the QUA program
### The configuration file

The configuration file is architectured around the following items :

- controllers :
We define the outputs and inputs of the OPX device, which will be of use for the experiment. In this case, we have two analog outputs for the qubit, and two others for its coupled readout resonator. We add an analog input which is the channel where will be sampled out the analog results of the readout operation.
- elements :
This defines the set of essential components of the quantum system interacting with the OPX. In this case, the two elements are the qubit and the coupled readout resonator. 
Here are specified the main characteristics of the element, such as its resonant frequency, its associated operations (i.e the operations the OPX can apply on the element).
- pulses : 
A description of the doable pulses introduced in the elements. Here is provided a description of the default pulse duration (length parameter), the associated waveform (which can be taken from an arbitrary array), the type of operation (e.g control or measurement)
- waveforms : 
Specification of the pulse shape based on pre-built arrays (either by the user in case the shape is arbitrary, or constant pulse otherwise)
- Integration weights :
Describe the demodulation process of the data 


### The QUA program
The structure of the program is very similar for both Time and Power Rabi experiments. The only difference is the using of the QUA for_ loop, which is used for tuning either the amplitude or the duration for the Power and Time Rabi experiments respectively. 
Once the pulse is applied on the qubit, we align the readout resonator and the qubit to launch a measurement command on the resonator, to retrieve its I & Q components.

Once the program is written, what we would do on a usual basis is to execute the program. However, in our case we do not have access to a real quantum system, we hence use the LoopbackInterface command, which allows to pick an output signal of the OPX and plug it in the input channel to replace the usual measurement done on the resonator.

Finally, the save command allows the saving of the data from the OPX into a Python structure reachable by the computer. We can then process the data using various tools to produce the corresponding graph of the Rabi oscillations, allowing the determination of optimal parameters for the desired X gate.


## Scripts

[download power-rabi script](Power_Rabi_Exp.py)

[download time-rabi script](Time_Rabi_Exp.py)

[download time-power-rabi script](Time_Power_Rabi_Exp.py)
  
 

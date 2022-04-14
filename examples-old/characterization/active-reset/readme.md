---
title: Active reset
sidebar_label: Active reset
slug: ./
id: index
---

The active reset procedure is a good example of how the OPX can use feedback from
a measurement to feed-forward the state of a qubit. 
It tests the qubit state by performing a measurement and comparing the demodulated 
and integrated in-phase signal to a threshold value. It then conditionally plays a 
$$\pi$$ pulse on that qubit. 

## Introduction
The idea behind active reset is to perform a single-shot readout of the qubit state and perform a classically conditioned $$\pi$$-pulse (assumed to be priorly calibrated via a Power Rabi experiment for example),
that is applying the pulse only if the readout yields a collapse of the qubit state into state $$|1\rangle$$.
The challenge here is to perform this conditional pulse sufficiently fast after the collapse to avoid any decoherence 
effect that would move the system from the excited state to another position on the Bloch sphere,
leading to an operation that would not accurately reset the system in the ground state.

This is where ultra-low latency and real time feedback of the OPX become useful. In a traditional setting, one could wait a duration equivalent to the relaxation time of the qubit, that is when the qubit would eventually move back to the ground state due to energy decay (if the quantum protocol is to be repeated multiple times, the total wait duration can become significant). A second option would be the retrieval of the point in IQ plane after demodulation and averaging to perform state discrimination once the information has been sent back to the user computer, 
and apply upon this classically processed result, the conditional pulse to flip the qubit. This second option can also be very time consuming since we would need to deal with the latency due to the signal processing that would be done back and forth with the user computer. 

## The solution of real time feedback
With the FPGA structure enabling classical flow control, such as *if* statements, it is possible to directly compare values of I and Q 
retrieved from the readout to the coefficients drawing the line separating the IQ plane for state discrimination (we would introduce a threshold in the QUA program), 
and apply the conditional pulse based on the real time evaluation of the resulting boolean of this comparison.

More specifically, once the OPX is provided in memory the necessary variables to perform state discrimination, 
it is able to do the signal processing in real-time and send back the conditional pulse, without having to send any information
to the user computer, avoiding a significant time waste for the realization of multiple experiments.
## Config
Configuration file is set according to the description of a superconducting qubit coupled to a readout resonator.
We assume that a few pulses have already been calibrated (for both quantum elements), including the $$\pi$$ pulse to be conditionally played upon the measurement result of the readout resonator.
A more specific description of the configuration file for this setup can be found here : https://docs.qualang.io/libs/examples/characterization/T1/superconducting-qubits/

## Program 
The program is straightforward and is made of two commands, a conditional play statement, and a measurement. 
This is repeated $$n$$ times for getting statistics. This allows an assessment of the two blobs that would be formed by the readout for the two computational basis states in the IQ plane.

```python
  with for_(n, 0, n < 1000, n + 1):
        play("pi", "qubit", condition=I > th)
        align("qubit", "rr")
        measure(
            "readout", "rr", None, demod.full("integW1", I), demod.full("integW2", Q)
        )
```

   
## Post-processing

Post-processing consists in retrieving the data saved via stream_processing, and to plot the data to identify the blobs described earlier.

## Script

[download script](active_reset.py)

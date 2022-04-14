---
id: index
title: Multiplexed readout
sidebar_label: Multiplexed readout
slug: ./
---

Often due to physical limitations we want to have the ability to use one transmission line or one cavity
to interact with multiple qubits with different resonance frequencies. A common technique used in communication systems
is to use FDM (Frequency division multiplexing) where numerous signals with different central frequency 
are combined on a single composite signal which carries all the information. It's made possible by the simple fact that 
pure signals with different frequencies are orthogonal. A requirement for the protocol is to divide the bandwidth for 
the desired signals such that there's no overlap in the frequency domain. 

## Config
We define three quantum elements `rr1,rr2,rr3` corresponding to three readout resonators. 
Each `rr` is defined with its own resonance frequency, these should be distant enough to enable the FDM protocol.

We also define a `readout` operation for all the resonators. Notice that the input/output ports for all `rr` elements
are the same, and this is because we 'communicate' with them through the same transmission line.

For simulation purposes the readout pulses for each element is different, 
and simulate g,e,f states using the Loopback Interface.

## Program
Firstly, currently it's only possible to do up to 10 parallel demodulations per device, therefore we readout up to 2 resonators
at the same time, because each requires 4 real demodulations. 
The code is written generally to allow a readout of an arbitrary number of resonators.

To begin the program we `align` all the resonators, which ensure a simultaneous measurement of all elements. 
Next, we use the `measure` command in a for loop to measure all the rr's.
Then, we `wait` on all elements and let the resonator/transmission line relax.
Finally, we save the IQ components for each resonator to its corresponding variable, 
i.e measurement from `rr2` is saved to variable `I2` and `Q2`.  

```python
with for_(n, 1, n < 500, n + 1):
      align(*["rr" + str(i) for i in range(1, rr_num + 1)])
      for i in range(rr_num):
          measure("readout_pulse","rr" + str(i + 1),"adc",
              demod.full("integW_cos", I1[i], "out1"),
              demod.full("integW_sin", Q1[i], "out1"),
              demod.full("integW_cos", I2[i], "out2"),
              demod.full("integW_sin", Q2[i], "out2"),
          )

      wait(wait_time, *["rr" + str(i) for i in range(1, rr_num + 1)])
      for i in range(rr_num):
          assign(I[i], I1[i] + Q2[i])
          assign(Q[i], -Q1[i] + I2[i])
          save(I[i], "I" + str(i + 1))
          save(Q[i], "Q" + str(i + 1))
```
## Post Processing

For illustration purposes, we fetch the results of each resonator and plot the IQ diagram.
Since in our example we use a ground state pulse for the `rr1` and excited state pulse for `rr2`
we expect to get one 'blob' for each resonator. 
We can verify the the frequency multiplexing worked since the blobs are well separated as excpected for different qubit state.

## Script


[download script](multiplexed_readout.py)

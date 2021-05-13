---
id: index
title: Resonator spectroscopy
sidebar_label: Resonator spectroscopy
slug: ./
---

To be able to properly interact with a superconducting quantum system 
one needs to know the resonance frequency of the readout resonator that is coupled to the qubit.
To find out the needed frequency we perform a straight forward spectroscopy routine.

## Config
The configuration defines the quantum element `rr` the readout resonator.
We define 2 inputs, one for the `I` component and one for the `Q` component.
Furthermore, we define two operations `readout` and `long_readout`, associated with the respective pulses.
Both pulses define a constant signal on the `I` component and zero on the `Q` component. 
That is because we are just interested in the resonance frequency so we care only about the magnitude of the signal.

## Program
The program `resonator_spectroscopy` consists of an outer averaging loop and an inner scanning loop.
The inner loop scans a range of frequencies and in each cycle changes the frequency using the `update_frequency` command,
and then measures the readout resonator using the `measure` command.
In between measurements we also use the `wait` command to let the resonator relax to its vacuum state. 
For the `wait` command one needs to specify a time period in `ns` during which all the specified elements receive zero signal.
In the `stream_processing` block we save the incoming stream of data into a `buffer`. 
A buffer is an array of a certain size that are used to store the incoming data in a shape of the buffer.
Here we create a `buffer` of size 100 with `buffer(100)` command, to store the results from each of the scanned frequencies.
While saving the results to the `buffer` we also keep a running average using `average()`.
Such that at the end we'll have an array cell for each frequency and the averaged value 
of the I,Q response at that frequency.

> ⚠ Note that the buffer output anything only when full, 
>i.e if the buffer of size 100 but one only saves 70 values it will be empty at the fetching stage.

```python
with for_(n, 0, n < 1000, n + 1):
        with for_(f, 1e6, f < 100.5e6, f + 1e6):
            wait(100, "rr")  # wait 100 clock cycles (4microS) for letting resonator relax to vacuum
            update_frequency("rr", f)
            measure(
                "long_readout","rr",None,
                demod.full("long_integW1", I, "out1"),
                demod.full("long_integW2", Q, "out1"),
            )
            save(I, I_stream)
            save(Q, Q_stream)
```
## Post Processing
No post processing provided. 
One needs to use the extracted I,Q values to determine the resonance frequency by the response spectrum.

## Script

[download script](resonator_spectroscopy.py)

---
id: index
title: Qubit spectroscopy
sidebar_label: Qubit spectroscopy
slug: ./
---

An important step in controlling a qubit is figuring out its resonance frequency.
Here we perform a spectroscopy to do exactly that.

## Config
The configuration defines the quantum element `rr` the readout resonator, and `qubit` the one we're measuring.
We define the 2 inputs, one for the `I` component and one for the `Q` component for both quantum elements.
Furthermore, we define one operation for the `qubit` - the `saturation` and the corresponding pulse `saturation_pulse`. 
The `saturation_pulse` is defined for several T1 lengths to ensure the qubit's response. 
In addition, we define the `readout` operation on the readout resonator, to measure the qubit's response using the `rr`
element. The `readout_pulse` pulse is defined as non-zero on the I component only as we're intrested just in the magnitude
of the response.

## Program
The program `qubit_spectroscopy` consists out of two loops. The outer used for averaging
and the inner using for a frequency scan. We define the range of fequencies to scan using python variables and then loop
over those with a qua `for_` loop. In each cycle we update the qubit's frequency that we want to examine and play a 
saturation pulse to the qubit. Then, we use the `align` command and wait for the saturation pulse to be done.
Afterwards, we measure the readout resonator and save the IQ components.

```python
with for_(n, 0, n < 10, n + 1):
    with for_(f, f_min, f < f_max, f + df):
        update_frequency("qubit", f)
        play("saturation", "qubit")

        align("qubit", "rr")
        measure(
            "readout",
            "rr",
            None,
            demod.full("integW1", I),
            demod.full("integW2", Q),
        )

        save(I, "I")
        save(Q, "Q")
```
## Post Processing
No post processing provided. 
One needs to use the extracted I,Q values to determine the resonance frequency by the response spectrum.

## Script


[download script](qubit_spectroscopy.py)

---
id: index
title: A method to sweep the gaussian with high temporal resolution
sidebar_label: High res Rabi sweeps
slug: ./
---

This example showcases how it is possible to improve the usual 4ns resolution to 2ns or 1ns.
The usual resolution limit comes from the clock cycle: each clock cycle is 4ns. Therefore, extending the pulse by 1 clock cycle increases the pulse duration by 4ns.

The minimum pulse duration is 4 clock cycles, which correspond to 16ns.
Here, we define a 8ns long gaussian, and we pad it with zeros in order to extend it to 16ns.
```python
def delayed_gauss(amp, length, sigma):
    gauss_arg = np.linspace(-sigma, sigma, length)
    delay = 16 - length - 4
    if delay < 0:
        return amp * np.exp(-(gauss_arg ** 2) / 2)

    return np.r_[np.zeros(delay), amp * np.exp(-(gauss_arg ** 2) / 2), np.zeros(4)]
```

In this case, each time we stretch the pulse by one clock cycle, we only stretch the gaussian by half a clock cycle, or by 2 ns.
We also define a 4ns long gaussian, which gives us a 1ns resolution.
```python
gauss_wf_4ns = delayed_gauss(0.2, 4, 2)
with for_(
    r, 0, r < n_repeats, r + 1
):  # Do a n_repeats times the experiment to obtain statistics
    with for_(
        t, t_start, t <= t_max, t + dt
    ):  # Sweep the pulse duration from t_start to t_max
        play(f"gauss_pulse_4ns_res", "qubit", duration=t)
```
The example here only construct the pulses, plays them, and plots the simulated outputs. It does not perform the readout part which can be found in the normal Time Rabi example.

## Script

[download high-res script](high-res-gaussian.py)

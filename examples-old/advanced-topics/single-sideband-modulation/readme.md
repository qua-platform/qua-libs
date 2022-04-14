---
title: Single-sideband modulation
sidebar_label: SSB
slug: ./
id: index
---

Although pulse generation in QUA is arguably built around the concept of 
amplitude modulation (AM), the system is perfectly capable of performing frequency 
modulation. This is a powerful technique which can, for example be used to 
very flexibly generate a multi-tone signal. In this example, we show how a very compact,
100 point arbitrary waveform vector can be used to describe many frequencies 
spaced at a narrow bandwidth around an intermediate frequency (IF). 

![ssb_example](SSB.png)


This is done by using an arbitrary waveform made from a combination of slow oscillating waveforms that is up-converted. 

The following code section is used to calculate the waveforms: 

```python 
t = np.linspace(0, pulseDuration, nSamples)

freqs = np.linspace(1, 4, 15).tolist()
phases = np.zeros_like(freqs).tolist()
amps = np.ones_like(phases).tolist()
m = np.sum(
    list(
        map(
            lambda a: a[2] * np.sin(2 * pi * a[0] * 1e6 * t + a[1]),
            zip(freqs, phases, amps),
        )
    ),
    0,
)
m = m / max(m) / 2
m = m.tolist()
mc = signal.hilbert(m)
wf1 = np.real(mc)
wf2 = np.imag(mc)

```

You can then set the sampling rate if you want to generate a slow modulation rather than 1 sample per nanosecond 

```python
config["pulses"]["ssbPulse"]["length"] = len(wf1) * (1e9 / samplingRate)
config["waveforms"]["wf1"]["samples"] = wf1
config["waveforms"]["wf1"]["sampling_rate"] = samplingRate
config["waveforms"]["wf2"]["samples"] = wf2
config["waveforms"]["wf2"]["sampling_rate"] = samplingRate
```

More details on the mathematical formalism is discussed in the jupyter notebook in the single-sideband-modulation folder on github. 

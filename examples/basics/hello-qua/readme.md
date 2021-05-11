---
title: Hello QUA
sidebar_label: Hello QUA
slug: ./
id: index
---

This is the most basic introductory script in QUA.
It plays a constant amplitude pulse to output #1 
of the OPX and displays the waveform.
  
## Config
We begin the config file by specifying the name and type of the controller
and the outputs used.

```python
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},
            },
        }
    }
```


We then declare one quantum element called `qe1`.
The quantum element has one operation called `playOp`
The `playOp` operation plays a constant amplitude pulse
called `constPulse` which has default duration 1 $\mu S$, 
a frequency of 5 MHz and an amplitude of 0.4V (peak-to-peak).

```python
    "elements": {
        "qe1": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": 5e6,
            "operations": {
                "playOp": "constPulse",
            },
        },
    },
    "pulses": {
        "constPulse": {
            "operation": "control",
            "length": 1000,  # in ns
            "waveforms": {"single": "const_wf"},
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
    }
```

## Program 

The QUA program simply calls the play command.

```python
with program() as prog:
   play("playOp", "qe1")
```

We open a quantum machine with the config file and run the program on the simulator for 1000 clock cycles (4000 ns).

```python
QMm = QuantumMachinesManager()
QM1 = QMm.open_qm(config)
job = QM1.simulate(prog, SimulationConfig(int(1000)))
```

## Post processing

We plot the simulated samples to observe a single cycle of the waveform (5 MHz for 1 micro-second).   

```python
samples = job.get_simulated_samples()
samples.con1.plot()
```

## Sample output

![Hello qua signal](hello_qua.png "Hello qua")

Note the delay (of approx 200ns) between the start of simulation and the start of the pulse.
This delay is faithful to the one produced by executing the QUA program on real hardware.


## Script

[download script](hello_qua.py)

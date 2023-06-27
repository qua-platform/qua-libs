---
title: Introduction to QuAM Rapid Prototyping
sidebar_label: Intro to quam rapid prototyping
slug: ./
id: index
---

This example shows how to build the OPX configuration from a user-defined structure describing the state of the quantum 
system using the QuAM Rapid Prototyping. The QuAM SDK can then be used to easily set and get parameters defined in this 
structure in the calibration scripts.

## Introduction
As most of you know, the OPX expects a python dictionary containing all the information regarding the experiment to conduct 
(wiring, waveforms, elements, frequency, integration weights...). The main drawback is that it has a very rigid structure as shown below. 

```python
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {...},
            "digital_outputs": {},
            "analog_inputs": {...},
        },
    },
    "elements": {
        "qubit": {            
            "mixInputs": {...},
            "intermediate_frequency": qubit_IF,
            "operations": {...},
        },
        "resonator": {...},
    },
    "pulses": {...},
    "waveforms": {...},
    "digital_waveforms": {...},
    "integration_weights": {...},
    "mixers": {...},
}
```

This structure is often good enough when handling a small number of elements/qubits, or when investigating new physics
that require to change the elements and waveforms regularly.

However, when going towards calibrating multi-qubit chips in a semi-automatic way, one may want to have the ability to define 
his/her own structure describing the state of the system and containing the different calibration parameters.

The QuAM Rapid Prototyping is a tool that has been developed to enable the user to easily define his/her own structure 
stored in a .json file and create a python class out of it to seamlessly set and get the different entries directly 
from the calibration scripts.

A detailed example focusing on calibrating two flux-tunable transmons can be found [here]().

## Getting started

The configuration for this example is included, but irrelevant as no pulses are 
played to any output and no data is read in. 

## Defining the state of the system

The first step is to define the desired structure describing the state of your quantum system.
This can be done by writing it as a python dictionary as shown in [state.py](state.py).

The structure is extremely flexible and below is an example showing how it can look like.
Note that here all the keys are defined by the user and some of them are not actually used by the OPX such as T1 and T2 for instance.
This show that you can use it to store the qubit parameters and keep track of them over time and experimental conditions.

```python
state = {
    "network": {"qop_ip": "127.0.0.1", "qop_port": 80, "save_dir": ""},
    "local_oscillators": {
        "qubits": [{"freq": 3.3e9, "power": 18}],
        "readout": [{"freq": 6.5e9, "power": 15}],
    },
    "qubits": [
        {
            "xy": {
                "f_01": 3.52e9,
                "anharmonicity": 250e6,
                "drag_coefficient": 0.0,
                "ac_stark_detuning": 0.0,
                "pi_length": 40,
                "pi_amp": 0.124,
                "wiring": {
                    "I": 1,
                    "Q": 2,
                    "mixer_correction": {"offset_I": 0.01, "offset_Q": -0.041, "gain": 0.015, "phase": -0.0236},
                },
            },
            "z": {
                "wiring": {
                    "port": 7,
                    "filter": {"iir_taps": [], "fir_taps": []},
                },
                "flux_pulse_length": 16,
                "flux_pulse_amp": 0.175,
                "max_frequency_point": 0.0,
                "iswap": {
                    "length": 16,
                    "level": 0.075,
                },
                "cz": {
                    "length": 16,
                    "level": 0.075,
                },
            },
            "ge_threshold": 0.0,
            "T1": 1230,
            "T2": 123,
        },
    ],
    "resonators": [
        {
            "f_res": 6.3e9,
            "f_opt": 6.3e9,
            "depletion_time": 10_000,
            "readout_pulse_length": 1_000,
            "readout_pulse_amp": 0.05,
            "rotation_angle": 0.0,
            "wiring": {
                "I": 5,
                "Q": 6,
                "mixer_correction": {"offset_I": 0.01, "offset_Q": -0.041, "gain": 0.015, "phase": -0.0236},
            },
        },
    ],
    "crosstalk": {
        "flux": {"dc": [[0.0, 0.0], [0.0, 0.0]], "fast_flux": [[0.0, 0.0], [0.0, 0.0]]},
        "rf": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    },
    "global_parameters": {
        "time_of_flight": 24,
        "downconversion_offset_I": 0.0,
        "downconversion_offset_Q": 0.0,
    },
}
```

One dimensional (1D) arrays of complex objects are supported, but two-dimensional and higher arrays of complex objects are not.

Valid 1D array of complex objects (individual qubits with key "freq") is
```python
"qubits" : [
    {"freq":1.23},
    {"freq":1.25},
    ...
]
```
Invalid array (2D) would be
```python
"qubits" : [[
    {"freq":1.23},
    {"freq":1.25},
    ...
    ],
    [
    {"freq":1.23},
    {"freq":1.25},
    ...
    ]
]
```
On the other hand, any dimensional array of simple values (int, float, str and bool) are supported, e.g.
```python
"crosstalk_matrix":[
    [0.98, 0.12, 0.11]
    [0.24, 0.91, 0.15]
    [0.09, 0.13, 0.89]
]
```

## Creating the machine class

Once the state is defined, one just needs to import the QuAM SDK and call the quamconstructor to construct the QuAM (Quantum Abstract Machine) as shown below.
```python
import quam_sdk.constructor

state = {
    "network": {},
    "local_oscillators": {},
    "qubits": [],
    "resonators": [],
    "crosstalk": {},
    "global_parameters": {},
}

quam_sdk.constructor.quamConstructor(state)
```
It will create a file called ``quam.py`` containing the Python class built from the defined state stored in a json file 
called `"quam_bootstrap_state.json"`.
As shown in the next section, this class can then be accessed by importing it from quam: ``from quam import QuAM``

Note that it is also possible to directly define the state in a json file `"quam_bootstrap_state.json"` and then 
construct the QuAM directly from it.
```python
import quam_sdk.constructor

quam_sdk.constructor.quamConstructor(
    "quam_bootstrap_state.json",
    flat_data=False
)
```

## Building the configuration
Now that the state and QuAM are defined, we can build the configuration that the OPX expects by simply filling its entries, 
that are now placeholders, with the values from our structure. 
The snippet below shows only a subset of the config and the complete mapping can be found in [configuration.py](configuration.py).  

```python
from quam import QuAM

def build_config(quam: QuAM):
    config = {
        "version": 1,
        "controllers": {
            "con1": {
                "analog_outputs": {
                    1: {"offset": quam.qubits[0].xy.wiring.mixer_correction.offset_I},  # I qubit1 XY
                    2: {"offset": quam.qubits[0].xy.wiring.mixer_correction.offset_Q},  # Q qubit1 XY
                    ...
                    7: {
                        "offset": quam.qubits[0].z.max_frequency_point,
                        "filter": {
                            "feedforward": quam.qubits[0].z.wiring.filter.fir_taps,
                            "feedback": quam.qubits[0].z.wiring.filter.iir_taps,
                        },
                    },  # qubit1 Z
                    ...
                },
                "digital_outputs": {},
                "analog_inputs": {
                    1: {"offset": quam.global_parameters.downconversion_offset_I, "gain_db": 0},
                    2: {"offset": quam.global_parameters.downconversion_offset_Q, "gain_db": 0}, 
                },
            },
        },
        "elements": {...
            'qubit0': {
                'mixInputs': {
                    'I': ('con1', quam.qubits[0].wiring.xy.I),  
                    'Q': ('con1', quam.qubits[0].wiring.xy.Q),
                    "lo_frequency": quam.local_oscillators.qubits[0].freq,
                    ...
                }
        },
        "pulses": {},
        "waveforms": {},
        "digital_waveforms": {},
        "integration_weights": {},
        "mixers": {},
    }
    return config
```

This one-to-one mapping between the user-defined structure and the OPX configuration allows an easy conversion towards the QuAM framework.
However, for scaling up to many qubit systems seamlessly, a more programmatic mapping between the two structures will be available soon. 

## How to set and get parameters
QuAM can now be used in the calibration scripts to set and get the parameters from the state.

```python
from quam import QuAM
from configuration import build_config
from qualang_tools.plot.fitting import Fit

machine = QuAM("quam_bootstrap_state.json", flat_data=False)

# Get the OPX config
config = build_config(machine)

# Perform some calibration ...
dfs = np.arange(-12e6, +12e6, 0.1e6)
res_if_1 = machine.resonators[0].f_res - machine.local_oscillators.readout[0].freq

with program() as res_spec:
    df = declare(int)
    with for_(*from_array(df, dfs)):
        update_frequency("rr0", df + res_if_1)
        measure()
# ...

qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.qop_port)
qm = qmm.open_qm(config)
job = qm.execute(res_spec)

# Update values based on the results
fit = Fit()
res_1 = fit.reflection_resonator_spectroscopy((machine.resonators[0].f_res + dfs) / u.MHz, np.abs(I**2 + Q**2))
machine.resonators[0].f_res = res_1["f"]

# And save the new state of the system.
machine._save("quam_state_after_res_spec.json", flat_data=False)
```
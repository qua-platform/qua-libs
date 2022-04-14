---
id: index
title: Characterization and tomography of a hidden qubit 
sidebar_label: tomography of a hidden qubit 
slug: ./
---


We implement work published in ref [[1]](#1). The device used in this paper has three elements:  A superconducting qubit connected to a control line and a readout resonator, 
a second transmon which is not connected to such lines and is therefore _hidden_ and a tunable coupling element, controlling the 
interaction of these two qubits. 

In the paper, it is explained how single qubit roations (on the _visible_ qubit) alongside two qubit 
operations enabled by the coupler allow to effectively perform any rotation on the _hidden_ qubit as well. 
To prove this ability, full tomography of several processes is performed. 
We implement this procedure using QUA in this [script](hidden_qubit_tomography.py).
The operations, initialization steps and tomography operation set are defined in dictionaries in the configuration file. 

The code for this procedure is quite succinctly implemented with just a few `for` loops:


```python

 with for_(var=N, init=0, cond=N < N_shots, update=N + 1):
        for process in processes.keys():  # 1/4
            for input_state in state_prep.keys():  # 16
                for readout_operator in tomography_set.keys():  # 15
                    align("RR", "control", "TC")
                    for pulse in state_prep[input_state]:
                        play_pulse(pulse)
                    for pulse in processes[process]:
                        play_pulse(pulse)

                    for op in tomography_set[readout_operator]:
                        if op != "readout":
                            play_pulse(op)
                        else:
                            align("RR", "control", "TC")
                            measure_and_reset_state("RR", I, Q, state_c, state_h, stream_c, stream_h)
```

<a id="1">[1]</a> Pechal, M., Salis, G., Ganzhorn, M., Egger, D. J., Werninghaus, M., & Filipp, S. (2020). Characterization and tomography of a hidden qubit. ArXiv, 1–14.

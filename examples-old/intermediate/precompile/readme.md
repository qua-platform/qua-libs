---
title: Introduction to precompile
sidebar_label:  precompile
slug: ./
id: index
---
#Precompilation

The precompile feature allows generating programs with waveforms you can modify without recompiling the program. 
This saves both the recompilation time itself, and the need to reopen the quantum machine and upload the `config`. 

| NOTE: This feature cannot be used in Simulator mode |
| --- |
The type of program where this works best is when you want to update a waveform between runs, but not the rest of the program.
The modified waveform must be of a constant duration and be defined as overridable in the config. This is shown below:

```(python) 
"waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
        "arb_wf": {"type": "arbitrary",  "samples": [0.2] * arb_len, 'is_overridable': True},
    },
```

You can perform a precompilation by calling the `compile` function of the quantum machine instance

```(python)
with program() as prog:
    for ind in range(2000):
        play("arbPulse", "qe1")

QM1 = QMm.open_qm(config)
program_id = QM1.compile(prog)

```

Compiled jobs are instantiated using the job queue:

```(python)
job = QM1.queue.add_compiled(compiled_program, overrides={
        'waveforms': {
            'arb_wf': make_wf(),
        }
    }).wait_for_execution()

```


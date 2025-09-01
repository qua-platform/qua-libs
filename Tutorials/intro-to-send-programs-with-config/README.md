## Send programs with config (QOP350)

This tutorial explains the new **Send programs with config** feature introduced in **QOP350**.

### Overview
In previous QOP versions, once you opened a qm (=qmm.open_qm(config)) with a specific configuration, there is only limited modifications that can be done with quantum machines API [opx1000](https://docs.quantum-machines.co/latest/docs/API_references/qm_opx1000_api/) and [opx+](https://docs.quantum-machines.co/latest/docs/API_references/qm_api/). If you wanted to change the configuration in more parameters, you had to close and reopen with new config file.

Starting wiht QOP350, you can:
- Modify the qm (i.e. `job=qm.execute()`) by providing a dedicated config that applies only to that job and can be changed **from job to job**.
- Avoid reopening qm after every changes of config.

---

### Backward Compatibility
This feature is **fully backward compatible**.  
You can still use the traditional workflow of opening a QM with a dedicated configuration for each program execution if you prefer.

---

### Configuration File Structure
Conceptually, we draw a clear boundary between classical components and quantum device configurations. The controller config represents classical components: it specifies LO frequencies for upconversion, digital filter to shape pulse, and mixer to correct demodulated input signal. The logical config characterizes quantum device: it defines elements, pulses, waveforms, digital waveforms, and integration weights, which likes a data sheet showing the property of quantum device. By dividing config into controller and logical config, the config file becomes easier to reason about, improving both clarity and readability. By opening the qm with a config, it allows OPX1000 idle value(e.g. DC offset) stay between programs and do not reset to zero. 

In short, the controller config anchors the physical setup and idle values, while the logical config focuses on experimental development. “Here is a list of what the configuration includes:

1. **Controller Configuration**  
   Contains:
   - Controllers
   - Mixers

2. **Logical Configuration**  
   Contains:
   - elements
   - pulses
   - waveforms
   - difital_waveforms
   - integration_weights


---

### Examples
   In the folder, we provide two examples to show how to use this feature.

   1. Modify config On-the-Fly Without Reopening– Shows how to open a qm with a full configuration (demonstrating backward compatibility), run a job, then modify settings like DC offsets or pulse lengths before re-running. No need to reopen the QM — you can change parameters on-the-fly for faster experiment cycles.
   2. Time of flight calibration: Demonstrates running a raw ADC acquisition program and adjusting time of flight through config overrides between runs. Perfect for quickly refining calibration settings without restarting the QM connection.


---

### Limitations
⚠ **Not compatible with Octave** – If your configuration includes Octave, you must use the previous method of opening a QM.

⚠ **FEM port cannot be changed**  – The FEM port in the elements field cannot be changed dynamically.



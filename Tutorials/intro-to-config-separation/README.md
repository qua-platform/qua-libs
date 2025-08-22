## Send program with config (QOP350)

This tutorial explains the new **Send program with config** feature introduced in **QOP350**.

### Overview
In previous QOP versions, once you opened a qm (=qmm.open_qm(config)) with a specific configuration, it could only run with that limited modification of config. If you wanted to change the configuration in more parameters, you had to close and reopen with new config file.

From QOP350 onwards, you can:
- Modify the job (i.e. `job=qm.execute()`) by providing a dedicated config that applies only to that job and can be changed **from job to job**..
- Avoid reopening qm after every change of config.

This makes testing and iterating on experiments much faster.  

---

### Backward Compatibility
This feature is **fully backward compatible**.  
You can still use the traditional workflow of opening a QM with a dedicated configuration for each program execution if you prefer.

---

### Configuration File Structure
Conceptually, we draw a clear boundary between hardware and software. The controller config represents the hardware layer: it captures FEM input, output, and mixer settings—parameters that rarely change. The logical config represents the software (experiment) layer: it defines elements, pulses, waveforms, digital waveforms, and integration weights, which you adjust frequently while calibrating and optimizing experiments. By isolating experiment logic from hardware details, the configuration becomes easier to reason about, improving both efficiency and readability. In short, the controller config anchors the physical setup, while the logical config accelerates experimental development. Below is how we seperate the configuration:

1. **Controller Configuration**  
   Contains:
   - Controllers
   - Mixers

2. **Logical Configuration**  
   Contains:
   - elements
   - pulses
   - waveform
   - difital_waveform
   - integration_weights


---

### Example
   In the folder, we provide two example to guide you how to use this feature.

   1. Modify config On-the-Fly Without Reopening– Shows how to open a qm with a full configuration (demonstrating backward compatibility), run a job, then modify settings like DC offsets or pulse lengths before re-running. No need to reopen the QM — you can change parameters on-the-fly for faster experiment cycles.
   2. Time of flight calibration: Demonstrates running a raw ADC acquisition program and adjusting time of flight through config overrides between runs. Perfect for quickly refining calibration settings without restarting the QM connection.


---

### Limitations
⚠ **Not compatible  with Octave** – If your configuration includes Octave, you must use the previous method of opening a QM.

⚠ **FEM port is not changeable**  – The FEM port in the elements field cannot be changed dynamically.



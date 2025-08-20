## config separation (QOP350)

This tutorial explains the new **config separation** feature introduced in **QOP350**.

### Overview
In previous QOP versions, once you opened a qm (=qmm.open_qm(config)) with a specific configuration, it could only run with that limited modification of config. If you wanted to change the configuration in more parameters, you had to close and reopen with new config file.

In the new version, you can now:
- Modify the job(=qm.execute()) throgh provding temporily different config **from job to job**.
- Avoid reopening qm after every change of config.

This makes testing and iterating on experiment much faster.

---

### Backward Compatibility
This feature is **fully backward compatible**.  
You can still use the traditional workflow of opening a QM with a dedicated configuration for each program execution if you prefer.

---

### Configuration File Structure
The configuration file is now split into two parts:

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

   1. example1– Shows how to open a qm with a full configuration (demonstrating backward compatibility), run a job, then modify settings like DC offsets or pulse lengths before re-running. No need to reopen the QM — you can change parameters on-the-fly for faster experiment cycles.
   2. example2: Demonstrates running a raw ADC acquisition program and adjusting time of flight through config overrides between runs. Perfect for quickly refining calibration settings without restarting the QM connection.


---

### Limitations
⚠ **Not compatible  with Octave** – If your configuration includes Octave, you must use the previous method of opening a QM.

⚠ **FEM port is not changeable**  – FEM port settings cannot be overridden dynamically.



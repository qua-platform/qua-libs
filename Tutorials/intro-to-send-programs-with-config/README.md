## Send programs with config (QOP350)

This tutorial explains the new **Send programs with config** feature introduced in **QOP350**.

### Overview
In previous QOP versions, once you opened a qm (=qmm.open_qm(config)) with a specific configuration, there is only limited modifications that can be done with quantum machines API [opx1000](https://docs.quantum-machines.co/latest/docs/API_references/qm_opx1000_api/) and [opx+](https://docs.quantum-machines.co/latest/docs/API_references/qm_api/). If you wanted to change the configuration in more parameters, you had to close and reopen with new config file.

Starting with QOP350, you can:
- Modify the qm (i.e. `job=qm.execute()`) by providing a dedicated config that applies only to that job and can be changed **from job to job**.
- Avoid reopening qm after every changes of config.

---

### Backward Compatibility
This feature is fully backward compatible. You can still use the traditional workflow of opening a QM with a dedicated configuration for each program execution if you prefer.

---

### Configuration File Structure
We divide the full configuration into two parts: the controller configuration and logical configuration. The controller configuration specifies LO frequency (for up-conversion), digital filters to compensate cable effects (e.g. skin effect) and mixer to correct output signal etc. The logical configuration provides device-specific settings for the quantum device, such as, elements (basis information), pulses to define the elements operations, waveforms to defines different pulses etc.

Imagine an admin who has a quantum device and grants a user access. The admin opens QM and updates calibration values as needed. The user can then submit jobs/experiments with only the logical configuration, without providing all the calibration details. This results in a simpler, more user-friendly workflow.

Here is a list of what the configuration includes:

1. **Controller Configuration**  
   - Controllers
   - Mixers

2. **Logical Configuration**  
   - elements
   - pulses
   - waveforms
   - digital_waveforms
   - integration_weights


---

### Examples
   In the folder, we provide two examples to show how to use this feature.

   1. Modify config On-the-Fly Without Reopening: Shows how to open a qm with a full configuration (demonstrating backward compatibility), run a job, then modify settings like DC offsets or pulse lengths before re-running. No need to reopen the QM — you can change parameters on-the-fly for faster experiment cycles.
   2. Time of flight calibration: Demonstrates running a raw ADC acquisition program and adjusting time of flight through config overrides between runs. Perfect for quickly refining calibration settings without restarting the QM connection.


---

### Limitations
⚠ **Not compatible with Octave** – If your configuration includes Octave, you must use the previous method of opening a QM.

⚠ **ports/FEMs/chassis cannot be added**  – The ports/FEMs/chassis in the controller config cannot be added.



---
title: Random access quantum information processors using multimode circuit quantum electrodynamics
sidebar_label: RAM-Multimode CQED
slug: ./
id: index
---



This script implements work published in: 
Random access quantum information processors using multimode circuit quantum electrodynamics (2017) by R.K. Naik et. al. ArXiv. https://doi.org/10.1038/s41467-017-02046-6

The system is composed of a transmon qubit coupled to a multimode cavity. The transmon 
serves as a non-linear element allowing for energy to be exchanged between cavity modes. 

The script `ram_multimode.py` implements the measurement performed in the paper by defining 
three quantum elements in the configuration file `configuration.py`: a `charge_line`, a `flux_line` 
and a `readout_resonator` each is a mixed-input element driving a mixer. The charge line provides the XY control
for the qubit. The flux line is modulated as to generate sidebands which can be made resonant with the different cavity modes, 
providing the coupling between these modes.

The script has a nested for loops (and an additional loop for averaging). The external loop sweeps the flux modulation 
frequency and the inner loop sweeps the duration of the drive on the `charge_line`. The state of the qubit is eventually read out
and saved to a stream. The stream is reshaped according to the parameter list sizes and averaged. 

```python
            with for_each_(freq, sb_freqs):
                update_frequency("flux_line", freq)
                with for_(t, t_init, t < t_final, t + step):
                    # prepare the transmon in the e or f states
                    prepare()
                    align("charge_line", "flux_line")

                    # modulate the flux bias to create sidebands
                    play("modulate", "flux_line", duration=t)
                    align("flux_line", "readout_resonator")

                    # measure the transmon state
                    measure("readout", "readout_resonator", None,
                            demod.full("integW_cos", I1, "out1"),
                            demod.full("integW_sin", Q1, "out1"),
                            demod.full("integW_cos", I2, "out2"),
                            demod.full("integW_sin", Q2, "out2"),
                            )
                    active_reset("charge_line", Q1)
                    assign(I, I1 + Q2)
                    assign(Q, -Q1 + I2)

                    # estimate and assign state
                    state_estimate(I, Q, state_var)
                    save(state_var, state)

        with stream_processing():
            state.buffer(len(sb_freqs), int((t_final - t_init) / step)).average().save("state")
```











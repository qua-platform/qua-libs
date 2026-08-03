# 09b_power_rabi_error_amplification

## Description


        POWER RABI WITH ERROR AMPLIFICATION
This sequence performs a 2D power-Rabi measurement with error amplification: for each amplitude prefactor, an even
number of π pulses is played and the spin state is measured. Joint-outcome streams are averaged and reduced to
conditional expectations. Small amplitude errors accumulate over many pulses, enabling a precise refinement of
the π-pulse amplitude prefactor in a narrow window around the value from node 09a.

Prerequisites:
    - Having calibrated the relevant voltage points.
    - Having calibrated the qubit frequency and gate duration.
    - Having run node 09a_power_rabi to obtain a coarse π-pulse amplitude prefactor.

Datasets:
    - ``ds_raw``: untouched joint-outcome streams fetched from the OPX (never modified after acquisition).
    - ``ds_fit``: processed 2D sweeps plus analysis outputs (conditional expectations and mean-signal diagnostics).
      Used by ``plot_data``.
    - ``fit_results``: compact per-qubit calibration dict (``FitParameters`` serialized with ``asdict``). Used by
      logging, ``node.outcomes``, and ``update_state``.

Results (``node.results["fit_results"][qubit]``):
    - ``success``: whether the fit passed sanity checks and the state update is applied.
    - ``opt_amp``: refined amplitude prefactor for a π rotation.
    - ``rabi_frequency`` [rad / (unit amplitude · pulse)]: fitted Rabi frequency from the mean-signal model.
    - ``decay_rate`` [1 / pulse]: exponential decay rate per pulse in the error-amplification sequence.
    - ``gauss_decay_rate`` [1 / pulse]: Gaussian decay contribution per pulse.
    - ``n_eff``: effective number of pulses before the contrast envelope decays to 1/e.

Figures (``node.results["figures"]``):
    - ``"heatmap"``: 2D map of the analysis signal vs amplitude prefactor and number of pulses.
    - ``"resonance"``: n_pulses-averaged signal vs amplitude with analytic fit overlay.

State update:
    - The amplitude prefactor of the selected operation (``node.parameters.operation``).
    - When calibrating x180, x90 is also updated to half the x180 prefactor.


## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `analysis_signal` | `E_p1_given_p0_0` | Which conditional expectation to use for fitting.
E_p1_given_p0_0: P(second=1 | first=0) — post-select on empty dot.
E_p1_given_p0_1: P(second=1 | first=1) — post-select on loaded dot. |
| `parity_measurement` | `False` | Whether or not to perform parity measurement. |
| `target_state` | `None` | The state you want to initialize into for heralded initialization. |
| `max_loops` | `100` | Maximum number of initialization loops for heralded initialization. |
| `return_n_loops` | `False` | Whether to return the number of times it has looped over the initialise sequence to achieve the desired result. |
| `multiplexed` | `False` | Whether to play control pulses, readout pulses and active/thermal reset at the same time for all qubits (True)
or to play the experiment sequentially for each qubit (False). Default is False. |
| `use_state_discrimination` | `False` | Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated
quadratures 'I' and 'Q'. Default is False. |
| `reset_type` | `thermal` | The qubit reset method to use. Must be implemented as a method of Quam.qubit. Can be "thermal", "active", or
"active_gef". Default is "thermal". |
| `qubits` | `['q1']` | A list of qubit names which should participate in the execution of the node. Default is None. |
| `num_shots` | `8` | Number of averages to perform. Default is 100. |
| `min_amp_factor` | `0.85` | Minimum amplitude prefactor. Narrow window around expected a_π after node 09a. |
| `max_amp_factor` | `1.15` | Maximum amplitude prefactor. Narrow window around expected a_π after node 09a. |
| `amp_factor_step` | `0.001` | Step size for the amplitude prefactor sweep. Default is 0.001. |
| `max_n_pulses` | `40` | Number of pulses in the error-amplified power Rabi pulse sequence. |
| `operation` | `x180` | The operation to perform to drive the qubit. |
| `use_simulated_data` | `False` | Whether to generate simulated data instead of measuring via the OPX. Default False. |
| `simulate` | `False` | Simulate the waveforms on the OPX instead of executing the program. Default is False. |
| `simulation_duration_ns` | `50000` | Duration over which the simulation will collect samples (in nanoseconds). Default is 50_000 ns. |
| `use_waveform_report` | `True` | Whether to use the interactive waveform report in simulation. Default is True. |
| `timeout` | `120` | Waiting time for the OPX resources to become available before giving up (in seconds). Default is 120 s. |
| `load_data_id` | `None` | Optional QUAlibrate node run index for loading historical data. Default is None. |

## Fit Results

| Qubit | f_res (GHz) | t_pi (ns) | Omega_R (rad/ns) | gamma (1/ns) | T2* (ns) | success |
|-------|-------------|----------|--------------|----------|----------|--------|
| q1 | 0.0000 | nan | 5.610737 | 0.00000 | 15864727907467 | True |

## Updated State

| Qubit | intermediate_frequency (Hz) | xy.operations.x180.length (ns) |
|-------|-----------------------------|-----------------------------------------|
| q1 | 0 | nan |

## Analysis Output

![heatmap](heatmap.png)
![resonance](resonance.png)

---
*Generated by analysis test infrastructure (virtual_qpu)*

# Quantum-dots calibration nodes

All nodes under `calibrations/` follow the same workflow and result layout. The resonator spectroscopy family (`loss_divincenzo/02a`, `02b`, `02c`) is the reference implementation.

When adding a new node, mirror that structure in both:

- `calibrations/<graph>/<node>.py` — QUAlibrate node (QUA program, actions, description)
- `calibration_utils/<package>/` — analysis, plotting, parameters, optional simulated data

---

## Node workflow

Every node uses the same `# %% {Section}` cells and `@node.run_action` actions, in this order:

| Section | Action | Purpose |
|---------|--------|---------|
| `{Imports}` | — | Imports and package wiring |
| `{Node initialisation}` | — | `description`, `QualibrationNode(...)`, `node.machine = Quam.load()` |
| `{Create_QUA_program}` | `create_qua_program` | Build sweep axes, QUA program, `node.namespace["sweep_axes"]` |
| `{Simulate}` | `simulate_qua_program` | Optional QUA simulation (`node.parameters.simulate`) |
| `{Execute}` | `execute_qua_program` | Run on hardware → `node.results["ds_raw"]` |
| `{Generate_simulated_data}` | `generate_simulated_data` | Offline dataset → `node.results["ds_raw"]` (`use_simulated_data`) |
| `{Load_historical_data}` | `load_data` | Reload a saved run by `load_data_id` |
| `{Analyse_data}` | `analyse_data` | Process + fit → `ds_fit`, `fit_results`, `node.outcomes` |
| `{Plot_data}` | `plot_data` | Figures → `node.results["figures"]` |
| `{Update_state}` | `update_state` | Write successful fits back to QuAM (`record_state_updates`) |
| `{Save_results}` | `save_results` | `node.save()` |

**Skip rules (standard across nodes)**

- `create_qua_program`: skip if `load_data_id` is set or `use_simulated_data`
- `execute_qua_program`: skip if `load_data_id`, `simulate`, or `use_simulated_data`
- `simulate_qua_program`: skip if `load_data_id`, not `simulate`, or `use_simulated_data`
- `generate_simulated_data`: skip unless `use_simulated_data`
- `load_data`: skip unless `load_data_id` is set
- `analyse_data` / `plot_data` / `update_state`: skip if `simulate` (simulation-only runs)

**`custom_param`**

Local debugging only (`skip_if=node.modes.external`). Keep active code commented; show relevant parameter examples.

---

## Data flow

```text
Execute / simulated data / load_data
        ↓
   ds_raw          ← never modified after acquisition
        ↓ process_raw_dataset (local, not stored)
   ds_fit          ← processed sweeps + per-sensor summary coords
   fit_results     ← scalar dict for logging and QuAM update
        ↓
   log + outcomes + plot_data + update_state
```

### `node.results` contract

| Key | Type | Written in | Used by |
|-----|------|------------|---------|
| `ds_raw` | `xr.Dataset` | `execute_qua_program`, `generate_simulated_data`, `load_data` | `analyse_data` (read-only) |
| `ds_fit` | `xr.Dataset` | `analyse_data` | `plot_data` |
| `fit_results` | `dict[str, dict]` | `analyse_data` | logging, `node.outcomes`, `update_state` |
| `figures` | `dict[str, Figure]` | `plot_data` | GUI, saved results |
| `simulation` | `dict` | `simulate_qua_program` | optional waveform report |

Also set in **`analyse_data`**:

```python
node.outcomes = {
    name: ("successful" if fit_result["success"] else "failed")
    for name, fit_result in node.results["fit_results"].items()
}
```

### What goes in each dataset

**`ds_raw`** — exactly what the fetcher returns: `I`, `Q`, sweep coordinates, entity dimension. No derived fields.

**`ds_fit`** — output of the analysis pipeline:

- Derived measurement fields (`IQ_abs`, `phase`, `full_freq`, node-specific normalizations, 2D maps if needed)
- Per-sensor **summary coordinates** useful for plotting (e.g. `frequency_shift`, `optimal_power`, `success`)
- Lorentzian fit variables when relevant (`amplitude`, `position`, `width`, …)

Do **not** store algorithm internals in `ds_fit` (derivative traces, boolean masks, PCA working arrays) if they are only used inside `fit_raw_data`. Recompute in plotting when cheap (e.g. `IQ_abs_norm.idxmin(dim="frequency_detuning")`).

**`fit_results`** — one entry per calibrated entity (sensor, qubit, dot pair, …). Values come from a node-specific `@dataclass FitParameters`, serialized with `asdict`. This is the **canonical scalar API** for scripts and state updates.

### Standard `analyse_data` body

```python
ds_processed = process_raw_dataset(node.results["ds_raw"], node)
node.results["ds_fit"], fit_results = fit_raw_data(ds_processed, node)
node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
log_fitted_results(node.results["fit_results"], log_callable=node.log)
node.outcomes = { ... }  # see above
```

Analysis lives in `calibration_utils/<package>/analysis.py`:

- `process_raw_dataset(ds, node)` — derived fields and coordinates
- `fit_raw_data(ds, node)` — inference; returns `(ds_fit, dict[str, FitParameters])`
- `log_fitted_results(...)` — human-readable per-entity log lines

---

## xarray conventions

Shared across nodes where the physics matches:

| Concept | Name | Notes |
|---------|------|-------|
| Entity dimension | `sensor` | Singular, all nodes |
| RF offset sweep (1D / vs power) | `frequency_detuning` | Hz; `long_name`: readout frequency detuning from IF |
| RF offset sweep (vs gate voltage) | `frequency` | Hz (02c) |
| QD pair gate voltage | `detuning` | V (02c only — standard “detuning” after virtualization) |

**`fit_results` field names** (use when applicable; one `FitParameters` class per node with only relevant fields):

| Field | Meaning |
|-------|---------|
| `success` | Fit passed sanity checks; gates `update_state` |
| `resonator_frequency` | Absolute readout frequency [Hz] |
| `frequency_shift` | Fitted offset from IF [Hz] |
| *(node-specific)* | e.g. `fwhm`, `optimal_power`, `optimal_detuning`, `peak_pca_signal` |

Do not add placeholder keys for fields other nodes use.

---

## Node `description` template

The string passed to `QualibrationNode(..., description=...)` is shown in the QUAlibrate GUI. Use this skeleton and fill in the node-specific sections:

```text
<TITLE IN CAPS>
<One short paragraph: what is measured, what is extracted, what QuAM parameters are updated.>

Prerequisites:
    - <prior nodes or calibrated quantities>

Datasets:
    - ``ds_raw``: untouched I/Q fetched from the OPX (never modified after acquisition).
    - ``ds_fit``: processed sweeps plus analysis outputs (derived fields and per-sensor summary
      coordinates). Used by ``plot_data``.
    - ``fit_results``: compact per-sensor calibration dict (``FitParameters`` serialized with
      ``asdict``). Used by logging, ``node.outcomes``, and ``update_state``.

Results (``node.results["fit_results"][<sensor>]``):
    - ``success``: whether the fit passed sanity checks and the state update is applied.
    - <node-specific fields with units>

Figures (``node.results["figures"]``):
    - <e.g. ``"amplitude"``, ``"phase"``, optional ``"summary"``>

State update:
    - <QuAM paths written on success, e.g. readout IF, power, gate voltage>
```

Keep **Results** limited to fields that node actually returns. Keep **Datasets** identical across nodes (copy from above).

---

## Optional calibration summary figure

For nodes with 2D sweeps or several scalars per sensor, add a lightweight **summary** figure so users can see pass/fail and before→after values without opening `ds_fit`.

**Store as:** `node.results["figures"]["summary"]`

**Suggested content (one block per sensor):**

- Outcome (`successful` / `failed`)
- QuAM parameter before → after (only if success)
- Key fitted scalars from `fit_results` (with units)

**Example call in `plot_data`:**

```python
fig_summary = plot_calibration_summary(
    sensors=node.namespace["sensors"],
    fit_results=node.results["fit_results"],
    outcomes=node.outcomes,
)
node.results["figures"]["summary"] = fig_summary
```

Implement `plot_calibration_summary` in the node’s `calibration_utils/.../plotting.py`. A simple text table or one-row-per-sensor matplotlib figure is enough.

Skip the summary when a single physics plot already annotates all decision points clearly.

---

## `calibration_utils` package layout

```text
calibration_utils/<node_name>/
    __init__.py          # re-exports Parameters, analysis, plotting helpers
    parameters.py        # QualibrationNode Parameters dataclass
    analysis.py          # process_raw_dataset, fit_raw_data, FitParameters, log_fitted_results
    plotting.py          # figure builders
    simulated_data_generator.py   # optional; for use_simulated_data
```

---

## Reference nodes

| Node | Package | Sweep dims |
|------|---------|------------|
| `02a_resonator_spectroscopy` | `resonator_spectroscopy` | `(sensor, frequency_detuning)` |
| `02b_resonator_spectroscopy_vs_power` | `resonator_spectroscopy_vs_power` | `(sensor, power, frequency_detuning)` |
| `02c_resonator_spectroscopy_vs_detuning` | `resonator_spectroscopy_vs_detuning` | `(sensor, frequency, detuning)` |

Use these as templates when porting or authoring new calibrations in this directory.

# Execute Tests for Quantum Dot Calibration Nodes

**Date:** 2026-04-16
**Status:** Approved

## Summary

Add a new `tests/execute/` test suite that runs each Loss-DiVincenzo calibration
node with `simulate=False` on a real QM cluster. The tests verify the full
pipeline (compile → execute → analyse → plot → update state) completes without
error, then collect all figures, fit results, and state updates into a
comprehensive README report saved under `tests/execute/artifacts/<node_name>/`.

## Design Decisions

| Decision | Choice |
|----------|--------|
| QM connection | Real hardware — requires `QM_HOST` or `.qm_cluster_config.json` |
| Validation | No-error execution is sufficient; no explicit shape assertions |
| Per-node files | Lightweight — `NODE_NAME` + call to `execute_runner` fixture |
| README content | Comprehensive — params, figures, fit results, state diffs, metadata |
| Initial node set | Same ~14 nodes that have simulation tests today |
| Directory | `tests/execute/` with own `execute` pytest marker |
| Execution | Single `node.run(simulate=False)`, `node.save()` patched to no-op |
| Approach | Mirror `simulation_runner` pattern (Approach A) |

## Directory Structure

```
tests/execute/
├── __init__.py
├── artifacts/
│   └── <node_name>/
│       ├── README.md
│       ├── figure_0.png  (or named: phase.png, amplitude.png, etc.)
│       └── ...
├── qualibration_graphs/
│   ├── __init__.py
│   └── quantum_dots/
│       ├── __init__.py
│       └── calibrations/
│           └── loss_divincenzo/
│               ├── __init__.py
│               ├── conftest.py
│               ├── test_01a_time_of_flight_exec.py
│               ├── test_02a_resonator_spectroscopy_exec.py
│               ├── test_02b_resonator_spectroscopy_vs_power_exec.py
│               ├── test_06a_PSB_search_opx_sweep_detuning_exec.py
│               ├── test_08a_power_rabi_exec.py
│               ├── test_09a_time_rabi_parity_diff_exec.py
│               ├── test_09b_time_rabi_chevron_parity_diff_exec.py
│               ├── test_10_T1_parity_diff_exec.py
│               ├── test_10a_ramsey_parity_diff_exec.py
│               ├── test_10b_ramsey_detuning_parity_diff_exec.py
│               ├── test_10c_ramsey_chevron_parity_diff_exec.py
│               ├── test_12_hahn_echo_parity_diff_exec.py
│               ├── test_14_single_qubit_randomized_benchmarking_exec.py
│               └── test_14a_crot_spectroscopy_parity_diff_exec.py
```

## Per-Node Test Files

Each test file follows a minimal pattern identical to simulation tests:

```python
"""Execute test for <node_name>."""

from __future__ import annotations

import pytest

NODE_NAME = "<node_stem>"


@pytest.mark.execute
def test_<descriptive_name>_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for <node>."""
    execute_runner(
        node_name=NODE_NAME,
        # param_overrides={...} only when needed (same as sim test)
    )
```

`param_overrides` per node match those from the corresponding simulation test
(same small sweep sizes to keep hardware execution fast). Nodes that need
`apply_small_sweep=False` (e.g., CROT spectroscopy) pass that explicitly.

## conftest.py — `execute_runner` Fixture

### Imports and Shared Helpers

Reuses from `shared_fixtures.py`:
- `load_library_node`, `configure_machine_network`, `ensure_quam_config_stub`
- `apply_param_overrides`, `get_parameters_dict`
- `setup_test_cache`, `patch_qualibrate_logger`

Reuses from `quam_factory.py`:
- `create_ld_quam`

### Constants

- `CALIBRATION_LIBRARY_ROOT` — path to `loss_divincenzo/` node files
- `EXECUTE_ROOT` — `tests/execute/`
- `ARTIFACTS_BASE` — `tests/execute/artifacts/`
- `DEFAULT_SMALL_SWEEP_PARAMS` — same dict as simulation conftest

### Fixtures

**`minimal_quam_factory`** — returns `create_ld_quam()` (same as simulation).

**`markdown_generator`** — uses new `make_markdown_generator_exec()` from
`shared_fixtures.py`.

**`execute_runner(minimal_quam_factory, markdown_generator)`** — the main
fixture:

1. **Build machine:** `machine = minimal_quam_factory()`
2. **Configure network:** `configure_machine_network(machine)` — skip if no
   QM host
3. **Stub Quam.load:** `ensure_quam_config_stub(machine)`
4. **Load node:** `load_library_node(node_name, library_root)`
5. **Apply overrides:** `DEFAULT_SMALL_SWEEP_PARAMS` + per-test overrides
6. **Set simulate=False:** `node.parameters.simulate = False`
7. **Snapshot machine state (before):**
   `machine_before = machine.to_dict(include_defaults=False)` — uses the
   `QuamBase.to_dict()` method (same mechanism that produces `state.json`
   files via `JSONSerialiser`). `BaseQuamQD` overrides `to_dict` to default
   `include_defaults=False` and exclude `voltage_sequences`. If `to_dict()`
   raises, the state diff section is gracefully omitted from the README.
8. **Execute:** with `Quam.load` patched and `node.save` patched to no-op:
   `node.run(simulate=False)`
9. **Snapshot machine state (after):**
   `machine_after = machine.to_dict(include_defaults=False)`
10. **Compute state diff:** recursively compare `machine_before` vs
    `machine_after`, collect changed leaf keys (dot-separated paths) with
    before/after values. Keys starting with `_` or `__class__` are excluded.
    QuAM has no built-in diff utility, so we implement a simple recursive
    dict comparator in `shared_fixtures.py`.
11. **Extract figures:** look for `node.results.get("figure")` (single fig) and
    `node.results.get("figures")` (dict of named figs). Save each to
    `artifacts_dir/<name>.png`
12. **Extract fit results:** `node.results.get("fit_results", {})`
13. **Record metadata:** timestamp (ISO 8601), node name
14. **Generate README:** call `markdown_generator(...)` with all collected data
15. **Assert:** `README.md` exists in artifacts dir

### Pytest Hooks

- `pytest_configure` — register `execute` marker, configure warning filters
  (same as simulation)
- `pytest_collection_modifyitems` — auto-add `pytest.mark.execute` to any test
  under `tests/execute/`

## Shared Fixtures Addition: `make_markdown_generator_exec()`

New function added to `shared_fixtures.py` alongside the existing
`make_markdown_generator_sim()`. Returns a callable with signature:

```python
def _generate(
    node,
    parameters_dict: dict,
    artifacts_dir: Path,
    figures_saved: list[str],       # filenames of saved PNGs
    fit_results: dict,              # {qubit_name: {param: value, ...}}
    state_diff: list[tuple],        # [(key, before, after), ...]
    metadata: dict,                 # {"timestamp": ..., "node_name": ...}
) -> Path:
```

### README Sections Generated

```markdown
# <node.name>

## Description
<node.description>

## Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `param`   | `val` | doc string  |

## Execution Output
![<figure_name>](<figure_name>.png)
...

## Fit Results
### <qubit_name>
| Parameter | Value |
|-----------|-------|
| `freq`    | `1.5e9` |
| `success` | `True` |

## State Updates
| Parameter | Before | After |
|-----------|--------|-------|
| `q1.xy.intermediate_frequency` | `5e9` | `5.1e9` |

## Metadata
| Key | Value |
|-----|-------|
| Timestamp | 2026-04-16T17:00:00 |
| Node | 10a_ramsey_parity_diff |

---
*Generated by execute test infrastructure*
```

Sections with no data (e.g., no fit results, no state updates) are omitted
from the README rather than showing empty tables.

## pyproject.toml Change

Add `execute` marker to the root `pyproject.toml`:

```toml
markers = [
    "unit: fast pure-Python unit tests",
    "analysis: analysis/fit/graph tests over data",
    "interop: controller API contract tests (mocked or dry-run)",
    "simulation: cloud simulator tests (opt-in; not run in CI)",
    "execute: full hardware execution tests (opt-in; requires QM host)",
    "hw: hardware sanity checks (manual/nightly)",
]
```

## Running the Tests

```bash
# Run all execute tests (requires QM_HOST or .qm_cluster_config.json)
pytest tests/execute/ -m execute -v

# Run a single node
pytest tests/execute/ -m execute -k "ramsey_parity_diff" -v
```

Tests skip gracefully when no QM host is configured.

## Nodes Covered (Initial Set)

| # | Node | Sim test exists |
|---|------|-----------------|
| 1 | `01a_time_of_flight` | yes |
| 2 | `02a_resonator_spectroscopy` | yes |
| 3 | `02b_resonator_spectroscopy_vs_power` | yes |
| 4 | `06a_PSB_search_opx_sweep_detuning` | yes |
| 5 | `08a_power_rabi` | yes |
| 6 | `09a_time_rabi_parity_diff` | yes |
| 7 | `09b_time_rabi_chevron_parity_diff` | yes |
| 8 | `10_T1_parity_diff` | yes |
| 9 | `10a_ramsey_parity_diff` | yes |
| 10 | `10b_ramsey_detuning_parity_diff` | yes |
| 11 | `10c_ramsey_chevron_parity_diff` | yes |
| 12 | `12_hahn_echo_parity_diff` | yes |
| 13 | `14_single_qubit_randomized_benchmarking` | yes |
| 14 | `14a_crot_spectroscopy_parity_diff` | yes |

## Adding a New Node

To add an execute test for a new node, create a single file:

```python
"""Execute test for <new_node>."""
from __future__ import annotations
import pytest

NODE_NAME = "<new_node_stem>"

@pytest.mark.execute
def test_<name>_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for <new_node>."""
    execute_runner(node_name=NODE_NAME)
```

No changes needed to `conftest.py` or `shared_fixtures.py`.

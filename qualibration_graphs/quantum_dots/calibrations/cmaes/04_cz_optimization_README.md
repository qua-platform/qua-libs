# 04_cz_optimization — Swapping Optimizer & Score Function

## Replacing the optimizer

The optimizer appears in two places in `04_cz_optimization.py`:

```python
# 1. Import (line ~56)
from calibration_utils.cmaes import run_cmaes_optimization

# 2. Call (inside run_optimization_loop)
opt_result = run_cmaes_optimization(
    evaluate_fn=evaluate_candidates,
    ...
)
```

Your replacement optimizer must accept at minimum:

```python
def run_my_optimizer(
    evaluate_fn: Callable[[np.ndarray], np.ndarray],
    # evaluate_fn takes (pop_size, 3) array, returns (pop_size,) scores
    ...
) -> OptimizationResult:
```

The `evaluate_fn` contract:
- **Input:** `np.ndarray` of shape `(batch_size, 3)` — normalized parameters in [0, 1]
- **Output:** `np.ndarray` of shape `(batch_size,)` — scores, higher is better

If your optimizer doesn't return an `OptimizationResult`, adapt the result handling and plotting accordingly.

## Replacing the score function

Edit `compute_score()` in `04_cz_optimization.py`:

```python
def compute_score(measurements: dict[str, np.ndarray]) -> np.ndarray:
```

**Input dict keys:**
| Key | Meaning |
|-----|---------|
| `"circuit_1"` | P(even parity) after X(target)/2 → CZ → X(target)/2 |
| `"circuit_2"` | P(even parity) after X(control)/2 → CZ → X(control)/2 |
| `"circuit_3"` | P(spin-up) on control after bare CZ |
| `"circuit_4"` | P(even parity) after Y(target)/2 → CZ → X(target)/2 *(optional)* |
| `"circuit_5"` | P(even parity) after Y(control)/2 → CZ → X(control)/2 *(optional)* |

Each value is `np.ndarray` of shape `(pop_size,)` with values in [0, 1].

**Output:** `np.ndarray` of shape `(pop_size,)` — scalar score per candidate, higher is better.

## What stays unchanged when swapping

- `_build_qua_program()` — the QUA program and circuit definitions
- `run_cz_circuits()` — streams parameters, collects results
- `compute_score()` — independent of optimizer choice
- `update_state` — just needs best parameter values

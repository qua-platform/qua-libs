# Plan: NumPyro-Based Bayesian Fit for Rabi Chevron

## Current state

- **Model**: Generalized Rabi formula `P(t, Δ) = A·(Ω/Ω_R)²·sin²(Ω_R·t/2) + offset` with `Ω_R = sqrt(Ω² + (2π·Δ)²)`
- **Fitter**: `scipy.optimize.curve_fit` (point estimates, no uncertainty)
- **Parameters**: `f_res`, `omega` (Rabi freq rad/ns), `amplitude`, `offset`
- **Outputs**: `optimal_frequency`, `optimal_duration` (π-time = π/Ω), `rabi_frequency`, `success`
- **Test**: Analysis restricted to Q1 only (`analyse_qubits=["Q1"]`)

## Problems with current fit

1. **No uncertainties** – only point estimates; downstream logic can’t quantify confidence
2. **Optimization robustness** – `curve_fit` can fail or converge to poor local minima
3. **Parameter bounds** – hard bounds can clip gradients; priors are more natural
4. **No posterior samples** – can’t propagate uncertainty to π-time, visibility, etc.
5. **Likelihood** – parity difference is binomial-like; Gaussian noise may be approximate

## Proposed: JAX + NumPyro Bayesian inference

### Benefits

- **Posterior samples** – full uncertainty on f_res, t_π, Ω, A, offset
- **Priors** – physical bounds encoded as priors instead of hard constraints
- **MCMC** – NUTS handles non-convex surfaces better than curve_fit
- **JAX** – gradients and vectorization for efficient sampling
- **Extensible** – easy to add hierarchical models, multiple qubits, etc.

### Dependencies

- `calibration_utils.bayesian_utils` — shared MCMC engine (`fit_model`, `MCMCConfig`, `posterior_summary`)
- `numpyro` — install via `uv sync --extra bayesian` (or `--extra analysis --extra bayesian`)
- `jax`, `jaxlib` (brought in by `virtual_qpu` or `numpyro`)
- Optional: `arviz` for diagnostics/plots

### Implementation outline

1. **Define NumPyro model**
   - Priors: `f_res` (e.g. Uniform over sweep), `omega` (LogUniform), `A` (HalfNormal), `offset` (Normal)
   - Likelihood: `numpyro.sample("obs", dist.Normal(pred, sigma), obs=y)` or Bernoulli/Beta for counts
   - Deterministic: `t_pi = np.pi / omega` for posterior samples

2. **Fit function**
   - Use `calibration_utils.bayesian_utils.fit_model` with a Rabi-specific `model_fn`
   - Input: `pdiff`, `freqs_hz`, `durations_ns` (same as current)
   - Return: posterior summary (mean, std, 5%/95%) for `f_res`, `t_pi`, `omega`, plus samples for propagation

3. **Interface compatibility**
   - Keep same return shape as current fit: `fit_results[qname] = {optimal_frequency, optimal_duration, rabi_frequency, success, ...}`
   - Add optional keys: `optimal_frequency_std`, `optimal_duration_std`, `samples` (if useful)

4. **Fallback**
   - Keep scipy `curve_fit` as fallback when NumPyro fails or is disabled
   - Config flag: `use_numpyro=True` (or parameter on the node)

### File structure

```
calibration_utils/time_rabi_chevron_parity_diff/
├── analysis.py          # main fit_raw_data, process_raw_dataset (unchanged public API)
├── analysis_scipy.py    # current curve_fit implementation (extracted)
├── analysis_numpyro.py  # new NumPyro model + MCMC fit
├── plotting.py
├── parameters.py
└── PLAN_NUMPYRO_FIT.md  # this file
```

### Likelihood choices

| Option | Pros | Cons |
|--------|------|------|
| Normal(pred, σ) | Simple, fast | σ may need tuning; not exact for binomial |
| Beta(α, β) from counts | Matches parity-diff support [0,1] | Need shot counts (n_avg) per point |
| StudentT(pred, ν, σ) | Robust to outliers | Extra parameter ν |
| **Recommendation** | Start with Normal; add Beta if shot info available | |

### Prior suggestions

- `f_res`: `Uniform(f_min, f_max)` from sweep
- `omega`: `LogUniform(2π·0.001, 2π·0.5)` rad/ns (~1–500 MHz)
- `A`: `HalfNormal(0.5)` or `Uniform(0, 2)`
- `offset`: `Normal(0, 0.2)` or `Uniform(-0.1, 1.1)`
- `sigma`: `HalfNormal(0.1)` for residual scale (if inferring)

### Next steps

1. Review/adjust this plan
2. Incorporate your existing NumPyro code as the starting point
3. Add `analysis_numpyro.py` with model + fit function
4. Wire into `analysis.py` (conditional import, `fit_raw_data` dispatcher)
5. Add tests comparing NumPyro vs scipy on the same synthetic data
6. Add node parameter `use_numpyro: bool = False` for gradual rollout

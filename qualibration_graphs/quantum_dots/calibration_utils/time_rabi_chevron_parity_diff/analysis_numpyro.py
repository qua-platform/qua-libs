"""NumPyro NUTS MCMC fit for Rabi chevron. Same model as scipy; outputs posterior mean ± std."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
import numpy as np

from calibration_utils.bayesian_utils import MCMCConfig, fit_model
from calibration_utils.time_rabi_chevron_parity_diff.init_utils import (
    _estimate_f_res_and_omega_from_chevron,
)


def _rabi_chevron_prediction(
    f_res: jnp.ndarray,
    omega: jnp.ndarray,
    amplitude: jnp.ndarray,
    offset: jnp.ndarray,
    freqs_hz: jnp.ndarray,
    durations_ns: jnp.ndarray,
) -> jnp.ndarray:
    """Rabi model P(t,Δ)=A·(Ω/Ω_R)²·sin²(Ω_R·t/2)+offset (JAX)."""
    # Build 2D grids
    f_2d, t_2d = jnp.meshgrid(freqs_hz, durations_ns, indexing="ij")
    delta_2d = 2.0 * jnp.pi * (f_2d - f_res) * 1e-9
    omega_R = jnp.sqrt(omega**2 + delta_2d**2)
    omega_R = jnp.where(omega_R > 1e-12, omega_R, 1e-12)
    visibility = (omega / omega_R) ** 2
    phase = omega_R * t_2d / 2.0
    return (amplitude * visibility * jnp.sin(phase) ** 2 + offset).ravel()


def build_rabi_chevron_model_fn(
    freqs_hz: jnp.ndarray,
    durations_ns: jnp.ndarray,
    f_min: float,
    f_max: float,
) -> Callable[[Dict[str, Any]], Callable[..., Any]]:
    """Build NumPyro model factory. model_fn(priors) → model sampling f_res, omega, A, offset, sigma."""

    def model_fn(priors: Dict[str, Any]):
        import numpyro
        import numpyro.distributions as dist

        def model(y=None):
            # Priors
            f_res = numpyro.sample(
                "f_res",
                dist.Uniform(
                    priors.get("f_min", f_min),
                    priors.get("f_max", f_max),
                ),
            )
            # omega: rad/ns, LogUniform ~1 MHz to ~500 MHz Rabi
            omega_min = priors.get("omega_min", 2 * np.pi * 0.001)
            omega_max = priors.get("omega_max", 2 * np.pi * 0.5)
            omega = numpyro.sample("omega", dist.LogUniform(omega_min, omega_max))
            amplitude = numpyro.sample("amplitude", dist.Uniform(0.0, 2.0))
            offset = numpyro.sample("offset", dist.Uniform(-0.1, 1.1))
            sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))

            # Deterministic predictions
            pred = _rabi_chevron_prediction(f_res, omega, amplitude, offset, freqs_hz, durations_ns)
            t_pi = numpyro.deterministic("t_pi", np.pi / omega)

            # Likelihood
            numpyro.sample("obs", dist.Normal(pred, sigma), obs=y)

        return model

    return model_fn


def _fit_chevron_single_qubit_numpyro(
    pdiff: np.ndarray,
    freqs_hz: np.ndarray,
    durations_ns: np.ndarray,
    nominal_freq_hz: float,
    config: MCMCConfig | None = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Fit one qubit via NumPyro NUTS. Init from FFT+Rabi heuristic."""
    f_min = float(np.min(freqs_hz)) - 1e6
    f_max = float(np.max(freqs_hz)) + 1e6
    y_flat = jnp.asarray(pdiff.ravel().astype(np.float32))
    freqs_jax = jnp.asarray(freqs_hz, dtype=jnp.float32)
    durations_jax = jnp.asarray(durations_ns, dtype=jnp.float32)

    model_fn = build_rabi_chevron_model_fn(freqs_jax, durations_jax, f_min, f_max)

    f_res_init, omega_init = _estimate_f_res_and_omega_from_chevron(pdiff, freqs_hz, durations_ns, nominal_freq_hz)

    init_vals = {
        "f_res": f_res_init,
        "omega": omega_init,
        "amplitude": float(np.ptp(pdiff)) if np.ptp(pdiff) > 0 else 0.5,
        "offset": float(np.min(pdiff)),
    }

    if config is None:
        config = MCMCConfig(num_warmup=500, num_samples=500, num_chains=1)

    try:
        mcmc, samples, summary = fit_model(
            model_fn,
            y_flat,
            config=config,
            init_vals=init_vals,
            print_summary=False,
        )
    except Exception:
        return {
            "optimal_frequency": nominal_freq_hz,
            "optimal_duration": np.nan,
            "rabi_frequency": np.nan,
            "success": False,
        }, np.full_like(pdiff, np.nan)

    f_res_mean = summary["f_res"]["mean"]
    omega_mean = summary["omega"]["mean"]
    t_pi_mean = summary["t_pi"]["mean"]

    # Reconstruct fit surface from posterior mean
    pred_flat = _rabi_chevron_prediction(
        jnp.array(f_res_mean),
        jnp.array(omega_mean),
        jnp.array(summary["amplitude"]["mean"]),
        jnp.array(summary["offset"]["mean"]),
        freqs_jax,
        durations_jax,
    )
    fit_surface = np.asarray(pred_flat).reshape(pdiff.shape)

    success = (
        f_min <= f_res_mean <= f_max and 4 <= t_pi_mean <= 1e6 and np.isfinite(t_pi_mean) and np.isfinite(f_res_mean)
    )

    result = {
        "optimal_frequency": float(f_res_mean),
        "optimal_duration": float(t_pi_mean),
        "rabi_frequency": float(omega_mean),
        "success": success,
    }
    # Add uncertainties for downstream use
    result["optimal_frequency_std"] = summary["f_res"]["std"]
    result["optimal_duration_std"] = summary["t_pi"]["std"]

    return result, fit_surface

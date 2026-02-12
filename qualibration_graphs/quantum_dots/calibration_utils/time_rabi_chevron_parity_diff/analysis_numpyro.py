"""NumPyro NUTS MCMC fit for damped Rabi chevron with steady-state relaxation.

P(t,Δ) = A·(Ω/Ω_R)²·[sin²(Ω_R t/2)·e^{-γt} + ½(1-e^{-γt})] + offset.
Outputs posterior mean ± std for f_res, ω, γ, t_π, T₂*.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
import numpy as np

_logger = logging.getLogger(__name__)

from calibration_utils.bayesian_utils import MCMCConfig, fit_model
from calibration_utils.time_rabi_chevron_parity_diff.init_utils import (
    _estimate_f_res_and_omega_from_chevron,
)


def _rabi_chevron_prediction(
    f_res: jnp.ndarray,
    omega: jnp.ndarray,
    amplitude: jnp.ndarray,
    offset: jnp.ndarray,
    gamma: jnp.ndarray,
    freqs_hz: jnp.ndarray,
    durations_ns: jnp.ndarray,
) -> jnp.ndarray:
    """Damped Rabi with steady-state relaxation (JAX).

    P(t,Δ) = A·(Ω/Ω_R)²·[sin²(Ω_R t/2)·e^{-γt} + ½(1-e^{-γt})] + offset

    At early times (γt≪1) this reduces to the coherent Rabi formula.
    At late times the oscillation decays toward the detuning-dependent
    steady state A·(Ω/Ω_R)²·½ + offset, matching the physical behaviour
    of a driven dissipative two-level system.
    """
    f_2d, t_2d = jnp.meshgrid(freqs_hz, durations_ns, indexing="ij")
    delta_2d = 2.0 * jnp.pi * (f_2d - f_res) * 1e-9
    omega_R = jnp.sqrt(omega**2 + delta_2d**2)
    omega_R = jnp.where(omega_R > 1e-12, omega_R, 1e-12)
    visibility = (omega / omega_R) ** 2
    phase = omega_R * t_2d / 2.0
    decay = jnp.exp(-gamma * t_2d)
    # Oscillation decays toward steady-state ½ (time-averaged sin²)
    osc = jnp.sin(phase) ** 2 * decay + 0.5 * (1.0 - decay)
    return (amplitude * visibility * osc + offset).ravel()


def build_rabi_chevron_model_fn(
    freqs_hz: jnp.ndarray,
    durations_ns: jnp.ndarray,
    f_min: float,
    f_max: float,
) -> Callable[[Dict[str, Any]], Callable[..., Any]]:
    """Build NumPyro model factory.

    model_fn(priors) → model sampling f_res, omega, gamma, A, offset, sigma.
    Derived deterministics: t_pi, T2_star.
    """

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
            # Amplitude ≈ max−min of the data.  A tight Normal prior
            # prevents amplitude from trading off with γ (decay).
            amp_loc = priors.get("amp_loc", 0.5)
            amp_sigma = priors.get("amp_sigma", 0.15)
            amplitude = numpyro.sample("amplitude", dist.TruncatedNormal(amp_loc, amp_sigma, low=0.0))
            offset = numpyro.sample("offset", dist.Uniform(-0.1, 1.1))

            # Decay rate γ (1/ns).  Use a LogNormal prior centred on the
            # heuristic estimate from the 1-D damped-Rabi / envelope fit.
            # This prevents the posterior from collapsing to γ→0 when
            # amplitude can trade off with decay in the 2-D model.
            # log_gamma_loc = log(gamma_init); sigma = 1.0 gives ~3× range.
            gamma_loc = priors.get("gamma_loc", -5.0)  # log(0.007) ≈ -5
            gamma_sigma = priors.get("gamma_sigma", 1.0)
            gamma = numpyro.sample("gamma", dist.LogNormal(gamma_loc, gamma_sigma))

            sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))

            # Deterministic predictions
            pred = _rabi_chevron_prediction(f_res, omega, amplitude, offset, gamma, freqs_hz, durations_ns)
            t_pi = numpyro.deterministic("t_pi", np.pi / omega)
            T2_star = numpyro.deterministic("T2_star", 1.0 / jnp.maximum(gamma, 1e-12))

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

    f_res_init, omega_init, gamma_init = _estimate_f_res_and_omega_from_chevron(
        pdiff, freqs_hz, durations_ns, nominal_freq_hz
    )

    gamma_init_safe = max(gamma_init, 1e-6)
    init_vals = {
        "f_res": f_res_init,
        "omega": omega_init,
        "amplitude": float(np.ptp(pdiff)) if np.ptp(pdiff) > 0 else 0.5,
        "offset": float(np.min(pdiff)),
        "gamma": gamma_init_safe,
    }

    # Amplitude prior: centre on peak-to-peak of observed data with ~30% σ.
    amp_est = float(np.ptp(pdiff)) if np.ptp(pdiff) > 0 else 0.5
    amp_sigma = max(amp_est * 0.3, 0.05)

    # γ prior: LogNormal centred on FFT HWHM estimate (σ=0.5 ≈ ×4.5 range).
    priors = {
        "gamma_loc": float(np.log(gamma_init_safe)),
        "gamma_sigma": 0.5,
        "amp_loc": amp_est,
        "amp_sigma": amp_sigma,
    }
    _logger.debug(
        "NumPyro priors: amp=N(%.3f, %.3f), gamma=LogN(%.3f, %.1f)  [gamma_init=%.6f]",
        amp_est,
        amp_sigma,
        priors["gamma_loc"],
        priors["gamma_sigma"],
        gamma_init_safe,
    )

    if config is None:
        config = MCMCConfig(num_warmup=500, num_samples=500, num_chains=1)

    try:
        mcmc, samples, summary = fit_model(
            model_fn,
            y_flat,
            config=config,
            init_vals=init_vals,
            priors=priors,
            print_summary=False,
        )
    except (ImportError, ModuleNotFoundError):
        raise  # Let import errors propagate to the caller's fallback handler
    except Exception:
        return {
            "optimal_frequency": nominal_freq_hz,
            "optimal_duration": np.nan,
            "rabi_frequency": np.nan,
            "decay_rate": np.nan,
            "success": False,
        }, np.full_like(pdiff, np.nan)

    f_res_mean = summary["f_res"]["mean"]
    omega_mean = summary["omega"]["mean"]
    _logger.debug("NumPyro posterior: gamma=%.6f ± %.6f", summary["gamma"]["mean"], summary["gamma"]["std"])
    gamma_mean = summary["gamma"]["mean"]
    t_pi_mean = summary["t_pi"]["mean"]
    T2_star_mean = summary["T2_star"]["mean"]

    # Reconstruct fit surface from posterior mean
    pred_flat = _rabi_chevron_prediction(
        jnp.array(f_res_mean),
        jnp.array(omega_mean),
        jnp.array(summary["amplitude"]["mean"]),
        jnp.array(summary["offset"]["mean"]),
        jnp.array(gamma_mean),
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
        "decay_rate": float(gamma_mean),
        "T2_star": float(T2_star_mean),
        "success": success,
    }
    # Add uncertainties for downstream use
    result["optimal_frequency_std"] = summary["f_res"]["std"]
    result["optimal_duration_std"] = summary["t_pi"]["std"]
    result["decay_rate_std"] = summary["gamma"]["std"]
    result["T2_star_std"] = summary["T2_star"]["std"]

    return result, fit_surface

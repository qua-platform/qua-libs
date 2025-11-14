"""
Analytic likelihood functions for the Barthel two-state spin readout model.

This module implements the mathematical core of the Barthel model for quantum spin readout,
where a two-level system (e.g., singlet-triplet) can undergo relaxation during measurement.

The key physics:
- State S (singlet) is stable and produces a Gaussian signal centered at μ_S
- State T (triplet) can decay into S with time constant T₁ during readout time τ_M
- If T decays during measurement, the signal is a mixture reflecting partial evolution

Mathematical framework (from Barthel et al., Phys. Rev. B 81, 161308(R) (2010)):

    n_S(y) = N(y; μ_S, σ)

    n_T(y) = exp(-τ_M / T₁) * N(y; μ_T, σ)
           + (1/T₁) * ∫₀^{τ_M} exp(-t/T₁) * N(y; μ_S + kt, σ) dt

    where k = (μ_T - μ_S) / τ_M

The integral represents spins that decay "in-flight" at various times during readout.

All functions are JAX-compatible for automatic differentiation and JIT compilation.
Numerical stability is ensured via exponent clipping and scaled error functions.
"""

import jax.numpy as jnp
from jax.scipy.special import erf, erfc
try:
    # modern JAX (>=0.4.36): already has it here
    from jax._src.scipy.special import erfcx as _erfcx
except Exception:
    # universal fallback: erfcx(x) = exp(x^2) * erfc(x)
    def _erfcx(x):
        return jnp.exp(x * x) * erfc(x)

# Exponent safety cap: exp(±60) ~ 1e26; safe in float32 and prevents overflow
EXP_CAP = 60.0

def _norm_pdf(y, mu, sigma):
    """
    Gaussian probability density function.

    Computes the PDF of a normal distribution N(μ, σ) at point y.

    Args:
        y: Point(s) at which to evaluate the PDF
        mu: Mean of the Gaussian distribution
        sigma: Standard deviation

    Returns:
        PDF value(s) at y
    """
    z = (y - mu) / sigma
    return jnp.exp(-0.5 * z * z) / (jnp.sqrt(2.0 * jnp.pi) * sigma)

def _std_norm_cdf(z):
    """
    Cumulative distribution function of the standard normal distribution.

    Computes Φ(z) = P(Z ≤ z) where Z ~ N(0, 1).

    Args:
        z: Standardized value

    Returns:
        CDF value at z
    """
    return 0.5 * (1.0 + _safe_erf(z / (2.0**0.5)))

def _safe_erf(z):
    """
    Numerically stable error function.

    Prevents overflow/underflow for extreme values of z by using:
    - Direct erf(z) for moderate values (|z| < 5)
    - Asymptotic limit erf(z) → 1 for large positive z
    - Scaled complementary error function erfcx for large negative z
      using the identity: erf(z) = -1 + exp(-z²) * erfcx(-z)

    This ensures stable computation across the full range of z values
    that may occur during MCMC sampling.

    Args:
        z: Input value(s)

    Returns:
        erf(z) computed stably
    """
    # Use scaled erfc for large negative z; for large positive z -> 1.
    zt = 5.0
    return jnp.where(
        z > zt,
        1.0,
        jnp.where(
            z < -zt,
            -1.0 + jnp.exp(-(z * z)) * _erfcx(-z),  # erf(z) = -1 + e^{-z^2} erfcx(-z)
            erf(z),
        ),
    )

def decay_inflight_integral(y, mu_S, mu_T, sigma, T1, tauM, eps=1e-12):
    """
    Analytic integral for triplet spins that decay during readout.

    Computes the integral:
        ∫₀^{τ_M} exp(-t/T₁) * N(y; μ_S + kt, σ) dt

    This represents the contribution from spins that start in state T but decay
    to state S at some time t during the measurement window [0, τ_M]. The signal
    evolves linearly from μ_T toward μ_S with rate k = (μ_T - μ_S) / τ_M.

    The integral is computed analytically using Gaussian integrals and error functions,
    with special handling for numerical stability:
    - When k ≈ 0 (μ_T ≈ μ_S), uses a limit form to avoid division by zero
    - Clips exponentials to prevent overflow
    - Uses safe error functions for extreme arguments

    Args:
        y: Measured voltage value(s)
        mu_S: Mean voltage for singlet state
        mu_T: Mean voltage for triplet state
        sigma: Readout noise (standard deviation)
        T1: Triplet relaxation time
        tauM: Measurement duration
        eps: Numerical tolerance for detecting k ≈ 0 case

    Returns:
        Value of the in-flight decay integral at y
    """
    # Rate of signal evolution during decay
    k = (mu_T - mu_S) / tauM

    # Quadratic form coefficients for the Gaussian integral
    # After completing the square: exp(-at² + bt - c)
    a = k * k / (2.0 * sigma * sigma)
    b = (k * (y - mu_S)) / (sigma * sigma) - 1.0 / T1
    c = (y - mu_S) * (y - mu_S) / (2.0 * sigma * sigma)

    # Prevent division by zero when k ≈ 0
    a_clipped = jnp.clip(a, a_min=eps)
    sqrt_a = jnp.sqrt(a_clipped)
    pref = (1.0 / (jnp.sqrt(2.0 * jnp.pi) * sigma)) * jnp.exp(-c)

    # Compute integral using completing the square method:
    # Result involves difference of error functions at boundaries
    exp_b2_over_4a = jnp.exp(jnp.clip((b * b) / (4.0 * a_clipped), a_min=-EXP_CAP, a_max=EXP_CAP))
    F_tau = exp_b2_over_4a * _safe_erf((2.0 * a * tauM - b) / (2.0 * sqrt_a))
    F_0   = exp_b2_over_4a * _safe_erf((-b) / (2.0 * sqrt_a))
    integral_general = pref * (jnp.sqrt(jnp.pi) / (2.0 * sqrt_a)) * (F_tau - F_0)

    # Special case: when μ_T ≈ μ_S (k ≈ 0), signal doesn't evolve during decay
    # Limiting form: N(y; μ_S, σ) * T₁ * (1 - exp(-τ_M/T₁))
    limit_k0 = _norm_pdf(y, mu_S, sigma) * T1 * (1.0 - jnp.exp(-tauM / T1))
    use_limit = (jnp.abs(k) < jnp.sqrt(2.0) * sigma * jnp.sqrt(eps))
    return jnp.where(use_limit, limit_k0, integral_general)

def triplet_pdf_analytic(y, mu_S, mu_T, sigma, T1, tauM):
    """
    Probability density function for the triplet state with relaxation.

    Implements the Barthel model PDF for state T:
        n_T(y) = p_no_decay * N(y; μ_T, σ) + (1/T₁) * ∫₀^{τ_M} exp(-t/T₁) * N(y; μ_S+kt, σ) dt

    This is a mixture of two components:
    1. Spins that survive the entire measurement without decaying (probability exp(-τ_M/T₁))
       → Gaussian centered at μ_T
    2. Spins that decay during measurement (weighted by 1/T₁)
       → Smeared distribution between μ_S and μ_T

    This is the core likelihood function used in Bayesian MCMC fitting.

    Args:
        y: Measured voltage value(s)
        mu_S: Mean voltage for singlet state
        mu_T: Mean voltage for triplet state (before decay)
        sigma: Readout noise (standard deviation)
        T1: Triplet → singlet relaxation time
        tauM: Measurement duration

    Returns:
        PDF value(s) for triplet state at y, guaranteed non-negative
    """
    # Probability of no decay during measurement
    p_no = jnp.exp(-tauM / T1)
    # Contribution from spins that don't decay
    n_Tno = p_no * _norm_pdf(y, mu_T, sigma)
    # Contribution from spins that decay in-flight
    n_Tdf = (1.0 / T1) * decay_inflight_integral(y, mu_S, mu_T, sigma, T1, tauM)
    # guard tiny negative due to roundoff
    return jnp.maximum(n_Tno + n_Tdf, 0.0)

def triplet_cdf_analytic(v, mu_S, mu_T, sigma, T1, tauM):
    """
    Cumulative distribution function for the triplet state with relaxation.

    Computes CDF_T(v) = P(Y ≤ v | state is T) by integrating the triplet PDF.
    Like the PDF, this is a mixture of:
    1. Spins that don't decay (probability exp(-τ_M/T₁)) → standard Gaussian CDF at μ_T
    2. Spins that decay in-flight → analytic integral of the in-flight contribution

    Used for computing readout fidelity F_T = 1 - CDF_T(threshold) and visibility.
    The CDF form allows efficient computation of classification performance metrics.

    Args:
        v: Threshold voltage value(s)
        mu_S: Mean voltage for singlet state
        mu_T: Mean voltage for triplet state (before decay)
        sigma: Readout noise (standard deviation)
        T1: Triplet → singlet relaxation time
        tauM: Measurement duration

    Returns:
        CDF value(s) at v, clipped to [0, 1] for numerical stability
    """
    # === Component 1: Spins that don't decay ===
    p_no = jnp.exp(-tauM / T1)
    zTno = (v - mu_T) / sigma
    cdf_T_no = 0.5 * (1.0 + _safe_erf(zTno / jnp.sqrt(2.0)))

    # === Component 2: Spins that decay in-flight ===
    # Integral of the decay contribution from -∞ to v
    # Uses analytic form derived from integrating the in-flight PDF
    I0 = T1 * (1.0 - jnp.exp(-tauM / T1))
    k = (mu_T - mu_S) / tauM
    a = -1.0 / T1  # Exponential decay rate
    c = -k / (sigma * jnp.sqrt(2.0))  # Drift term in standardized coordinates
    d = (v - mu_S) / (sigma * jnp.sqrt(2.0))  # Threshold in standardized coordinates
    eps = 1e-14

    # Special case: c ≈ 0 (when μ_T ≈ μ_S, no drift during decay)
    I1_smallc = (jnp.exp(a * tauM) - 1.0) * (1.0 / a) * (2.0 * (0.5 * (1.0 + _safe_erf(d))) - 1.0)

    # General case: analytic integral using error functions and exponentials
    # This formula comes from integrating exp(-t/T₁) * Φ((v - μ_S - kt) / σ) over t
    expo = jnp.exp(jnp.clip(a * (a - 4.0 * c * d) / (4.0 * c * c), a_min=-EXP_CAP, a_max=EXP_CAP))
    inv_2c = a / (2.0 * c)
    term1 = expo * _safe_erf(inv_2c - c * tauM - d)
    term2 = expo * _safe_erf(inv_2c - d)
    term3 = jnp.exp(jnp.clip(a * tauM, a_min=-EXP_CAP, a_max=EXP_CAP)) * _safe_erf(c * tauM + d)
    term4 = _safe_erf(d)
    I1_gen = (term1 - term2 + term3 - term4) / (a + 0.0)

    # Select between special case and general case based on magnitude of c
    I1 = jnp.where(jnp.abs(c) < eps, I1_smallc, I1_gen)
    cdf_T_decay = (I0 + I1) / (2.0 * T1)

    # Combine both components
    cdf_T = p_no * cdf_T_no + cdf_T_decay
    # guard numeric drift outside [0,1]
    return jnp.clip(cdf_T, a_min=0.0, a_max=1.0)

"""
Bayesian Offline Changepoint Detection via Fearnhead Dynamic Programming (JAX)
================================================================================

This module implements **offline** (smoothing) Bayesian changepoint detection using
the Fearnhead dynamic programming algorithm. Unlike online filtering approaches,
this algorithm uses the entire observed time series to compute changepoint posteriors.

Algorithm Overview
------------------
- **Segment Model**: Piecewise i.i.d. Student-t distribution (obtained by integrating
  out mean and variance with Normal-Gamma conjugate prior)
- **Duration Prior**: Constant-hazard (geometric) prior on segment lengths
- **Inference**: Forward-backward dynamic programming to compute posterior probability
  P(changepoint at t | x_{1:T}) for all time indices t

Key Features
------------
- Fully vectorized JAX implementation for GPU/TPU acceleration
- Robust numerical stability via log-space computation
- Optional data standardization (median/MAD scaling)
- Temperature parameter for posterior sharpening/smoothing

Mathematical Background
-----------------------
The algorithm computes:
    P(cp at t | x) = exp(F[t] + B[t] - log Z)

where:
- F[t] = log p(x_{1:t}): forward evidence (probability of data up to t)
- B[t] = log p(x_{t+1:T} | cp at t): backward evidence (data after t given cp)
- log Z = log p(x_{1:T}): total evidence (normalization constant)

The segment likelihood integrates over unknown mean μ and variance σ²:
    p(x_{s:t}) = ∫∫ p(x_{s:t} | μ, σ²) p(μ, σ²) dμ dσ²

This integral has a closed form (Student-t) using the Normal-Gamma conjugate prior.

References
----------
Fearnhead, P. (2006). "Exact and efficient Bayesian inference for multiple
changepoint problems." Statistics and Computing, 16(2), 203-213.

Author: 2025
License: Apache-2.0
Dependencies: jax, jaxlib, matplotlib (for demos)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp

from .bayesian_base import BayesianMCMCBase
from .standardization import Standardization


# =============================================================================
# Normal-Gamma Conjugate Prior Utilities
# =============================================================================
# The Normal-Gamma distribution is the conjugate prior for a Normal likelihood
# with unknown mean and variance. It allows closed-form Bayesian updates.

def _posterior_from_suff(
    n: jnp.ndarray,
    s1: jnp.ndarray,
    s2: jnp.ndarray,
    mu0: float,
    kappa0: float,
    alpha0: float,
    beta0: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute posterior Normal-Gamma hyperparameters from sufficient statistics.

    Given a prior Normal-Gamma(μ₀, κ₀, α₀, β₀) and data with sufficient statistics
    (n, s1, s2), compute the posterior hyperparameters (μₙ, κₙ, αₙ, βₙ).

    The Normal-Gamma prior assumes:
        σ² ~ InverseGamma(α, β)
        μ | σ² ~ Normal(μ₀, σ²/κ)

    Parameters
    ----------
    n : jnp.ndarray
        Number of observations in the segment
    s1 : jnp.ndarray
        Sum of observations: Σx
    s2 : jnp.ndarray
        Sum of squared observations: Σx²
    mu0 : float
        Prior mean
    kappa0 : float
        Prior precision (larger = stronger belief in μ₀)
    alpha0 : float
        Prior shape parameter for variance
    beta0 : float
        Prior scale parameter for variance

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Posterior hyperparameters (μₙ, κₙ, αₙ, βₙ)

    Notes
    -----
    The function supports broadcasting, allowing batch computation over multiple
    segments simultaneously.
    """
    # Posterior precision: prior + data count
    kappa_n = kappa0 + n

    # Posterior mean: weighted average of prior mean and sample mean
    mu_n = (kappa0 * mu0 + s1) / kappa_n

    # Posterior shape: prior + half the data count
    alpha_n = alpha0 + 0.5 * n

    # Posterior scale: incorporates data variance and prior-data discrepancy
    # Formula accounts for sum of squared deviations from posterior mean
    beta_n = beta0 + 0.5 * (s2 + kappa0 * mu0**2 - kappa_n * mu_n**2)

    return mu_n, kappa_n, alpha_n, beta_n


def _segment_log_marginal(
    n: jnp.ndarray,
    s1: jnp.ndarray,
    s2: jnp.ndarray,
    mu0: float,
    kappa0: float,
    alpha0: float,
    beta0: float,
) -> jnp.ndarray:
    """
    Compute log marginal likelihood of a segment under Normal-Gamma prior.

    This integrates out the unknown mean μ and variance σ² to get the marginal
    probability of the data p(x_{1:n}). The result is a Student-t distribution.

    The closed-form expression is:
        log p(D) = 0.5 * log(κ₀/κₙ)
                   + α₀ * log(β₀) - αₙ * log(βₙ)
                   + log Γ(αₙ) - log Γ(α₀)
                   - (n/2) * log(π)

    Parameters
    ----------
    n : jnp.ndarray
        Number of observations in the segment
    s1 : jnp.ndarray
        Sum of observations: Σx
    s2 : jnp.ndarray
        Sum of squared observations: Σx²
    mu0 : float
        Prior mean
    kappa0 : float
        Prior precision parameter
    alpha0 : float
        Prior shape parameter for variance
    beta0 : float
        Prior scale parameter for variance

    Returns
    -------
    jnp.ndarray
        Log marginal likelihood log p(x_{1:n})

    Notes
    -----
    This function is the core of the segment model. It evaluates how likely
    a segment is to be homogeneous (no changepoints) under the prior assumptions.
    """
    # Compute posterior hyperparameters
    mu_n, kappa_n, alpha_n, beta_n = _posterior_from_suff(
        n, s1, s2, mu0, kappa0, alpha0, beta0
    )

    # Compute log marginal likelihood using closed-form Student-t formula
    log_marginal = (
        0.5 * (jnp.log(kappa0) - jnp.log(kappa_n))  # Precision ratio term
        + alpha0 * jnp.log(beta0)  # Prior scale term
        - alpha_n * jnp.log(beta_n)  # Posterior scale term
        + jax.scipy.special.gammaln(alpha_n)  # Gamma function (posterior)
        - jax.scipy.special.gammaln(alpha0)  # Gamma function (prior)
        - 0.5 * n * jnp.log(jnp.pi)  # Normalization constant
    )

    return log_marginal


# =============================================================================
# Bayesian Changepoint Detector
# =============================================================================

@dataclass
class BayesianCP(BayesianMCMCBase):
    """
    Bayesian offline changepoint detector using Fearnhead dynamic programming.

    This class implements an exact Bayesian inference algorithm for detecting
    changepoints in univariate time series. It computes the posterior probability
    of a changepoint at each time index given the entire observed sequence.

    Attributes
    ----------
    hazard : float, default=1/200
        Constant hazard rate for the geometric duration prior.
        P(segment ends | length n) = hazard * (1 - hazard)^(n-1)
        Smaller values favor longer segments (fewer changepoints).
        Typical range: 1/100 to 1/500
    standardize : bool, default=True
        Whether to standardize data using median and MAD (median absolute deviation).
        Recommended for robustness to outliers and scale variations.
    kappa0 : float, default=5e-3
        Prior precision parameter (pseudo-count for the mean).
        Smaller values => weaker prior => more sensitive to local changes.
        Larger values => stronger prior => smoother changepoint posteriors.
    alpha0 : float, default=5e-1
        Prior shape parameter for the inverse-gamma variance prior.
        Controls prior belief about variance (alpha0 > 1 for finite mean).
    beta0 : float, default=5e-3
        Prior scale parameter for the inverse-gamma variance prior.
        Works with alpha0 to set prior variance: E[σ²] = beta0 / (alpha0 - 1)
    mu0 : float, default=0.0
        Prior mean. Usually set to 0 when data is standardized.
    temp : float, default=1.0
        Temperature parameter for posterior smoothing/sharpening.
        - temp > 1: Flattens/broadens posteriors (less confident)
        - temp < 1: Sharpens posteriors (more confident)
        - temp = 1: No modification (standard posterior)

    Methods
    -------
    fit(x: jnp.ndarray) -> Tuple[jnp.ndarray, float]
        Compute changepoint posterior probabilities for the entire time series.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> # Generate synthetic data with a changepoint at t=50
    >>> x = jnp.concatenate([jnp.zeros(50), jnp.ones(50)])
    >>> x = x + 0.1 * jax.random.normal(jax.random.PRNGKey(0), shape=x.shape)
    >>>
    >>> # Detect changepoints
    >>> model = BayesianCP(hazard=1/100)
    >>> cp_prob, log_evidence = model.fit(x)
    >>> # cp_prob[t] gives P(changepoint at t | data)

    Notes
    -----
    Computational Complexity: O(T²) time, O(T²) space for sequence length T.
    This is exact inference; for very long sequences (T > 10,000), consider
    online approximations or windowing strategies.
    """
    hazard: float = 1 / 200.0
    standardize: bool = True

    # Prior hyperparameters
    # Weaker priors (smaller kappa0, alpha0, beta0) => broader/sharper peaks
    kappa0: float = 5e-3  # Prior precision (pseudo-count)
    alpha0: float = 5e-2  # Prior shape for variance
    beta0: float = 5e-3   # Prior scale for variance
    mu0: float = 0.0      # Prior mean (typically 0 for standardized data)

    # Posterior temperature (>1 flattens, <1 sharpens)
    temp: float = 1.3

    def __post_init__(self):
        super().__init__(standardize=self.standardize)

    def fit(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        """
        Compute posterior changepoint probabilities for all time indices.

        Returns
        -------
        Tuple[jnp.ndarray, float]
            (cp_prob, log_evidence)
        """
        return super().fit(x)

    def _prefix_sums(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute prefix sums for efficient segment statistic calculation.

        Prefix sums enable O(1) computation of sum and sum-of-squares for
        any segment [s, t) via: sum[s:t] = S[t] - S[s]

        Parameters
        ----------
        x : jnp.ndarray
            Input data of shape (T,)

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            (S1, S2) where:
            - S1[t] = sum of x[0:t] (exclusive, so S1[0] = 0)
            - S2[t] = sum of x[0:t]² (exclusive, so S2[0] = 0)

        Notes
        -----
        After prepending 0, array indices align with segment endpoints:
        - S1[t] - S1[s] = sum(x[s:t])
        - S2[t] - S2[s] = sum(x[s:t]²)
        """
        # Compute cumulative sums
        S1 = jnp.cumsum(x)  # S1[i] = x[0] + ... + x[i]
        S2 = jnp.cumsum(x * x)  # S2[i] = x[0]² + ... + x[i]²

        # Prepend 0 for convenient indexing: S[0] = 0 (empty prefix)
        S1 = jnp.concatenate([jnp.array([0.0]), S1])
        S2 = jnp.concatenate([jnp.array([0.0]), S2])

        return S1, S2

    def _seg_suff_from_prefix(
        self,
        S1: jnp.ndarray,
        S2: jnp.ndarray,
        s: jnp.ndarray,
        t: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Extract segment sufficient statistics from prefix sums.

        Computes the sufficient statistics (n, sum, sum_of_squares) for
        segment (s, t] efficiently using prefix arrays.

        Parameters
        ----------
        S1 : jnp.ndarray
            Prefix sum array (from _prefix_sums)
        S2 : jnp.ndarray
            Prefix sum-of-squares array (from _prefix_sums)
        s : jnp.ndarray
            Start index (exclusive)
        t : jnp.ndarray
            End index (inclusive)

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (n, s1, s2) where:
            - n = t - s (segment length)
            - s1 = sum of x[s:t]
            - s2 = sum of x[s:t]²

        Notes
        -----
        Segment notation: (s, t] means indices s+1, s+2, ..., t (half-open)
        """
        n = t - s  # Length of segment
        s1 = S1[t] - S1[s]  # Sum over segment
        s2 = S2[t] - S2[s]  # Sum of squares over segment

        return n, s1, s2

    def _fit_impl(self, x: jnp.ndarray):
        """
        Internal template implementation used by BayesianMCMCBase.

        This method retains the original algorithm while delegating packaging
        and result bookkeeping to the shared base class.
        """
        # -----------------------------------------------------------------
        # Step 1: Standardize data
        # -----------------------------------------------------------------
        x = jnp.asarray(x)
        if self.standardize:
            std = Standardization.from_signal(x, method="robust")
            x_std = std.standardize_signal(x)
        else:
            std = Standardization.identity()
            x_std = x
        self._set_standardization(std)
        T = x_std.shape[0]

        # -----------------------------------------------------------------
        # Step 2: Compute prefix sums for fast segment statistics
        # -----------------------------------------------------------------
        S1, S2 = self._prefix_sums(x_std)

        # -----------------------------------------------------------------
        # Step 3: Build duration prior log probabilities
        # -----------------------------------------------------------------
        # Geometric prior: P(length = n) = h * (1-h)^(n-1)
        # where h = hazard rate
        n_vec = jnp.arange(1, T + 1)  # Possible segment lengths
        log_duration_prior = (
            jnp.log(self.hazard) + (n_vec - 1) * jnp.log1p(-self.hazard)
        )

        # -----------------------------------------------------------------
        # Step 4: Build log-likelihood matrix L[s, t] for all segments
        # -----------------------------------------------------------------
        # L[s, t] = log p(x_{s+1:t}) for segment starting after s, ending at t
        # We use meshgrid to vectorize over all (s, t) pairs

        s_indices = jnp.arange(0, T)  # Start indices: 0..T-1
        t_indices = jnp.arange(1, T + 1)  # End indices: 1..T

        # Create 2D grids: S_grid[i,j] = s_indices[i], T_grid[i,j] = t_indices[j]
        S_grid, T_grid = jnp.meshgrid(s_indices, t_indices, indexing="ij")

        # Segment lengths
        n = T_grid - S_grid

        # Valid mask: only consider segments where t > s (n > 0)
        valid_segments = n > 0

        # Safe lengths (replace invalid with 1 to avoid indexing errors)
        n_safe = jnp.where(valid_segments, n, 1)

        # Extract segment statistics using prefix sums
        s1_segment = jnp.take(S1, T_grid) - jnp.take(S1, S_grid)
        s2_segment = jnp.take(S2, T_grid) - jnp.take(S2, S_grid)

        # Compute log marginal likelihood for each segment
        segment_log_likelihood = _segment_log_marginal(
            n_safe, s1_segment, s2_segment,
            self.mu0, self.kappa0, self.alpha0, self.beta0
        )

        # Apply temperature to likelihood (not duration prior)
        # Higher temp => flatter posterior, lower temp => sharper posterior
        segment_log_likelihood = segment_log_likelihood / self.temp

        # Combine likelihood with duration prior
        # Note: n_safe - 1 gives index into log_duration_prior (0-indexed)
        segment_log_likelihood = segment_log_likelihood + jnp.take(
            log_duration_prior, n_safe - 1
        )

        # Mask invalid segments with -inf (will be ignored in logsumexp)
        L = jnp.where(valid_segments, segment_log_likelihood, -jnp.inf)

        # -----------------------------------------------------------------
        # Step 5: Forward Pass - Compute F[t] = log p(x_{1:t})
        # -----------------------------------------------------------------
        # F[t] represents the total probability of observing data up to time t
        # Recurrence: F[t] = log Σ_s exp(F[s] + L[s, t-1])
        #   (sum over all ways to reach t via a segment ending at t)

        F = -jnp.inf * jnp.ones((T + 1,))
        F = F.at[0].set(0.0)  # Base case: log p(empty sequence) = 0

        def forward_step(t, F_array):
            """
            Forward DP step: compute F[t] from F[0..t-1] and L[:,t-1].
            """
            # Extract column: log likelihoods of all segments ending at t
            segments_ending_at_t = L[:, t - 1]

            # Compute log Σ exp(F[s] + L[s, t-1]) over all s < t
            log_sum = jax.scipy.special.logsumexp(F_array[:T] + segments_ending_at_t)

            return F_array.at[t].set(log_sum)

        # Run forward pass for t = 1, 2, ..., T
        F = jax.lax.fori_loop(1, T + 1, forward_step, F)

        # -----------------------------------------------------------------
        # Step 6: Backward Pass - Compute B[t] = log p(x_{t+1:T} | cp at t)
        # -----------------------------------------------------------------
        # B[t] represents the probability of observing data after t,
        # given that there's a changepoint at t
        # Recurrence: B[t] = log Σ_u exp(L[t, u-1] + B[u])
        #   (sum over all ways to continue from t to end)

        B = -jnp.inf * jnp.ones((T + 1,))
        B = B.at[T].set(0.0)  # Base case: log p(empty future sequence) = 0

        def backward_step(i, B_array):
            """
            Backward DP step: compute B[t] from B[t+1..T] and L[t, :].
            We iterate backwards from t = T-1 to 0.
            """
            t = T - 1 - i  # Convert loop index to time index

            # Extract row: log likelihoods of all segments starting at t
            segments_starting_at_t = L[t, :]  # L[t, u-1] for u > t

            # Compute log Σ exp(L[t, u-1] + B[u]) over all u > t
            # Note: B[1:] aligns with segments ending at u=1..T
            log_sum = jax.scipy.special.logsumexp(
                segments_starting_at_t + B_array[1:]
            )

            return B_array.at[t].set(log_sum)

        # Run backward pass for t = T-1, T-2, ..., 0
        B = jax.lax.fori_loop(0, T, backward_step, B)

        # -----------------------------------------------------------------
        # Step 7: Compute posterior changepoint probabilities
        # -----------------------------------------------------------------
        # log Z = log p(x_{1:T}) = F[T] (total evidence)
        log_Z = F[T]

        # Posterior: P(cp at t | data) = exp(F[t] + B[t] - log Z)
        # We compute for t = 1..T-1 (changepoints between observations)
        log_cp = F[1:T] + B[1:T] - log_Z

        # Exponentiate and clip to [0, 1] for numerical safety
        cp_prob = jnp.clip(jnp.exp(log_cp), 0.0, 1.0)

        diagnostics = {
            "hazard": float(self.hazard),
            "temperature": float(self.temp),
            "n_observations": int(T),
        }
        extras = {
            "posterior_axis": jnp.arange(1, T),
            "forward_messages": F,
            "backward_messages": B,
        }

        return self._finalize_fit(
            posterior=cp_prob,
            log_evidence=log_Z,
            diagnostics=diagnostics,
            extras=extras,
        )
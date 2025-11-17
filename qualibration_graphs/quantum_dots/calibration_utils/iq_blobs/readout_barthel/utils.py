"""
Utility functions for data normalization and readout performance metrics.

This module provides:
- Normalizer1D: Robust standardization of 1D voltage coordinates for numerical stability
- Barthel1DMetricCurves: Computation of fidelity, visibility, and error rates from
  posterior samples using the analytic Barthel model

These tools support the full workflow from data preprocessing through performance evaluation.
"""

from typing import Optional, Dict, Any, Tuple
from .analytic import _std_norm_cdf, triplet_cdf_analytic
import jax.numpy as jnp
import jax
from dataclasses import dataclass


@dataclass
class Normalizer1D:
    """
    Normalizer for 1D data using robust or standard statistics.

    Transforms data to have zero mean and unit scale, improving numerical stability
    for MCMC sampling. Supports two methods:
    - 'mad': Median Absolute Deviation (robust to outliers)
    - 'std': Standard mean and standard deviation (classical)

    Attributes:
        loc: Location parameter (center) for normalization. Subtracted from data.
        scale: Scale parameter (spread) for normalization. Data is divided by this.
    """

    loc: float
    scale: float

    @staticmethod
    def fit(y: jnp.ndarray, method: str = "mad") -> "Normalizer1D":
        """
        Fit a normalizer to data using robust or standard statistics.

        Args:
            y: 1D array of data to normalize
            method: Normalization method:
                   'mad' - Use median and MAD (robust to outliers, recommended)
                   'std' - Use mean and standard deviation (classical)

        Returns:
            Normalizer1D instance with fitted loc and scale

        Raises:
            ValueError: If method is not 'mad' or 'std'
        """
        y = jnp.asarray(y, float).ravel()
        if method == "mad":
            # Median Absolute Deviation: robust to outliers
            med = jnp.median(y)
            mad = jnp.median(jnp.abs(y - med))
            # Scale factor 1.4826 makes MAD consistent with std for Gaussian data
            scale = 1.4826 * (mad if mad > 0 else jnp.std(y, ddof=1) or 1.0)
            return Normalizer1D(loc=float(med), scale=float(scale))
        elif method == "std":
            # Classical mean and standard deviation
            mu = float(jnp.mean(y))
            sd = float(jnp.std(y, ddof=1) or 1.0)
            return Normalizer1D(loc=mu, scale=sd)
        else:
            raise ValueError("method must be 'mad' or 'std'")

    def transform(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize data using fitted location and scale.

        Args:
            y: Data to normalize

        Returns:
            Normalized data: (y - loc) / scale
        """
        return (jnp.asarray(y, float) - self.loc) / self.scale

    def inverse(self, y_norm: jnp.ndarray) -> jnp.ndarray:
        """
        Transform normalized data back to original scale.

        Args:
            y_norm: Normalized data

        Returns:
            Data in original scale: loc + scale * y_norm
        """
        return self.loc + self.scale * jnp.asarray(y_norm, float)

# ---------------- Readout Performance Metrics ----------------


class Barthel1DMetricCurves:
    """
    Compute readout fidelity and visibility curves from Barthel posterior samples.

    This class provides static methods to evaluate readout performance metrics:
    - FS: Singlet fidelity P(classify as S | state is S)
    - FT: Triplet fidelity P(classify as T | state is T)
    - Fidelity: Average F = 0.5 * (FS + FT)
    - Visibility: V = FS + FT - 1

    The metrics are computed by:
    1. Sweeping a classification threshold across a voltage grid
    2. Computing FS/FT at each threshold using the Barthel analytic CDFs
    3. Averaging over posterior samples (posterior predictive) to account for uncertainty
    4. Finding optimal threshold that maximizes fidelity or visibility

    All computations use the analytic Barthel model and are JAX-JIT compiled for speed.
    """

    @staticmethod
    def default_draws() -> int:
        """Default number of posterior draws for posterior predictive averaging."""
        return 64

    @staticmethod
    def default_grid_points() -> int:
        """Default number of threshold points in the voltage grid."""
        return 801

    @staticmethod
    def _prepare_tauM(
        samples: Dict[str, jnp.ndarray], tauM_fixed: Optional[float]
    ) -> jnp.ndarray:
        """
        Extract or construct τ_M values for each posterior sample.

        Args:
            samples: Posterior samples dictionary
            tauM_fixed: Fixed τ_M value if not in samples

        Returns:
            Array of τ_M values, one per posterior sample
        """
        if "tauM" in samples:
            return jnp.asarray(samples["tauM"])
        default_tau = 1.0 if tauM_fixed is None else float(tauM_fixed)
        return jnp.full_like(jnp.asarray(samples["mu_S"]), default_tau)

    @classmethod
    def compute_curves(
        cls,
        samples: Dict[str, jnp.ndarray],
        *,
        tauM_fixed: Optional[float] = None,
        vrf_grid: Optional[jnp.ndarray] = None,
        use_ppd: bool = True,
        draws: Optional[int] = None,
        include_draws: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute FS/FT, fidelity, and visibility curves over a threshold grid.

        Args:
            samples: Posterior dictionary with keys such as `mu_S`, `mu_T`, `sigma`, `pT`, `T1`
                (and optionally `tauM`).
            tauM_fixed: Deterministic τ_M to use when it is absent from `samples`.
            vrf_grid: Optional custom threshold grid; defaults to a ±5σ span around μ_S/μ_T.
            use_ppd: If True, approximate curves by averaging across a subset of posterior draws
                (posterior predictive). If False, evaluate once at the posterior means (faster,
                but ignores uncertainty).
            draws: Number of draws to include when `use_ppd` is True. Falls back to
                `default_draws()` when omitted.
            include_draws: Whether to stash the selected draws in the returned dictionary for
                downstream use (e.g., threshold evaluation).

        Returns:
            Dictionary containing the threshold grid, FS/FT curves, derived fidelity/visibility curves,
            and basic statistics about the posterior means.
        """
        draws = draws if draws is not None else cls.default_draws()

        mu_S_samples = jnp.asarray(samples["mu_S"])
        mu_T_samples = jnp.asarray(samples["mu_T"])
        sigma_samples = jnp.asarray(samples["sigma"])
        T1_samples = jnp.asarray(samples["T1"])
        tauM_samples = cls._prepare_tauM(samples, tauM_fixed)

        mu_S_mean = float(mu_S_samples.mean())
        mu_T_mean = float(mu_T_samples.mean())
        sigma_mean = float(sigma_samples.mean())
        T1_mean = float(T1_samples.mean())
        tauM_mean = float(tauM_samples.mean())

        if vrf_grid is None:
            lo = min(mu_S_mean, mu_T_mean) - 5.0 * sigma_mean
            hi = max(mu_S_mean, mu_T_mean) + 5.0 * sigma_mean
            vrf_grid = jnp.linspace(lo, hi, cls.default_grid_points())
        else:
            vrf_grid = jnp.asarray(vrf_grid, float)

        if use_ppd:
            n = mu_S_samples.shape[0]
            draws_eff = min(draws, n)
            idx = jnp.linspace(0, n - 1, draws_eff).astype(int)
            mu_S_sel = mu_S_samples[idx]
            mu_T_sel = mu_T_samples[idx]
            sigma_sel = sigma_samples[idx]
            T1_sel = T1_samples[idx]
            tauM_sel = tauM_samples[idx]
        else:
            mu_S_sel = jnp.asarray([mu_S_mean])
            mu_T_sel = jnp.asarray([mu_T_mean])
            sigma_sel = jnp.asarray([sigma_mean])
            T1_sel = jnp.asarray([T1_mean])
            tauM_sel = jnp.asarray([tauM_mean])

        @jax.jit
        def _fs_ft_curve(mu_S, mu_T, sigma, T1, tauM):
            FS_curve = _std_norm_cdf((vrf_grid - mu_S) / sigma)
            CDF_T = jax.vmap(lambda v: triplet_cdf_analytic(v, mu_S, mu_T, sigma, T1, tauM))(vrf_grid)
            FT_curve = 1.0 - CDF_T
            return FS_curve, FT_curve

        FS_draws, FT_draws = jax.jit(jax.vmap(_fs_ft_curve))(mu_S_sel, mu_T_sel, sigma_sel, T1_sel, tauM_sel)
        FS_curve = FS_draws.mean(axis=0)
        FT_curve = FT_draws.mean(axis=0)

        result: Dict[str, Any] = {
            "vrf_grid": vrf_grid,
            "FS_curve": FS_curve,
            "FT_curve": FT_curve,
            "fidelity_curve": 0.5 * (FS_curve + FT_curve),
            "visibility_curve": FS_curve + FT_curve - 1.0,
            "mu_S_mean": mu_S_mean,
            "mu_T_mean": mu_T_mean,
        }

        if include_draws:
            result["_draws"] = {
                "mu_S": mu_S_sel,
                "mu_T": mu_T_sel,
                "sigma": sigma_sel,
                "T1": T1_sel,
                "tauM": tauM_sel,
            }

        return result

    @staticmethod
    def _evaluate_threshold(
        threshold: float, draws: Dict[str, jnp.ndarray]
    ) -> Tuple[float, float]:
        """
        Evaluate FS and FT at a specific threshold using posterior predictive averaging.

        Args:
            threshold: Voltage threshold at which to evaluate metrics
            draws: Dictionary of selected posterior draws (from compute_curves)

        Returns:
            Tuple of (FS, FT) probabilities averaged over posterior draws
        """
        thr = jnp.asarray(threshold, float)

        @jax.jit
        def _fs_ft_point(mu_S, mu_T, sigma, T1, tauM):
            # FS = P(V < threshold | S) - singlet correctly classified as singlet
            FS = _std_norm_cdf((thr - mu_S) / sigma)
            # FT = P(V > threshold | T) - triplet correctly classified as triplet
            FT = 1.0 - triplet_cdf_analytic(thr, mu_S, mu_T, sigma, T1, tauM)
            return FS, FT

        # Evaluate at each posterior draw and average
        FS_vals, FT_vals = jax.vmap(_fs_ft_point)(
            draws["mu_S"], draws["mu_T"], draws["sigma"], draws["T1"], draws["tauM"]
        )
        return float(FS_vals.mean()), float(FT_vals.mean())

    @classmethod
    def summarize_metric(
        cls,
        samples: Dict[str, jnp.ndarray],
        *,
        metric: str = "fidelity",
        tauM_fixed: Optional[float] = None,
        vrf_grid: Optional[jnp.ndarray] = None,
        use_ppd: bool = True,
        draws: Optional[int] = None,
        vrf: Optional[float] = None,
        return_aligned_curve: bool = True,
        return_components: bool = False,
    ) -> Dict[str, Any]:
        """
        Summarise the requested metric (`'fidelity'` or `'visibility'`) using the pre-computed curves.

        Args:
            samples: Posterior dictionary (same requirements as `compute_curves`).
            metric: Which metric to optimise; choose `'fidelity'` or `'visibility'`.
            tauM_fixed, vrf_grid, use_ppd, draws: Forwarded to `compute_curves`. Setting `use_ppd=False`
                skips posterior predictive averaging and instead uses the posterior means directly
                (faster but ignores posterior uncertainty).
            vrf: Optional explicit threshold at which to evaluate the metrics; when omitted, the optimal
                threshold for the requested metric is selected from the grid.
            return_aligned_curve: When `metric == 'fidelity'`, whether to include sign-aligned curves.
            return_components: Whether to include raw FS/FT curves in the output.

        Returns:
            Dictionary containing optimal metric values, FS/FT probabilities, error rates, and any
            requested supporting curves.
        """
        curves = cls.compute_curves(
            samples,
            tauM_fixed=tauM_fixed,
            vrf_grid=vrf_grid,
            use_ppd=use_ppd,
            draws=draws,
            include_draws=True,
        )

        metric_key = "fidelity_curve" if metric == "fidelity" else "visibility_curve"
        target_curve = jnp.asarray(curves[metric_key])
        grid = jnp.asarray(curves["vrf_grid"])

        if vrf is None:
            idx_opt = int(jnp.argmax(target_curve))
            vrf_opt = float(grid[idx_opt])
            FS_opt = float(curves["FS_curve"][idx_opt])
            FT_opt = float(curves["FT_curve"][idx_opt])
        else:
            vrf_opt = float(vrf)
            FS_opt, FT_opt = cls._evaluate_threshold(vrf_opt, curves["_draws"])

        fidelity_opt_val = 0.5 * (FS_opt + FT_opt)
        visibility_opt_val = FS_opt + FT_opt - 1.0

        result: Dict[str, Any] = {
            "FS": float(FS_opt),
            "FT": float(FT_opt),
            "eps_S": float(1.0 - FS_opt),
            "eps_T": float(1.0 - FT_opt),
            "fidelity_opt": float(fidelity_opt_val),
            "visibility_opt": float(visibility_opt_val),
        }

        if return_components:
            result["FS_curve"] = curves["FS_curve"]
            result["FT_curve"] = curves["FT_curve"]

        if metric == "fidelity":
            fidelity_curve = curves["fidelity_curve"]
            s_align = 1.0 if curves["mu_T_mean"] >= curves["mu_S_mean"] else -1.0
            result["vrf"] = grid
            result["fidelity"] = fidelity_curve
            result["vrf_opt"] = vrf_opt
            result["fidelity_opt"] = float(fidelity_opt_val)
            result["sign_align"] = s_align
            result["vrf_opt_aligned"] = s_align * vrf_opt
            if return_aligned_curve:
                result["vrf_aligned"] = s_align * grid
                result["fidelity_aligned"] = fidelity_curve
        else:
            result["vrf"] = float(vrf_opt)
            result["visibility"] = float(visibility_opt_val)
            result["fidelity"] = float(fidelity_opt_val)
            result["vrf_grid"] = grid
            result["visibility_curve"] = curves["visibility_curve"]
            result["fidelity_curve"] = curves["fidelity_curve"]

        return result

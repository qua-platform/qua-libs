"""
Calibration and fitting pipeline for the Barthel readout model from IQ data.

This module orchestrates the complete workflow for fitting the Barthel two-state readout
model to experimental IQ measurements. It handles:

1. **PCA projection**: Reduce 2D IQ data to 1D voltage coordinates
2. **Calibration**: Use pure-state measurements (S or T) to set informative priors
3. **Normalization**: Standardize coordinates for numerical stability
4. **MCMC fitting**: Bayesian parameter estimation via NUTS sampling
5. **Prior construction**: Automatic prior setup from calibration data

The main interface is the `Barthel1DFromIQ` class, which provides a high-level `.fit()`
method that returns posterior samples for downstream analysis (fidelity, visibility, etc.).

Typical usage:
    >>> from readout_barthel.calibrate import Barthel1DFromIQ
    >>> # X: IQ data (N, 2), X_calib: pure singlet shots for calibration
    >>> y, proj, normalizer, mcmc, samples, _, calib = Barthel1DFromIQ.fit(
    ...     X, calib=(X_calib, "S"), fix_tau_M=1.0
    ... )
    >>> print(samples.keys())  # ['mu_S', 'mu_T', 'sigma', 'T1', 'pT', ...]
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Dict, Any, Callable

from readout_barthel.fit import fit_model, MCMCConfig
from readout_barthel.models.barthel_model import build_barthel_model_1d_analytic
from readout_barthel.pca import pca_project_1d, PCAProjection
from readout_barthel.utils import Normalizer1D


@dataclass
class CalibrationResult:
    """
    Statistics and priors extracted from calibration measurements.

    When pure singlet or triplet measurements are provided, this structure
    captures the learned location/scale parameters and uses them to construct
    informative priors for the Barthel model fit.

    Attributes:
        which: Which state was calibrated ('S' or 'T')
        mu_known: Mean voltage of the known (calibrated) state
        sigma_known: Standard deviation of the known state
        mu_other: Estimated mean of the other (unknown) state from k-means clustering
        sigma_other: Estimated std of the other state from clustering
        priors: Dictionary of prior hyperparameters constructed from calibration
        sign_align: PCA sign orientation (+1 or -1) determined by calibration
    """

    which: str
    mu_known: float
    sigma_known: float
    mu_other: float
    sigma_other: float
    priors: Dict[str, Any]
    sign_align: float

def _fit_gaussian_1d(y: jnp.ndarray) -> Tuple[float, float]:
    """
    Fit Gaussian parameters (mean and std) to 1D data.

    Simple moment-based estimation for a single Gaussian distribution.

    Args:
        y: 1D array of voltage measurements

    Returns:
        Tuple of (mean, std_dev)
    """
    y = jnp.asarray(y, float).ravel()
    mu = float(y.mean())
    sd = float(y.std(ddof=1) if y.size > 1 else 1e-3)
    return mu, sd


def _kmeans_1d_two_clusters(
    y: jnp.ndarray, iters: int = 30
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Cluster 1D data into two groups using k-means.

    Used during calibration to identify the singlet and triplet peaks in
    mixed-state measurements when only one state is known from calibration data.

    Args:
        y: 1D array of projected voltage measurements
        iters: Maximum number of k-means iterations

    Returns:
        Tuple of (centers, stds) where:
            centers: Array of 2 cluster centers, sorted in ascending order
            stds: Array of 2 cluster standard deviations, matching center order
    """
    y = jnp.asarray(y, float).ravel()
    centers, stds = _kmeans_1d_two_clusters_compiled(y, iters)
    return centers.astype(float), stds.astype(float)


@partial(jax.jit, static_argnums=(1,))
def _kmeans_1d_two_clusters_compiled(y: jnp.ndarray, iters: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled helper performing the 1D two-cluster k-means iterations."""
    dtype = y.dtype
    initial_centers = jnp.quantile(y, jnp.array([0.3, 0.7], dtype=dtype))

    def update_centers(c: jnp.ndarray) -> jnp.ndarray:
        d0 = (y - c[0])**2
        d1 = (y - c[1])**2
        mask0 = d0 <= d1
        mask1 = jnp.logical_not(mask0)

        count0 = mask0.sum()
        count1 = mask1.sum()
        denom0 = jnp.maximum(count0, 1).astype(dtype)
        denom1 = jnp.maximum(count1, 1).astype(dtype)
        sum0 = jnp.sum(jnp.where(mask0, y, 0.0))
        sum1 = jnp.sum(jnp.where(mask1, y, 0.0))

        c0 = jnp.where(count0 > 0, sum0 / denom0, c[0])
        c1 = jnp.where(count1 > 0, sum1 / denom1, c[1])
        return jnp.array([c0, c1], dtype=dtype)

    def body(_, state):
        c, done = state

        def run_update(current_state):
            current_c, _ = current_state
            new_c = update_centers(current_c)
            delta = jnp.max(jnp.abs(new_c - current_c))
            converged = delta < 1e-6
            return (new_c, converged)

        new_c, converged = run_update(state)
        next_c = jnp.where(done, c, new_c)
        next_done = jnp.logical_or(done, converged)
        return (next_c, next_done)

    centers, _ = jax.lax.fori_loop(0, iters, body, (initial_centers, jnp.array(False)))

    def cluster_std(mask: jnp.ndarray, center: jnp.ndarray, fallback: jnp.ndarray) -> jnp.ndarray:
        count = mask.sum()
        weight = mask.astype(dtype)
        residual = (y - center)**2
        sumsq = jnp.sum(residual * weight)
        denom = jnp.maximum(count - 1, 1).astype(dtype)
        std = jnp.sqrt(sumsq / denom)
        return jnp.where(count > 1, std, fallback)

    mask0 = (y - centers[0])**2 <= (y - centers[1])**2
    mask1 = jnp.logical_not(mask0)
    global_std = jnp.std(y, ddof=1)
    std0 = cluster_std(mask0, centers[0], global_std)
    std1 = cluster_std(mask1, centers[1], global_std)

    order = jnp.argsort(centers)
    return centers[order], jnp.stack([std0, std1])[order]

def _calibrate_from_single_state(
    X_mixed: jnp.ndarray,
    X_calib: jnp.ndarray,
    which: str,
    prior_strength: float = 0.3,
    sigma_scale_default: float = 0.3,
) -> Tuple[CalibrationResult, PCAProjection, jnp.ndarray]:
    """
    Calibrate priors and PCA orientation using pure-state calibration measurements.

    Uses measurements from a known pure state (either all-singlet or all-triplet)
    to determine PCA sign orientation and construct informative priors for MCMC fitting.

    The calibration process:
    1. Compute PCA from mixed measurements (without orientation)
    2. Project calibration data onto the PC axis
    3. Determine sign: orient so calibrated state is on correct side
       (S→negative, T→positive by convention)
    4. Fit Gaussian to calibration data → known state parameters
    5. Run k-means(k=2) on mixed data → identify unknown state peak
    6. Construct priors centered at empirical estimates with controlled width

    Args:
        X_mixed: Mixed IQ measurements (N, 2) containing both S and T states
        X_calib: Pure calibration IQ measurements (M, 2) from a single known state
        which: Which state is calibrated: 'S' (singlet) or 'T' (triplet)
        prior_strength: Tightness of priors relative to empirical spread.
                       0.3 = priors centered at empirical values with 30% of empirical width
        sigma_scale_default: Prior width for readout noise σ

    Returns:
        Tuple of (CalibrationResult, PCAProjection, y_mixed) where:
            CalibrationResult contains empirical parameters and constructed priors
            PCAProjection has the calibrated sign orientation
            y_mixed is the 1D projected mixed data with correct sign
    """
    which = which.upper()
    assert which in ("S", "T")
    # PCA axis from mixed; sign decided by calibration set
    y_mixed_tmp, proj = pca_project_1d(X_mixed, labels=None, orient="none")
    y_cal_tmp = (jnp.asarray(X_calib, float) - proj.mean[None, :]) @ proj.pc1
    if which == "T":
        sign_align = 1.0 if y_cal_tmp.mean() >= 0 else -1.0
    else:  # 'S'
        sign_align = -1.0 if y_cal_tmp.mean() >= 0 else 1.0
    proj.sign = sign_align

    y_mixed = y_mixed_tmp * sign_align
    y_cal = y_cal_tmp * sign_align

    mu_known, sd_known = _fit_gaussian_1d(y_cal)
    centers, stds = _kmeans_1d_two_clusters(y_mixed)
    idx_known = int(jnp.argmin(jnp.abs(centers - mu_known)))
    idx_other = 1 - idx_known
    mu_other, sd_other = float(centers[idx_other]), float(stds[idx_other])

    # Priors for μS, μT with ordering; use delta scale ~ std of their separation
    mu_S_loc = mu_known if which == "S" else mu_other
    mu_T_loc = mu_other if which == "S" else mu_known  # only used when enforce_order=False
    mu_S_scale = max(prior_strength * (sd_known if which == "S" else sd_other), 1e-3)
    mu_T_scale = max(prior_strength * (sd_other if which == "S" else sd_known), 1e-3)
    delta_scale = max(prior_strength * abs(mu_T_loc - mu_S_loc), 1e-3)

    priors = {
        "mu_S_loc": mu_S_loc,
        "mu_S_scale": mu_S_scale,
        "mu_T_loc": mu_T_loc,      # kept for completeness (unused if enforce_order=True)
        "mu_T_scale": mu_T_scale,  # kept for completeness
        "delta_scale": delta_scale,
        "sigma_scale": sigma_scale_default,
    }

    return CalibrationResult(
        which=which,
        mu_known=mu_known, sigma_known=sd_known,
        mu_other=mu_other, sigma_other=sd_other,
        priors=priors,
        sign_align=sign_align,
    ), proj, y_mixed

class Barthel1DFromIQ:
    """
    High-level interface for fitting the Barthel readout model to IQ data.

    This class provides a streamlined workflow that handles the complete pipeline
    from raw 2D IQ measurements to posterior parameter samples:

    1. PCA projection: X(N,2) → y(N,)
    2. Optional calibration: use pure-S or pure-T shots for priors
    3. Normalization: standardize y for MCMC stability
    4. MCMC fitting: sample posterior with NUTS
    5. Return everything needed for downstream analysis

    All methods are static, making this a stateless facade around the fitting pipeline.
    """

    @staticmethod
    def default_priors() -> Dict[str, Any]:
        """
        Return a new empty dictionary for prior overrides.

        Users can populate this with custom prior hyperparameters before fitting.
        If calibration is used, calibration priors will be added automatically.

        Returns:
            Empty mutable dictionary
        """
        return {}

    @staticmethod
    def default_mcmc_config() -> MCMCConfig:
        """
        Return the default MCMC configuration.

        Default settings: 500 warmup, 500 samples, 1 chain, target_accept=0.8

        Returns:
            MCMCConfig with default parameters
        """
        return MCMCConfig()

    @classmethod
    def fit(
        cls,
        X: jnp.ndarray,
        priors: Optional[Dict[str, Any]] = None,
        mcmc_config: Optional[MCMCConfig] = None,
        fix_tau_M: float = 1.0,
        labels: Optional[jnp.ndarray] = None,
        orient: str = "auto",
        *,
        calib: Optional[Tuple[jnp.ndarray, str]] = None,
        prior_strength: float = 0.3,
        sigma_scale_default: float = 0.3,
        normalize: bool = True,
        norm_method: str = "mad",
    ) -> Tuple[
        jnp.ndarray,
        PCAProjection,
        Normalizer1D,
        Any,
        Any,
        Any,
        Optional[CalibrationResult],
    ]:
        """
        Fit the Barthel two-state readout model to IQ measurements.

        Complete workflow from raw IQ data to posterior samples. Handles PCA projection,
        optional calibration for informative priors, normalization for numerical stability,
        and MCMC sampling via NUTS.

        Args:
            X: IQ measurements, shape (N, 2) where N is number of shots.
               Columns are I and Q quadratures.
            priors: Optional dictionary of prior hyperparameters to override defaults.
                   If calibration is used, calibration priors are added/updated automatically.
            mcmc_config: MCMC configuration (warmup, samples, chains, etc.).
                        Uses defaults if None.
            fix_tau_M: Measurement duration τ_M (fixed, not fitted). Typically in
                      units consistent with T₁. Default 1.0.
            labels: Optional binary labels {0,1} for PCA orientation. If provided,
                   PCA is oriented so label=1 (triplet) > label=0 (singlet).
                   Ignored if calib is provided.
            orient: PCA orientation mode when labels=None and calib=None:
                   'auto' - heuristic based on tail asymmetry
                   'none' - keep natural SVD orientation
            calib: Optional calibration tuple (X_calib, which) where:
                  X_calib: Pure-state IQ measurements, shape (M, 2)
                  which: 'S' (singlet) or 'T' (triplet)
                  Calibration overrides labels/orient and sets informative priors.
            prior_strength: Tightness of calibration priors relative to empirical spread.
                          0.3 = 30% of empirical width. Only used if calib is provided.
            sigma_scale_default: Prior scale for readout noise σ. Default 0.3.
            normalize: Whether to standardize projected coordinates for MCMC stability.
                      Recommended: True.
            norm_method: Normalization method if normalize=True:
                        'mad' - robust (median + MAD, recommended)
                        'std' - classical (mean + std)

        Returns:
            Tuple of (y, proj, normalizer, mcmc, samples, summary, calib_result):
                y: Projected 1D coordinates (NOT normalized), shape (N,)
                proj: PCAProjection object for projecting new data
                normalizer: Normalizer1D for transforming to normalized coordinates
                mcmc: NumPyro MCMC object with full sampling state
                samples: Dictionary of posterior samples (in normalized coordinates):
                        {'mu_S', 'mu_T', 'sigma', 'T1', 'pT', 'delta', ...}
                summary: Same as mcmc (for backward compatibility)
                calib_result: CalibrationResult if calib was provided, else None

        """
        priors = priors if priors is not None else cls.default_priors()
        mcmc_config = mcmc_config if mcmc_config is not None else cls.default_mcmc_config()

        y, proj, priors, calib_result = cls._project_and_calibrate(
            X,
            priors,
            labels=labels,
            orient=orient,
            calib=calib,
            prior_strength=prior_strength,
            sigma_scale_default=sigma_scale_default,
        )

        y_fit, normalizer = cls._normalize_coordinate(y, normalize=normalize, norm_method=norm_method)
        cls._apply_normalizer_to_priors(priors, normalizer, sigma_scale_default=sigma_scale_default)

        model_factory = cls._build_model_factory(fix_tau_M=fix_tau_M)
        mcmc, samples, summary = fit_model(model_factory, y_fit, priors, mcmc_config)

        return y, proj, normalizer, mcmc, samples, summary, calib_result

    @staticmethod
    def _project_and_calibrate(
        X: jnp.ndarray,
        priors: Dict[str, Any],
        *,
        labels: Optional[jnp.ndarray],
        orient: str,
        calib: Optional[Tuple[jnp.ndarray, str]],
        prior_strength: float,
        sigma_scale_default: float,
    ) -> Tuple[jnp.ndarray, PCAProjection, Dict[str, Any], Optional[CalibrationResult]]:
        """Project IQ data to 1D and optionally align it using calibration data."""
        if calib is not None:
            X_calib, which = calib
            calib_result, proj, y_mixed = _calibrate_from_single_state(
                X_mixed=jnp.asarray(X, float),
                X_calib=jnp.asarray(X_calib, float),
                which=which,
                prior_strength=prior_strength,
                sigma_scale_default=sigma_scale_default,
            )
            priors.update(calib_result.priors)
            return y_mixed, proj, priors, calib_result

        y, proj = pca_project_1d(X, labels=labels, orient=orient)
        return y, proj, priors, None

    @staticmethod
    def _normalize_coordinate(
        y: jnp.ndarray,
        *,
        normalize: bool,
        norm_method: str,
    ) -> Tuple[jnp.ndarray, Normalizer1D]:
        """Normalize the 1D coordinate to improve numerical stability."""
        if normalize:
            normalizer = Normalizer1D.fit(y, method=norm_method)
            return normalizer.transform(y), normalizer
        normalizer = Normalizer1D(loc=0.0, scale=1.0)
        return y, normalizer

    @staticmethod
    def _apply_normalizer_to_priors(
        priors: Dict[str, Any],
        normalizer: Normalizer1D,
        *,
        sigma_scale_default: float,
    ) -> Dict[str, Any]:
        """Convert prior parameters into the normalized coordinate system."""
        loc = normalizer.loc
        scale = normalizer.scale
        if "mu_S_loc" in priors:
            priors["mu_S_loc"] = (priors["mu_S_loc"] - loc) / scale
        if "mu_T_loc" in priors:
            priors["mu_T_loc"] = (priors["mu_T_loc"] - loc) / scale
        if "mu_S_scale" in priors:
            priors["mu_S_scale"] /= scale
        if "mu_T_scale" in priors:
            priors["mu_T_scale"] /= scale
        if "delta_scale" in priors:
            priors["delta_scale"] /= scale
        priors["sigma_scale"] = priors.get("sigma_scale", sigma_scale_default) / scale
        return priors

    @staticmethod
    def _build_model_factory(
        *,
        fix_tau_M: float,
    ) -> Callable[[Dict[str, Any]], Any]:
        """Create a Barthel model factory for the analytic 1D model."""
        tau_M_value = float(fix_tau_M)
        return lambda p: build_barthel_model_1d_analytic(
            p, fix_tau_M=tau_M_value, enforce_order=True
        )

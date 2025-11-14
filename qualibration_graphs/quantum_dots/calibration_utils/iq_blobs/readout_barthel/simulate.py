from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from numpy.random import Generator, default_rng
from scipy.stats import truncnorm

@dataclass
class SimulationParams:
    n_samples: int = 10_000
    p_triplet: float = 0.5          # ⟨P_T⟩, preparation probability of triplet
    V_S_rf: float = 0.0             # Mean voltage for singlet
    V_T_rf: float = 1.0             # Mean voltage for triplet (no decay)
    sigma: float = 0.1              # Measurement noise std (assumed equal for S/T)
    tau_M: float = 1.0              # Measurement window duration
    T1: float = 2.0                 # Relaxation time from T→S during measurement

def _sample_truncated_exponential(rng: Generator, low: float, high: float, lam: float, size: int) -> np.ndarray:
    """Sample from a truncated exponential on [low, high] with density ∝ exp(-lam*(x-low)).
    Equivalent to inverse-CDF sampling on a bounded interval.
    """
    if lam <= 0:
        # uniform fallback if lam ~ 0
        return rng.uniform(low, high, size=size)
    width = high - low
    # CDF at upper bound:
    Z = 1.0 - np.exp(-lam * width)
    u = rng.uniform(size=size)
    return low - (1.0/lam) * np.log(1.0 - u * Z)

def simulate_readout(params: SimulationParams, rng: Optional[Generator] = None, return_labels: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Simulate single-shot readout voltages according to Barthel-style model.

    For an initially prepared T state:
      - With probability exp(-tau_M/T1), no decay during measurement → Gaussian at V_T_rf.
      - Otherwise, an ideal voltage U is drawn from a truncated exponential over [V_S_rf, V_T_rf]
        with rate lambda = tau_M / (T1 * ΔV_rf). The observed voltage is then N(U, sigma).
    For an initially prepared S state: Gaussian at V_S_rf.

    Returns an array of shape (n_samples,) and optional labels array in {0:S,1:T}.
    """
    if rng is None:
        rng = default_rng()

    n = params.n_samples
    V_S, V_T, s, tau_M, T1 = params.V_S_rf, params.V_T_rf, params.sigma, params.tau_M, params.T1
    dV = V_T - V_S
    if dV <= 0:
        raise ValueError("Require V_T_rf > V_S_rf for a meaningful model.")
    lam = tau_M / (T1 * dV)  # matches the exponential factor in the notes

    # Prepare initial states
    labels = rng.uniform(size=n) < params.p_triplet   # True means initial T, False means S
    x = np.empty(n, dtype=float)

    # Singlets
    idx_S = np.where(~labels)[0]
    x[idx_S] = rng.normal(loc=V_S, scale=s, size=idx_S.size)

    # Triplets
    idx_T = np.where(labels)[0]
    if idx_T.size > 0:
        p_no_decay = np.exp(-tau_M / T1)
        no_decay = rng.uniform(size=idx_T.size) < p_no_decay
        idx_T_no = idx_T[no_decay]
        idx_T_de = idx_T[~no_decay]

        # No decay: Gaussian at V_T
        if idx_T_no.size > 0:
            x[idx_T_no] = rng.normal(loc=V_T, scale=s, size=idx_T_no.size)

        # Decay-in-flight: ideal voltage from truncated exponential over [V_S, V_T], then Gaussian noise
        if idx_T_de.size > 0:
            U = _sample_truncated_exponential(rng, low=V_S, high=V_T, lam=lam, size=idx_T_de.size)
            x[idx_T_de] = rng.normal(loc=U, scale=s, size=idx_T_de.size)

    if return_labels:
        return x, labels.astype(int)
    return x, None

# --- add these imports if not already present ---
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from numpy.random import Generator, default_rng

# --- NEW: simulation params for IQ ---
@dataclass
class SimulationParamsIQ:
    n_samples: int = 10_000
    p_triplet: float = 0.5              # P(T) at state preparation
    mu_S: Tuple[float, float] = (0.0, 0.0)   # mean IQ for singlet
    mu_T: Tuple[float, float] = (1.0, 0.2)   # mean IQ for triplet (no decay)
    sigma_I: float = 0.1                # noise std along I
    sigma_Q: float = 0.1                # noise std along Q
    rho: float = 0.0                    # correlation between I and Q (-1 < rho < 1)
    tau_M: float = 1.0                  # measurement window
    T1: float = 2.0                     # relaxation time T→S

def _chol_from_sigmas(sigma_I: float, sigma_Q: float, rho: float) -> np.ndarray:
    """Cholesky factor for 2x2 covariance from (sigma_I, sigma_Q, rho)."""
    if not (-0.999 < rho < 0.999):
        raise ValueError("rho must lie in (-1,1) for a valid covariance.")
    cov = np.array([
        [sigma_I**2, rho * sigma_I * sigma_Q],
        [rho * sigma_I * sigma_Q, sigma_Q**2],
    ], dtype=float)
    return np.linalg.cholesky(cov)

def _sample_trunc_exp_time(rng: Generator, tau_M: float, T1: float, size: int) -> np.ndarray:
    """Sample decay times t in [0, tau_M) from an exponential with rate 1/T1 truncated at tau_M,
    i.e. conditional on at least one decay before tau_M."""
    if T1 <= 0:
        # Degenerate: immediate decay → t ≈ 0
        return np.zeros(size, dtype=float)
    Z = 1.0 - np.exp(-tau_M / T1)  # CDF(tau_M)
    u = rng.uniform(size=size)
    return -T1 * np.log(1.0 - u * Z)

def simulate_readout_iq(params: SimulationParamsIQ,
                        rng: Optional[Generator] = None,
                        return_labels: bool = False):
    """Simulate I/Q single-shot readouts for a two-state system with T→S decay during readout.

    Model:
      - Prepare S with prob 1-p_T, T with prob p_T.
      - If initial S: Y ~ N(μ_S, Σ).
      - If initial T: with prob exp(-τ_M/T1) no decay → Y ~ N(μ_T, Σ).
        Else, decay at time t∼TruncExp(rate=1/T1 on [0,τ_M)) → ideal mean
            μ*(t) = μ_S + (t/τ_M) (μ_T - μ_S)  (linear interpolation),
        and Y ~ N(μ*(t), Σ).

    Returns:
      X : (n_samples, 2) array
      labels : (n_samples,) in {0:S,1:T} if return_labels=True else None
    """
    if rng is None:
        rng = default_rng()

    n = params.n_samples
    mu_S = np.asarray(params.mu_S, dtype=float)
    mu_T = np.asarray(params.mu_T, dtype=float)
    if mu_S.shape != (2,) or mu_T.shape != (2,):
        raise ValueError("mu_S and mu_T must be length-2 sequences.")

    # Shared noise for both states
    L = _chol_from_sigmas(params.sigma_I, params.sigma_Q, params.rho)

    # Preparation
    is_T = rng.uniform(size=n) < params.p_triplet
    X = np.empty((n, 2), dtype=float)

    # S shots
    idx_S = np.where(~is_T)[0]
    if idx_S.size > 0:
        eps = rng.normal(size=(idx_S.size, 2)) @ L.T
        X[idx_S] = mu_S + eps

    # T shots
    idx_T = np.where(is_T)[0]
    if idx_T.size > 0:
        p_no = np.exp(-params.tau_M / params.T1) if params.T1 > 0 else 0.0
        no_decay = rng.uniform(size=idx_T.size) < p_no
        idx_no = idx_T[no_decay]
        idx_de = idx_T[~no_decay]

        # No-decay: around μ_T
        if idx_no.size > 0:
            eps = rng.normal(size=(idx_no.size, 2)) @ L.T
            X[idx_no] = mu_T + eps

        # Decay-in-flight: μ*(t) along the μ_S ↔ μ_T line
        if idx_de.size > 0:
            t = _sample_trunc_exp_time(rng, params.tau_M, params.T1, size=idx_de.size)
            alpha = (t / params.tau_M).reshape(-1, 1)  # in [0,1)
            mu_line = mu_S + alpha * (mu_T - mu_S)
            eps = rng.normal(size=(idx_de.size, 2)) @ L.T
            X[idx_de] = mu_line + eps

    return (X, is_T.astype(int)) if return_labels else (X, None)

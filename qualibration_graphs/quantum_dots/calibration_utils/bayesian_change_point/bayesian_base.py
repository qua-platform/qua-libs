from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import jax.numpy as jnp


@dataclass
class FitResult:
    posterior: jnp.ndarray
    log_evidence: Any
    diagnostics: Dict[str, Any]
    extras: Dict[str, Any]


class BayesianMCMCBase:
    """
    Shared template for Bayesian MCMC-style estimators.

    Subclasses implement `_fit_impl` and use `_finalize_fit` to package the
    result. The public `fit()` method returns a tuple consisting of
    ``(posterior, log_evidence)`` to mirror the historical `BayesianCP.fit`
    contract, while the complete result is stored in `self.last_result`.
    """

    def __init__(self, *, standardize: bool = True):
        self.standardize = bool(standardize)
        self.standardization: Optional[Any] = None
        self.last_result: Optional[FitResult] = None

    # ------------------------------------------------------------------ #
    def fit(self, *args, **kwargs):
        """
        Run inference and return ``posterior, log_evidence``.

        The full result (including diagnostics/extras) is available via
        `get_last_result()`.
        """
        result = self._fit_impl(*args, **kwargs)
        self.last_result = result
        return result.posterior, result.log_evidence

    # ------------------------------------------------------------------ #
    def _fit_impl(self, *args, **kwargs) -> FitResult:
        raise NotImplementedError("_fit_impl must be implemented by subclasses.")

    def _set_standardization(self, standardization: Any):
        self.standardization = standardization

    def _finalize_fit(
        self,
        posterior: Any,
        log_evidence: Any,
        *,
        diagnostics: Optional[Dict[str, Any]] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        posterior_arr = jnp.asarray(posterior)
        diagnostics = diagnostics or {}
        extras = extras or {}

        if self.standardization is not None:
            if hasattr(self.standardization, "to_dict"):
                std_dict = self.standardization.to_dict()
            else:
                try:
                    std_dict = asdict(self.standardization)
                except TypeError:
                    # Fall back to storing the object directly if asdict fails
                    std_dict = self.standardization
            extras = {"standardization": std_dict, **extras}

        return FitResult(
            posterior=posterior_arr,
            log_evidence=_maybe_python_scalar(log_evidence),
            diagnostics=diagnostics,
            extras=extras,
        )

    # ------------------------------------------------------------------ #
    def get_last_result(self, *, flatten: bool = False):
        """
        Retrieve the result from the latest `fit` call.

        Parameters
        ----------
        flatten : bool, default=False
            When True, merge `posterior`, `log_evidence`, and `diagnostics`
            into a single dictionary alongside `extras`.
        """
        if self.last_result is None:
            raise RuntimeError("No fit() call has been executed yet.")

        if not flatten:
            return self.last_result

        result = {
            "posterior": self.last_result.posterior,
            "log_evidence": self.last_result.log_evidence,
            "diagnostics": self.last_result.diagnostics,
        }
        result.update(self.last_result.extras)
        return result


def _maybe_python_scalar(value: Any) -> Any:
    """
    Convert 0-D JAX arrays to Python floats when possible without forcing
    traced values during vmapped execution.
    """
    try:
        arr = jnp.asarray(value)
    except Exception:
        return value

    if arr.ndim == 0:
        try:
            return float(arr)
        except TypeError:
            return arr
    return arr

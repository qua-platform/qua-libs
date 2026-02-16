from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import jax.numpy as jnp


@dataclass
class Standardization:
    """
    Shared utility for standardizing x/y data pairs or standalone signals.

    Parameters
    ----------
    x_mu, x_std, y_mu, y_std
        Location and scale parameters used for normalization. For 1D signals
        (no explicit x-axis), `x_mu`/`x_std` fall back to 0/1.
    method : str
        Name of the statistic used to compute location/scale. Supported values:
        - ``moment``: mean / standard deviation (default)
        - ``robust``: median / MAD (scaled by 1.4826)
        - ``identity``: no scaling (mu=0, std=1)
    """

    x_mu: Union[float, jnp.ndarray]
    x_std: Union[float, jnp.ndarray]
    y_mu: Union[float, jnp.ndarray]
    y_std: Union[float, jnp.ndarray]
    method: str = "moment"

    # --------------------------------------------------------------------- #
    # Constructors
    # --------------------------------------------------------------------- #
    @classmethod
    def from_data(
        cls,
        x: jnp.ndarray,
        y: jnp.ndarray,
        *,
        method: str = "moment",
    ) -> "Standardization":
        """
        Build a standardization object for paired (x, y) observations.
        """
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        x_mu, x_std = _location_scale(x, method)
        y_mu, y_std = _location_scale(y, method)
        return cls(x_mu=x_mu, x_std=x_std, y_mu=y_mu, y_std=y_std, method=method)

    @classmethod
    def from_signal(
        cls,
        y: jnp.ndarray,
        *,
        method: str = "robust",
    ) -> "Standardization":
        """
        Build a standardization object for a 1D signal.
        """
        y = jnp.asarray(y)
        _, _ = _location_scale(jnp.arange(y.shape[0]), "identity")
        y_mu, y_std = _location_scale(y, method)
        return cls(x_mu=0.0, x_std=1.0, y_mu=y_mu, y_std=y_std, method=method)

    @classmethod
    def identity(cls) -> "Standardization":
        """
        No-op standardization (useful when normalization is disabled).
        """
        return cls(x_mu=0.0, x_std=1.0, y_mu=0.0, y_std=1.0, method="identity")

    # --------------------------------------------------------------------- #
    # Transformations
    # --------------------------------------------------------------------- #
    def standardize(self, x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        xp = (x - self.x_mu) / self._safe_scale(self.x_std)
        yp = (y - self.y_mu) / self._safe_scale(self.y_std)
        return xp, yp

    def standardize_signal(self, y: jnp.ndarray) -> jnp.ndarray:
        return (y - self.y_mu) / self._safe_scale(self.y_std)

    def unstandardize_linear(
        self,
        b0p: jnp.ndarray,
        b1p: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        slope = self.y_std * b1p / self._safe_scale(self.x_std)
        intercept = self.y_mu + self.y_std * (b0p - b1p * self.x_mu / self._safe_scale(self.x_std))
        return intercept, slope

    def unstandardize_peak(
        self,
        a: jnp.ndarray,
        x0: jnp.ndarray,
        g: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return (
            self.y_std * a,
            self.x_mu + self.x_std * x0,
            self.x_std * g,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe_scale(scale: Union[float, jnp.ndarray]) -> jnp.ndarray:
        return jnp.maximum(jnp.asarray(scale), 1e-12)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_mu": _safe_to_serializable(self.x_mu),
            "x_std": _safe_to_serializable(self.x_std),
            "y_mu": _safe_to_serializable(self.y_mu),
            "y_std": _safe_to_serializable(self.y_std),
            "method": self.method,
        }


def _location_scale(values: jnp.ndarray, method: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if method == "identity":
        return jnp.array(0.0), jnp.array(1.0)
    if method == "robust":
        median = jnp.median(values)
        mad = 1.4826 * jnp.median(jnp.abs(values - median)) + 1e-12
        return median, mad
    if method == "moment":
        mu = jnp.mean(values)
        sigma = jnp.std(values) + 1e-12
        return mu, sigma
    raise ValueError(f"Unknown standardization method: {method}")


def _safe_to_serializable(value: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray, list]:
    """
    Convert standardization statistics to Python scalars or lists when possible
    without forcing evaluation of traced JAX values.
    """
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        try:
            return float(arr)
        except TypeError:
            return arr
    try:
        return arr.tolist()
    except TypeError:
        return arr

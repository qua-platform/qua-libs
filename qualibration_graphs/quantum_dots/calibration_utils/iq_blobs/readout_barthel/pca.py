"""
Principal Component Analysis (PCA) utilities for IQ readout data.

This module provides PCA-based dimensionality reduction from 2D IQ measurements
to 1D voltage coordinates. The projection automatically handles orientation
(sign flipping) to ensure that triplet states have higher values than singlet states
on the projected axis, which simplifies downstream threshold-based classification.
"""

import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PCAProjection:
    """
    Encapsulates a 1D PCA projection of 2D IQ data.

    Stores the parameters needed to project new IQ measurements onto the same
    1D axis used during training, maintaining consistent orientation.

    Attributes:
        mean: Center point of the IQ data, shape (2,). Subtracted before projection.
        pc1: First principal component (unit vector), shape (2,). Direction of maximum variance.
        sign: Orientation multiplier (+1 or -1). Ensures triplet > singlet on projected axis.
    """

    mean: jnp.ndarray  # (2,) - IQ center point
    pc1: jnp.ndarray  # (2,) - unit vector along first principal component
    sign: float  # +1 or -1, orientation used for projection

    def project(self, X2: jnp.ndarray) -> jnp.ndarray:
        """
        Project 2D IQ data onto the 1D principal component axis.

        Args:
            X2: IQ measurements to project, shape (N, 2)

        Returns:
            1D projected coordinates, shape (N,)
        """
        Xc = X2 - self.mean[None, :]
        return Xc @ (self.pc1 * self.sign)

    def backproject(self, y1: jnp.ndarray) -> jnp.ndarray:
        """
        Map 1D coordinates back to the IQ plane for visualization.

        Reconstructs points on the 1D PCA line in the original 2D IQ space.
        Useful for visualizing the projection axis and threshold on IQ plots.

        Args:
            y1: 1D projected coordinates, shape (N,)

        Returns:
            Reconstructed 2D IQ points on the PCA line, shape (N, 2)
        """
        return self.mean[None, :] + y1[:, None] * (self.pc1 * self.sign)[None, :]

def pca_project_1d(
    X: jnp.ndarray,
    labels: Optional[jnp.ndarray] = None,
    orient: str = "auto",
) -> Tuple[jnp.ndarray, PCAProjection]:
    """
    Perform PCA and project 2D IQ data onto the first principal component.

    Computes the principal component direction from the IQ data and projects
    all points onto this 1D axis. The orientation (sign) of the projection is
    determined either by provided labels or automatically by a heuristic.

    The projection process:
    1. Center the data by subtracting the mean
    2. Compute principal components via SVD
    3. Determine orientation sign based on labels or heuristic
    4. Project: y = (X - mean) · (pc1 * sign)

    Orientation logic:
    - If labels provided: orient so mean(label=1) > mean(label=0)
      This ensures triplet (label=1) has higher values than singlet (label=0)
    - If labels=None and orient='auto': choose sign so the longer tail
      points toward +∞ (compares 5th vs 95th percentile)
    - If orient='none': keep the natural SVD orientation (sign=1)

    Args:
        X: IQ measurements array, shape (N, 2) where N is number of shots
        labels: Optional binary labels {0, 1} for orientation. If provided,
               the projection is oriented so that label=1 (typically triplet)
               has higher mean than label=0 (typically singlet).
        orient: Orientation mode when labels is None:
               'auto' - use heuristic (longer tail toward +∞)
               'none' - keep natural SVD orientation

    Returns:
        y: Projected 1D coordinates, shape (N,)
        proj: PCAProjection object containing mean, pc1, and sign for
              applying the same projection to new data

    Example:
        >>> X = jnp.array([[0.0, 0.0], [1.0, 0.1], [0.1, 1.0]])
        >>> labels = jnp.array([0, 1, 1])  # First shot is singlet
        >>> y, proj = pca_project_1d(X, labels=labels)
        >>> # Now use proj to project new data
        >>> X_new = jnp.array([[0.5, 0.5]])
        >>> y_new = proj.project(X_new)
    """
    X = jnp.asarray(X, float)
    assert X.ndim == 2 and X.shape[1] == 2, "X must be shape (N, 2) IQ array"

    # Center the data
    mu = X.mean(axis=0)
    Xc = X - mu

    # Compute principal components via SVD: X = U Σ V^T
    _, _, Vt = jnp.linalg.svd(Xc, full_matrices=False)
    pc1 = Vt[0]  # First principal component (already unit length from SVD)

    # Determine orientation sign to ensure meaningful direction
    sign = 1.0
    y_tmp = Xc @ pc1  # Temporary projection with arbitrary sign

    if labels is not None:
        # Orient based on labels: ensure label=1 (triplet) > label=0 (singlet)
        m0 = y_tmp[labels == 0].mean() if jnp.any(labels == 0) else 0.0
        m1 = y_tmp[labels == 1].mean() if jnp.any(labels == 1) else 0.0
        sign = 1.0 if m1 >= m0 else -1.0
    elif orient == "auto":
        # Heuristic: orient so that the longer tail (higher variance side) points toward +∞
        # This typically puts triplet (minority with larger spread) on the positive side
        q5, q95 = jnp.percentile(y_tmp, [5, 95])
        sign = 1.0 if abs(q95) >= abs(q5) else -1.0

    # Apply orientation and package results
    y = y_tmp * sign
    proj = PCAProjection(mean=mu, pc1=pc1, sign=sign)
    return y, proj

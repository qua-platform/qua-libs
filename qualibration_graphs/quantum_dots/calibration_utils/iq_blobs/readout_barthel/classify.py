"""
Classification utilities for quantum readout using fitted PCA threshold.

This module provides functions to classify new IQ measurement data into singlet/triplet
states using a pre-determined threshold on the PCA-projected 1D axis. The classifier
uses the PCA projection learned from training data and applies a voltage threshold
to assign binary labels (0 = singlet, 1 = triplet).
"""

import jax.numpy as jnp
from typing import Tuple, Optional, Union
from readout_barthel.utils import Normalizer1D
from readout_barthel.pca import PCAProjection


def classify_iq_with_pca_threshold(
    X_new: jnp.ndarray,
    proj: PCAProjection,
    v_rf: float,
    *,
    normalizer: Optional[Normalizer1D] = None,
    return_margin: bool = False,
) -> Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, None]]:
    """
    Classify IQ measurements using a PCA projection and voltage threshold.

    Projects new IQ data onto the 1D PCA axis (using a pre-computed projection)
    and classifies each point based on whether it exceeds the threshold v_rf.
    Points above the threshold are labeled as 1 (triplet), below as 0 (singlet).

    The workflow:
    1. Center the IQ data using the PCA mean
    2. Project onto the principal component with sign alignment
    3. Optionally normalize using the same normalizer from training
    4. Apply threshold: label = 1 if y > v_rf, else 0
    5. Optionally compute classification margin (distance from threshold)

    Args:
        X_new: New IQ measurements to classify, shape (N, 2) where N is number of shots
        proj: PCA projection object containing mean, pc1, and sign from training data
        v_rf: Voltage threshold for classification (in the same coordinate system
              as the projected data, after normalization if normalizer is provided)
        normalizer: Optional normalizer to apply to projected coordinates. Must use
                   the same normalizer that was used during training/fitting.
        return_margin: If True, also return the signed distance from threshold for each point

    Returns:
        If return_margin is False:
            (labels, None) where labels is shape (N,) with values 0 or 1
        If return_margin is True:
            (labels, margins) where margins is shape (N,) with signed distance from threshold
            Positive margin means confidently classified as triplet (1)
            Negative margin means confidently classified as singlet (0)

    Example:
        >>> # After fitting and obtaining proj, normalizer, and optimal v_rf
        >>> X_test = jnp.array([[0.01, 0.001], [0.02, 0.002]])  # New measurements
        >>> labels, margins = classify_iq_with_pca_threshold(
        ...     X_test, proj, v_rf, normalizer=normalizer, return_margin=True
        ... )
        >>> print(labels)  # [0, 1] - first shot is singlet, second is triplet
    """
    # Convert to JAX arrays with proper dtype
    Xn = jnp.asarray(X_new, float)
    a = jnp.asarray(proj.pc1, float) * float(proj.sign)
    mu = jnp.asarray(proj.mean, float)

    # Project onto 1D axis: y = (X - mean) · (pc1 * sign)
    # y = (Xn - mu[None, :]) @ a
    offset = mu@a
    y = Xn @ a - offset
    # Apply normalization if provided (must match training normalization)
    if normalizer is not None:
        y = normalizer.transform(y)

    # Classify: 1 if above threshold (triplet), 0 if below (singlet)
    labels = (y > float(v_rf)).astype(jnp.int32)

    # Optionally return margin (signed distance from threshold)
    return (labels, (y - float(v_rf))) if return_margin else (labels, None)

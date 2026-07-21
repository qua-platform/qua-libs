"""Analysis utilities for fixed-point PSB readout (06d): labeled-stream GMM fitting and helpers."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import xarray as xr
from scipy.stats import norm as _scipy_norm

from calibration_utils.common_utils.experiment import get_qubits
from calibration_utils.iq_blobs.analysis import FitParameters


def resolve_qubits_and_dot_pairs(node):
    """Return list of (qubit, dot_pair) tuples resolved from qubit.preferred_readout_quantum_dot."""
    machine = node.machine
    qubits = get_qubits(node)
    pairs = []
    for qubit in qubits:
        preferred_dot_id = getattr(qubit, "preferred_readout_quantum_dot", None)
        if preferred_dot_id is None:
            raise ValueError(
                f"Qubit {qubit.id!r} has no preferred_readout_quantum_dot set; "
                "configure it to the partner dot used for PSB readout."
            )
        pair_name = machine.find_quantum_dot_pair(qubit.quantum_dot.id, preferred_dot_id)
        if pair_name is None:
            raise ValueError(
                f"No QuantumDotPair registered for dots {qubit.quantum_dot.id!r} and "
                f"{preferred_dot_id!r} (qubit {qubit.id!r})."
            )
        pairs.append((qubit, machine.quantum_dot_pairs[pair_name]))
    return qubits, pairs


def build_labeled_dataset(ds_raw: xr.Dataset, init_state_label: str) -> xr.Dataset:
    """Map I_no_pi/Q_no_pi/I_pi/Q_pi → Ig/Qg/Ie/Qe based on init_state_label."""
    if init_state_label == "decay":
        # No pi pulse loads T (decay/excited); pi pulse loads S (ground)
        return xr.Dataset({
            "Ig": ds_raw["I_pi"],
            "Qg": ds_raw["Q_pi"],
            "Ie": ds_raw["I_no_pi"],
            "Qe": ds_raw["Q_no_pi"],
        })
    else:  # "no_decay"
        # No pi pulse loads S (ground); pi pulse loads T (excited)
        return xr.Dataset({
            "Ig": ds_raw["I_no_pi"],
            "Qg": ds_raw["Q_no_pi"],
            "Ie": ds_raw["I_pi"],
            "Qe": ds_raw["Q_pi"],
        })


def gmm_analytic_fidelity(means, stds, weights, t_grid=None):
    """Compute optimal analytic fidelity F = 0.5*(FS + FT) for a 2-component GMM.

    Uses the same convention as Barthel1DMetricCurves (utils.py:334):
        FS = P(X <= t | component 0, the S/ground component)
        FT = P(X  > t | component 1, the T/excited component)
        F  = 0.5 * (FS + FT)
    Sweeps t over a dense grid and returns the maximum.
    """
    m0, m1 = float(means[0]), float(means[1])  # S component < T component
    s0, s1 = float(stds[0]),  float(stds[1])

    if t_grid is None:
        lo = min(m0 - 4 * s0, m1 - 4 * s1)
        hi = max(m0 + 4 * s0, m1 + 4 * s1)
        t_grid = np.linspace(lo, hi, 2000)

    FS = _scipy_norm.cdf(t_grid, loc=m0, scale=s0)          # P(X <= t | S)
    FT = 1.0 - _scipy_norm.cdf(t_grid, loc=m1, scale=s1)    # P(X > t  | T)
    fidelity_curve = 0.5 * (FS + FT)

    best_idx = int(np.argmax(fidelity_curve))
    return float(fidelity_curve[best_idx]), float(t_grid[best_idx])


def fit_gmm_labeled(
    ds_labeled: xr.Dataset,
    qubits: Sequence,
) -> Tuple[Dict[str, FitParameters], xr.Dataset]:
    """2-component sklearn GMM on PCA-projected labeled IQ data.

    Fits two Gaussians on ALL pooled data (both streams combined).  The two
    experimental streams (no-pi / pi) are used only *after* fitting to decide
    which Gaussian corresponds to S vs T via posterior probabilities — making
    the analysis robust to mixed-state loading in either stream.

    Returns ``(fit_results, ds_gmm_fit)`` where ``ds_gmm_fit`` contains the per-qubit
    PCA projections and GMM component parameters needed by ``plot_labeled_histogram_gmm``.

    Fidelity is the analytic model optimum 0.5*(FS+FT) — same convention as
    Barthel1DMetricCurves — NOT a confusion-matrix count.
    """
    import jax.numpy as jnp
    from sklearn.mixture import GaussianMixture
    from calibration_utils.iq_blobs.readout_barthel.pca import pca_project_1d

    fit_results = {}
    qnames = [q.name for q in qubits]

    y_g_list, y_e_list = [], []
    gmm_mean_S_list, gmm_mean_T_list = [], []
    gmm_std_S_list,  gmm_std_T_list  = [], []
    gmm_weight_S_list, gmm_weight_T_list = [], []
    ge_threshold_list, readout_fidelity_list = [], []

    for q in qubits:
        qname = q.name
        Ig = np.asarray(ds_labeled["Ig"].sel(qubit=qname).values, dtype=float).ravel()
        Qg = np.asarray(ds_labeled["Qg"].sel(qubit=qname).values, dtype=float).ravel()
        Ie = np.asarray(ds_labeled["Ie"].sel(qubit=qname).values, dtype=float).ravel()
        Qe = np.asarray(ds_labeled["Qe"].sel(qubit=qname).values, dtype=float).ravel()

        mask = np.isfinite(Ig) & np.isfinite(Qg) & np.isfinite(Ie) & np.isfinite(Qe)
        Ig, Qg, Ie, Qe = Ig[mask], Qg[mask], Ie[mask], Qe[mask]

        X_g = np.column_stack([Ig, Qg])
        X_e = np.column_stack([Ie, Qe])
        n_g = len(X_g)
        X_all = np.vstack([X_g, X_e])
        labels_arr = np.concatenate([np.zeros(n_g), np.ones(len(X_e))])

        y, proj = pca_project_1d(jnp.asarray(X_all), labels=jnp.asarray(labels_arr))
        y = np.asarray(y, dtype=float)

        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(y.reshape(-1, 1))

        raw_means   = gmm.means_.ravel()
        raw_stds    = np.sqrt(gmm.covariances_.ravel())
        raw_weights = gmm.weights_

        # Identify which GMM component is S vs T using the two streams'
        # posterior probabilities, not by assuming a fixed mean ordering.
        y_g_raw = y[:n_g]
        proba_g = gmm.predict_proba(y_g_raw.reshape(-1, 1))
        avg_posterior_g = proba_g.mean(axis=0)
        s_comp = int(np.argmax(avg_posterior_g))
        t_comp = 1 - s_comp

        means_st   = raw_means[[s_comp, t_comp]]
        stds_st    = raw_stds[[s_comp, t_comp]]
        weights_st = raw_weights[[s_comp, t_comp]]

        # Ensure S < T on the projected axis (required by the fidelity
        # convention FS = P(X <= t | S), FT = P(X > t | T)).
        axis_sign = float(proj.sign)
        if means_st[0] > means_st[1]:
            y = -y
            means_st = -means_st
            axis_sign = -axis_sign

        y_g = y[:n_g]
        y_e = y[n_g:]

        readout_fidelity, threshold = gmm_analytic_fidelity(means_st, stds_st, weights_st)
        readout_fidelity *= 100.0

        proj_dir = np.asarray(proj.pc1) * axis_sign
        iw_angle = float(np.arctan2(proj_dir[1], proj_dir[0]))
        mu = np.asarray(proj.mean)
        ca, sa = np.cos(iw_angle), np.sin(iw_angle)
        I_threshold = float(threshold + mu[0] * ca + mu[1] * sa)

        gg = float(np.mean(y_g <= threshold))
        ge = float(np.mean(y_g > threshold))
        eg = float(np.mean(y_e <= threshold))
        ee = float(np.mean(y_e > threshold))

        fit_results[qname] = FitParameters(
            iw_angle=iw_angle,
            ge_threshold=threshold,
            I_threshold=I_threshold,
            readout_fidelity=readout_fidelity,
            confusion_matrix=[[gg, ge], [eg, ee]],
            success=not (np.isnan(threshold) or np.isnan(readout_fidelity)),
        )

        y_g_list.append(y_g)
        y_e_list.append(y_e)
        gmm_mean_S_list.append(float(means_st[0]))
        gmm_mean_T_list.append(float(means_st[1]))
        gmm_std_S_list.append(float(stds_st[0]))
        gmm_std_T_list.append(float(stds_st[1]))
        gmm_weight_S_list.append(float(weights_st[0]))
        gmm_weight_T_list.append(float(weights_st[1]))
        ge_threshold_list.append(threshold)
        readout_fidelity_list.append(readout_fidelity / 100.0)

    # Pad y_g / y_e to a common length for storage in xr.Dataset
    n_s_max = max(len(a) for a in y_g_list)
    n_t_max = max(len(a) for a in y_e_list)
    y_g_pad = np.full((len(qnames), n_s_max), np.nan)
    y_e_pad = np.full((len(qnames), n_t_max), np.nan)
    for i, (yg, ye) in enumerate(zip(y_g_list, y_e_list)):
        y_g_pad[i, :len(yg)] = yg
        y_e_pad[i, :len(ye)] = ye

    ds_gmm_fit = xr.Dataset(
        coords={"qubit": qnames},
        data_vars={
            "y_g":           (["qubit", "n_s"],  y_g_pad),
            "y_e":           (["qubit", "n_t"],  y_e_pad),
            "gmm_mean_S":    ("qubit", np.array(gmm_mean_S_list)),
            "gmm_mean_T":    ("qubit", np.array(gmm_mean_T_list)),
            "gmm_std_S":     ("qubit", np.array(gmm_std_S_list)),
            "gmm_std_T":     ("qubit", np.array(gmm_std_T_list)),
            "gmm_weight_S":  ("qubit", np.array(gmm_weight_S_list)),
            "gmm_weight_T":  ("qubit", np.array(gmm_weight_T_list)),
            "ge_threshold":  ("qubit", np.array(ge_threshold_list)),
            "readout_fidelity": ("qubit", np.array(readout_fidelity_list)),
        },
    )

    return fit_results, ds_gmm_fit

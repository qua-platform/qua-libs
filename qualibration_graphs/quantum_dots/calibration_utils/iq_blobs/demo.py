from readout_barthel.simulate import SimulationParamsIQ, simulate_readout_iq
from readout_barthel.pca import pca_project_1d, PCAProjection
from readout_barthel.utils import Barthel1DMetricCurves
from readout_barthel.plotting import (
    plot_barthel_fit_1d,
    plot_iq_with_pca_and_threshold,
    plot_fidelity_and_visibility_barthel_1d,
)
from readout_barthel.calibrate import Barthel1DFromIQ
from readout_barthel.classify import classify_iq_with_pca_threshold

from pprint import pprint
import matplotlib
# matplotlib.use("QtAgg")  # or "Qt5Agg"/"TkAgg" if you prefer
import jax.numpy as jnp
import numpyro
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

if __name__ == "__main__":
    # 1) Simulate 2D IQ shots
    params = SimulationParamsIQ(
        n_samples=3000,
        p_triplet=0.5,
        mu_S=(0.0, 0.0),
        mu_T=(1.0e-2, 0.25e-2),   # place T somewhere else in IQ plane
        sigma_I=0.12e-2,
        sigma_Q=0.10e-2,
        rho=0.0,            # set nonzero if you want correlated I/Q noise in the simulator
        tau_M=1.0,
        T1=2.0,
    )
    X, labels = simulate_readout_iq(params, return_labels=True)
    X = jnp.array(X)

    params.p_triplet = 0.05
    X_S, _ = simulate_readout_iq(params, return_labels=True)
    X_S = jnp.array(X_S)

    # X_mixed: usual mixed shots; X_S: calibration set containing only S (or X_T with only T)
    y, proj, normalizer, mcmc, samples, _, calib_res = Barthel1DFromIQ.fit(
        X,
        fix_tau_M=1.0,
        calib=(X_S, "S"),        # ← tells the fitter which peak is S
        prior_strength=0.3,      # tighten/loosen μ priors from calibration
        sigma_scale_default=0.25 # noise prior scale
    )

    # Fidelity & visibility metrics computed via the shared static helper
    fidelity_res = Barthel1DMetricCurves.summarize_metric(
        samples,
        tauM_fixed=1.0,
        use_ppd=True,
        draws=64,
        metric="fidelity",
        return_aligned_curve=True,
    )
    visibility_res = Barthel1DMetricCurves.summarize_metric(
        samples,
        tauM_fixed=1.0,
        use_ppd=True,
        draws=64,
        metric="visibility",
        return_components=True,
    )
    v_rf_norm = fidelity_res["vrf_opt_aligned"]
    v_rf_phys = normalizer.inverse(v_rf_norm)  # physical 1D threshold

    plot_fidelity_and_visibility_barthel_1d(fidelity_res, visibility_res)
    plot_barthel_fit_1d(normalizer.transform(y), samples, tauM_fixed=1.0, v_rf=v_rf_norm)
    plot_iq_with_pca_and_threshold(X, proj, v_rf_phys, align="none", labels=labels)  # no extra alignment needed

    labels_new, margin = classify_iq_with_pca_threshold(X, proj, v_rf_norm, normalizer=normalizer, return_margin=True)
    plot_iq_with_pca_and_threshold(X, proj, v_rf_phys, align="none", labels=labels_new)  # no extra alignment needed

    # 1) Best visibility (and fidelity) at the optimal threshold
    print(f"V* = {visibility_res['visibility']:.3f}, F* = {visibility_res['fidelity']:.3f}, at Vrf = {visibility_res['vrf']:.4g}")

    # Rotate I and Q so all variance lies in I (align proj with I-axis)
    # Get the effective projection direction (pc1 * sign)
    proj_dir = proj.pc1 * proj.sign
    angle = jnp.arctan2(proj_dir[1], proj_dir[0])

    # Rotation matrix for rotating by -angle to align with I-axis
    rotation_matrix = jnp.array([
        [jnp.cos(angle), jnp.sin(angle)],
        [-jnp.sin(angle), jnp.cos(angle)]
    ])

    # Apply rotation to data
    X_rotated = X @ rotation_matrix.T

    # Create rotated PCAProjection object with rotated mean and pc1
    proj_rotated = PCAProjection(
        mean=proj.mean @ rotation_matrix.T,
        pc1=proj.pc1 @ rotation_matrix.T,
        sign=proj.sign
    )

    # Plot rotated version
    plot_iq_with_pca_and_threshold(X_rotated, proj_rotated, v_rf_phys, align="none", labels=labels)

    # Simple scatter plot with vertical threshold line
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot of rotated data
    ax.scatter(X_rotated[:, 0], X_rotated[:, 1], c=labels, alpha=0.5, s=10, cmap='viridis')
    ax.set_xlabel('I (rotated)')
    ax.set_ylabel('Q (rotated)')
    ax.set_title('Rotated IQ Data with Threshold')

    # Threshold as vertical line
    # In rotated space: (X_rotated - mean_rotated) @ (pc1_rotated * sign) = v_rf_phys
    # Since pc1_rotated * sign ≈ [1, 0], this gives: I_rotated = v_rf_phys + mean_rotated[0]
    threshold_I = float(v_rf_phys + proj_rotated.mean[0])
    ax.axvline(threshold_I, color='red', linestyle='--', linewidth=2, label=f'Threshold')

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ============================================================================
    # Analytic density calculation and plotting for debugging
    # ============================================================================
    from readout_barthel.analytic import _norm_pdf, triplet_pdf_analytic, decay_inflight_integral
    import numpy as np

    # Use the PCA projected data (this is what was actually fitted)
    y_norm = normalizer.transform(y)

    # Get posterior means for parameters (these are already in normalized space)
    mu_S = float(np.asarray(samples["mu_S"]).mean())
    mu_T = float(np.asarray(samples["mu_T"]).mean())
    sigma = float(np.asarray(samples["sigma"]).mean())
    pT = float(np.asarray(samples["pT"]).mean())
    T1 = float(np.asarray(samples["T1"]).mean())
    tauM = 1.0  # Fixed measurement time

    print(f"\nPosterior means (normalized space):")
    print(f"  mu_S = {mu_S:.6f}")
    print(f"  mu_T = {mu_T:.6f}")
    print(f"  sigma = {sigma:.6f}")
    print(f"  pT = {pT:.3f}")
    print(f"  T1 = {T1:.3f}")
    print(f"  tauM = {tauM:.3f}")

    # Set up grid for density plots in normalized space
    rng_norm = np.ptp(y_norm) or 1.0
    xs_norm = np.linspace(y_norm.min() - 0.1 * rng_norm, y_norm.max() + 0.1 * rng_norm, 800)
    xs_jax = jnp.array(xs_norm)

    # Compute density components analytically
    print(f"\nComputing analytic densities...")

    # Singlet component: (1 - pT) * N(y; mu_S, sigma)
    S_comp = (1 - pT) * _norm_pdf(xs_jax, mu_S, sigma)
    print(f"  S_comp: min={float(S_comp.min()):.6e}, max={float(S_comp.max()):.6e}")

    # Triplet component breakdown:
    p_no = jnp.exp(-tauM / T1) if T1 > 0 else 0.0
    print(f"  p_no (no decay) = {float(p_no):.3f}")

    # T (no decay): pT * p_no * N(y; mu_T, sigma)
    T_no_comp = pT * p_no * _norm_pdf(xs_jax, mu_T, sigma)
    print(f"  T_no_comp: min={float(T_no_comp.min()):.6e}, max={float(T_no_comp.max()):.6e}")

    # T (decay): pT * (1/T1) * integral
    T_dec_comp = pT * (1.0 / T1) * decay_inflight_integral(xs_jax, mu_S, mu_T, sigma, T1, tauM)
    print(f"  T_dec_comp: min={float(T_dec_comp.min()):.6e}, max={float(T_dec_comp.max()):.6e}")

    # Total density: singlet + triplet (no decay) + triplet (decay)
    total = S_comp + T_no_comp + T_dec_comp
    print(f"  Total: min={float(total.min()):.6e}, max={float(total.max()):.6e}")

    # Weights
    w_S = 1 - pT
    w_T_no = pT * float(p_no)
    w_T_dec = pT * (1 - float(p_no))
    print(f"\nWeights:")
    print(f"  w_S = {w_S:.3f}")
    print(f"  w_T_no = {w_T_no:.3f}")
    print(f"  w_T_dec = {w_T_dec:.3f}")
    print(f"  Sum = {w_S + w_T_no + w_T_dec:.3f}")

    # Plot histogram with analytic curves
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram of PCA projected data (normalized)
    ax.hist(y_norm, bins=120, alpha=0.5, density=True, label="Data (histogram)")

    # Plot analytic curves
    ax.plot(xs_norm, np.array(total), lw=2, label="Total (analytic)", color='black')
    ax.plot(xs_norm, np.array(S_comp), ls="--", label="S component", color='blue')
    ax.plot(xs_norm, np.array(T_no_comp), ls="--", label="T (no decay)", color='green')
    ax.plot(xs_norm, np.array(T_dec_comp), ls="--", label="T (decay)", color='orange')

    # Plot threshold line (in normalized space)
    ax.axvline(v_rf_norm, color='red', linestyle='--', lw=1.5, label='Optimal threshold')

    # Annotate weights
    ax.text(0.02, 0.98, f"w_S={w_S:.2f}  w_T(no)={w_T_no:.2f}  w_T(dec)={w_T_dec:.2f}",
            transform=ax.transAxes, va="top", fontsize=10)

    ax.set_xlabel("PCA projection (normalized)")
    ax.set_ylabel("Density")
    ax.set_title("Analytic Barthel Model Fit (Normalized Space)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

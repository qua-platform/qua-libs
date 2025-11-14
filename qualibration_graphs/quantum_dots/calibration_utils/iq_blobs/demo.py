from readout_barthel.simulate import SimulationParamsIQ, simulate_readout_iq
from readout_barthel.pca import pca_project_1d
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

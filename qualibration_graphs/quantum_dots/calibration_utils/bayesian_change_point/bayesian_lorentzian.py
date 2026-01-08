from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_likelihood

from .bayesian_base import BayesianMCMCBase
from .standardization import Standardization


class LorentzMixtureFitter(BayesianMCMCBase):
    """
    Bayesian model selector for linear-plus-Lorentz mixtures using MCMC.

    The public ``fit`` method mirrors :class:`BayesianCP` by returning
    ``(posterior, log_evidence)`` where ``posterior`` represents the
    probability of observing a Lorentzian peak at each provided x-value.
    Detailed diagnostics and summaries are available via
    :meth:`get_last_result`.
    """

    def __init__(
        self,
        max_components: int = 3,
        num_warmup: int = 800,
        num_samples: int = 1200,
        num_chains: int = 1,
        rng_seed: int = 0,
        *,
        standardize: bool = True,
    ):
        super().__init__(standardize=standardize)
        self.max_components = int(max_components)
        self.num_warmup = int(num_warmup)
        self.num_samples = int(num_samples)
        self.num_chains = int(num_chains)
        self.rng_seed = int(rng_seed)

        self.std: Optional[Standardization] = None
        self.best: Optional[Dict[str, Any]] = None
        self._candidates: Optional[List[Dict[str, Any]]] = None
        self._samples_best: Optional[Dict[str, jnp.ndarray]] = None
        self._posterior_axis: Optional[jnp.ndarray] = None

    # ------------------------------------------------------------------ #
    # Template implementation
    # ------------------------------------------------------------------ #
    def _fit_impl(self, x: jnp.ndarray, y: jnp.ndarray):
        x = jnp.asarray(x)
        y = jnp.asarray(y)

        if self.standardize:
            std = Standardization.from_data(x, y, method="moment")
            xp, yp = std.standardize(x, y)
        else:
            std = Standardization.identity()
            xp, yp = x, y

        self.std = std
        self._set_standardization(std)
        self._posterior_axis = jnp.asarray(x)

        n = xp.shape[0]
        rng = jax.random.PRNGKey(self.rng_seed)
        candidates: List[Dict[str, Any]] = []

        # Linear model (no Lorentz peaks)
        rng, key = jax.random.split(rng)
        mcmc_lin = self._run_mcmc(self._linear_model, xp, yp, key)
        smp_lin = mcmc_lin.get_samples(group_by_chain=False)
        ll_max_lin = self._max_loglik(self._linear_model, xp, yp, smp_lin)
        waic_lin = self._waic(self._linear_model, xp, yp, smp_lin)
        k_lin = 3
        bic_lin = self._bic(ll_max_lin, k_lin, n)
        candidates.append(
            dict(
                type="linear",
                K=0,
                bic=float(bic_lin),
                waic=float(waic_lin),
                loglik_max=float(ll_max_lin),
                k_params=k_lin,
                samples=smp_lin,
            )
        )

        # Lorentz mixtures with K peaks
        for K in range(1, self.max_components + 1):
            modelK = self._lorentz_model_factory(K)
            rng, key = jax.random.split(rng)
            mcmcK = self._run_mcmc(modelK, xp, yp, key)
            sK = mcmcK.get_samples(group_by_chain=False)
            ll_maxK = self._max_loglik(modelK, xp, yp, sK)
            waicK = self._waic(modelK, xp, yp, sK)
            kK = 3 + 3 * K
            bicK = self._bic(ll_maxK, kK, n)
            candidates.append(
                dict(
                    type="lorentzian",
                    K=K,
                    bic=float(bicK),
                    waic=float(waicK),
                    loglik_max=float(ll_maxK),
                    k_params=kK,
                    samples=sK,
                )
            )

        # Model selection (lowest BIC)
        waics = jnp.array([cand["waic"] for cand in candidates])
        best_idx = int(jnp.argmin(waics))
        best = candidates[best_idx]
        samples_best = best["samples"]

        posterior_peak_means: List[Dict[str, float]] = []
        if "a" in samples_best:
            std_local = self.std
            assert std_local is not None
            _, centers_u, widths_u = std_local.unstandardize_peak(
                samples_best["a"],
                samples_best["x0"],
                samples_best["gamma"],
            )
            center_mean = jnp.mean(centers_u, axis=0)
            width_mean = jnp.mean(widths_u, axis=0)
            order = jnp.argsort(center_mean)
            for idx in order.tolist():
                posterior_peak_means.append(
                    {
                        "center_mean": float(center_mean[idx]),
                        "width_mean": float(width_mean[idx]),
                    }
                )
        self.best = {
            "type": best["type"],
            "K": best["K"],
            "bic": float(best["bic"]),
            "loglik_max": float(best["loglik_max"]),
            "k_params": int(best["k_params"]),
            "posterior_peak_means": posterior_peak_means,
        }
        self._candidates = [{k: c[k] for k in ("type", "K", "bic", "loglik_max", "k_params")} for c in candidates]
        self._samples_best = samples_best

        posterior_mean, posterior_std = self.posterior_predictive_stats(x)
        peak_prob = posterior_mean

        diagnostics = {
            "bic": float(best["bic"]),
            "k_params": int(best["k_params"]),
            "n_observations": int(n),
        }
        extras = {
            "best_model": self.best,
            "candidates": self._candidates,
            "posterior_axis": self._posterior_axis,
            "peak_prob": peak_prob,
            "posterior_predictive_std": posterior_std,
            "posterior_summaries": self._summaries_unstd(samples_best),
            "point_estimates": self._points_unstd(samples_best),
        }

        return self._finalize_fit(
            posterior=peak_prob,
            log_evidence=best["loglik_max"],
            diagnostics=diagnostics,
            extras=extras,
        )

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def predict(self, x_new: jnp.ndarray) -> jnp.ndarray:
        """
        Deterministic mean prediction in original units using posterior medians.
        """
        assert (
            self.best is not None and self._samples_best is not None and self.std is not None
        ), "Call fit() before predict()."

        params = self._points_unstd(self._samples_best)
        y = params["intercept"] + params["slope"] * x_new
        for pk in params.get("peaks", []):
            a = pk["amplitude"]
            x0 = pk["center"]
            g = pk["width"]
            y = y + a / (1.0 + ((x_new - x0) / g) ** 2)
        return y

    def posterior_predictive_stats(self, x_new: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute posterior predictive mean and standard deviation for `x_new`.

        The mean reflects the posterior expectation of the latent function,
        while the standard deviation includes both parameter uncertainty and
        observation noise implied by the posterior samples.
        """
        assert (
            self.best is not None and self._samples_best is not None and self.std is not None
        ), "Call fit() before posterior_predictive_stats()."

        x_new = jnp.asarray(x_new)
        samples = self._samples_best
        std = self.std

        # Unstandardize linear parameters to original units
        intercepts, slopes = std.unstandardize_linear(samples["b0"], samples["b1"])
        mu = intercepts[:, None] + slopes[:, None] * x_new[None, :]

        # Add contributions from Lorentzian peaks when present
        if "a" in samples:
            a_u, x0_u, g_u = std.unstandardize_peak(
                samples["a"],
                samples["x0"],
                samples["gamma"],
            )
            kernels = 1.0 / (1.0 + ((x_new[None, None, :] - x0_u[:, :, None]) / g_u[:, :, None]) ** 2)
            mu = mu + jnp.sum(a_u[:, :, None] * kernels, axis=1)

        # Posterior predictive mean
        mean = jnp.mean(mu, axis=0)

        # Combine parameter uncertainty with observation noise variance
        sigma = samples["sigma"] * std.y_std
        centered = mu - mean
        predictive_var = jnp.mean(centered**2 + sigma[:, None] ** 2, axis=0)
        std_dev = jnp.sqrt(jnp.maximum(predictive_var, 0.0))

        return mean, std_dev

    def candidates(self) -> List[Dict[str, Any]]:
        assert self._candidates is not None, "Call fit() first."
        return self._candidates

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _run_mcmc(self, model, xp, yp, rng_key):
        kernel = NUTS(model)
        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=False,
        )
        mcmc.run(rng_key, xp, yp)
        return mcmc

    @staticmethod
    def _bic(loglik_max: float, k_params: int, n: int) -> float:
        return k_params * math.log(n) - 2.0 * float(loglik_max)

    def _waic(self, model, xp, yp, samples):
        # pointwise log-likelihood draws: (S, N)
        ll = log_likelihood(model, samples, xp, yp)["y"]

        # log pointwise predictive density (lppd)
        # use log-mean-exp for numerical stability
        def logmeanexp(v, axis=0):
            m = jnp.max(v, axis=axis, keepdims=True)
            return (m + jnp.log(jnp.mean(jnp.exp(v - m), axis=axis, keepdims=True))).squeeze(axis)

        lppd = jnp.sum(logmeanexp(ll, axis=0))
        # effective number of parameters
        p_waic = jnp.sum(jnp.var(ll, axis=0))
        waic = -2.0 * (lppd - p_waic)
        return float(waic)

    @staticmethod
    def _linear_model(xp, yp=None):
        n = xp.shape[0]
        b0 = numpyro.sample("b0", dist.Normal(0.0, 2.0))
        b1 = numpyro.sample("b1", dist.Normal(0.0, 2.0))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        mean = b0 + b1 * xp
        with numpyro.plate("data", n):
            numpyro.sample("y", dist.Normal(mean, sigma), obs=yp)

    @staticmethod
    def _lorentz_sum(xp, a, x0, g):
        return jnp.sum(a / (1.0 + ((xp[..., None] - x0) / g) ** 2), axis=-1)

    @classmethod
    def _lorentz_model_factory(cls, K: int):
        assert K >= 1

        def model(xp, yp=None):
            n = xp.shape[0]

            # ----- linear background -----
            b0 = numpyro.sample("b0", dist.Normal(0.0, 2.0))
            b1 = numpyro.sample("b1", dist.Normal(0.0, 2.0))

            # ===== shared (hierarchical) priors for amplitudes & widths =====
            # Draw these ONCE (shared across peaks) -> encourages similarity.
            amp_loc = numpyro.sample("amp_loc", dist.HalfNormal(1.0))  # typical amplitude scale
            amp_sd = numpyro.sample("amp_sd", dist.HalfNormal(0.3))  # small sd -> tight similarity

            width_loc = numpyro.sample("width_loc", dist.HalfNormal(0.5))  # typical width scale
            width_sd = numpyro.sample("width_sd", dist.HalfNormal(0.2))  # small sd -> tight similarity

            eps = 1e-8
            with numpyro.plate("peaks", K):
                # positive, partially pooled amplitudes and widths
                a = numpyro.sample("a", dist.LogNormal(jnp.log(jnp.maximum(amp_loc, eps)), amp_sd))
                g = numpyro.sample("gamma", dist.LogNormal(jnp.log(jnp.maximum(width_loc, eps)), width_sd))

            # ----- centers: allow outside the window but prefer interior -----
            xmid = 0.5 * (jnp.min(xp) + jnp.max(xp))
            xrng = jnp.max(xp) - jnp.min(xp)
            low = jnp.min(xp) - 3.0 * xrng
            high = jnp.max(xp) + 3.0 * xrng
            with numpyro.plate("peaks_x0", K):
                x0 = numpyro.sample("x0", dist.TruncatedNormal(xmid, xrng, low=low, high=high))

            # ===== soft repulsion between nearby centers (prevents splitting) =====
            if K > 1:
                # pairwise distances of centers
                d = jnp.abs(x0[:, None] - x0[None, :]) + 1e-12
                # characteristic distance ~ mean width
                gbar = jnp.mean(g) + 1e-12
                rho = 0.75  # "too close" ≈ 0.75 * gbar (tune 0.5–1.0)

                # learnable strength so data can override the prior when truly needed
                repel_strength = numpyro.sample("repel_strength", dist.HalfNormal(1.0))

                # amplitude-weighted repulsion: big peaks repel more than tiny ones
                w = a / (jnp.mean(a) + 1e-12)  # scale-free weights
                penalty_matrix = (w[:, None] * w[None, :]) * jnp.exp(-((d / (rho * gbar)) ** 2))

                # only i<j terms (upper triangle), avoid self terms
                iu = jnp.triu_indices(K, k=1)
                penalty = -repel_strength * jnp.sum(penalty_matrix[iu])

                # inject into log density
                numpyro.factor("repulsion", penalty)

            # ----- likelihood -----
            sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
            mean = b0 + b1 * xp + cls._lorentz_sum(xp, a, x0, g)
            with numpyro.plate("data", n):
                numpyro.sample("y", dist.Normal(mean, sigma), obs=yp)

        return model

    @staticmethod
    def _max_loglik(model, xp, yp, samples):
        """Compute maximum summed log-likelihood over posterior draws."""
        ll_dict = log_likelihood(model, samples, xp, yp)
        ll = ll_dict["y"]  # shape: (num_draws, num_data)
        ll_sum = jnp.sum(ll, axis=-1)
        return float(jnp.max(ll_sum))

    def _compute_peak_probability(self, x: jnp.ndarray, samples: Dict[str, jnp.ndarray]):
        x = jnp.asarray(x)
        if samples is None or "a" not in samples:
            return jnp.zeros_like(x)

        std = self.std
        assert std is not None

        a_u, x0_u, g_u = std.unstandardize_peak(samples["a"], samples["x0"], samples["gamma"])

        # Evaluate each posterior draw on the original x-grid
        kernels = 1.0 / (1.0 + ((x[None, None, :] - x0_u[:, :, None]) / g_u[:, :, None]) ** 2)
        weighted = a_u[:, :, None] * kernels
        density = jnp.sum(weighted, axis=1)  # (num_draws, n_x)
        density_sum = jnp.sum(density, axis=-1, keepdims=True)
        normalized = jnp.where(
            density_sum > 0,
            density / jnp.maximum(density_sum, 1e-12),
            0.0,
        )
        peak_prob = jnp.mean(normalized, axis=0)
        total = jnp.sum(peak_prob)
        peak_prob = jnp.where(total > 0, peak_prob / total, peak_prob)
        return peak_prob

    # ---------- Summaries / points in original units ----------
    def _summaries_unstd(self, samples: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
        std = self.std
        assert std is not None

        def qdict(v):
            qs = jnp.quantile(v, jnp.array([0.16, 0.5, 0.84]))
            return {"q16": float(qs[0]), "q50": float(qs[1]), "q84": float(qs[2])}

        b0p = samples["b0"]
        b1p = samples["b1"]
        sigp = samples["sigma"]
        idx = jnp.linspace(0, b0p.shape[0] - 1, min(300, b0p.shape[0])).round().astype(int)
        inter, slope = std.unstandardize_linear(b0p[idx], b1p[idx])
        out = {
            "intercept": qdict(inter),
            "slope": qdict(slope),
            "sigma": qdict(sigp * std.y_std),
            "peaks": [],
        }

        if "a" in samples:
            a = samples["a"]
            x0 = samples["x0"]
            g = samples["gamma"]
            x0_med_un = std.unstandardize_peak(
                jnp.zeros_like(x0[:, 0]) + 1,
                jnp.median(x0, axis=0),
                jnp.ones_like(x0[:, 0]),
            )[1]
            order = jnp.argsort(x0_med_un)
            for i in order.tolist():
                a_u, x0_u, g_u = std.unstandardize_peak(a[:, i], x0[:, i], g[:, i])
                out["peaks"].append(
                    {
                        "amplitude": qdict(a_u),
                        "center": qdict(x0_u),
                        "width": qdict(g_u),
                    }
                )
        return out

    def _points_unstd(self, samples: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
        std = self.std
        assert std is not None

        def med(v):
            return float(jnp.quantile(v, 0.5))

        b0m = med(samples["b0"])
        b1m = med(samples["b1"])
        sigm = med(samples["sigma"])
        intercept, slope = std.unstandardize_linear(jnp.array(b0m), jnp.array(b1m))
        point = {
            "intercept": float(intercept),
            "slope": float(slope),
            "sigma": float(sigm * std.y_std),
            "peaks": [],
        }
        if "a" in samples:
            a_m = jnp.quantile(samples["a"], 0.5, axis=0)
            x0_m = jnp.quantile(samples["x0"], 0.5, axis=0)
            g_m = jnp.quantile(samples["gamma"], 0.5, axis=0)
            a_u, x0_u, g_u = std.unstandardize_peak(a_m, x0_m, g_m)
            order = jnp.argsort(x0_u)
            for i in order.tolist():
                point["peaks"].append(
                    {
                        "amplitude": float(a_u[i]),
                        "center": float(x0_u[i]),
                        "width": float(g_u[i]),
                    }
                )
        return point


# ========= Convenience free functions =========
def fit_lorentzians_bic(
    x: jnp.ndarray,
    y: jnp.ndarray,
    max_components: int = 3,
    num_warmup: int = 800,
    num_samples: int = 1200,
    num_chains: int = 1,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """
    Backward-compatible helper returning rich diagnostics in dictionary form.
    """
    fitter = LorentzMixtureFitter(
        max_components=max_components,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        rng_seed=rng_seed,
    )
    posterior, log_evidence = fitter.fit(x, y)
    result = fitter.get_last_result(flatten=True)
    return {
        "posterior": posterior,
        "peak_prob": posterior,
        "posterior_axis": result["posterior_axis"],
        "log_evidence": log_evidence,
        "best": result["best_model"],
        "candidates": result["candidates"],
        "posterior_summaries": result["posterior_summaries"],
        "point_estimates": result["point_estimates"],
        "standardization": result["standardization"],
    }


def predict_in_original_units(x_new: jnp.ndarray, fit_result: Dict[str, Any]) -> jnp.ndarray:
    """
    Deterministic mean prediction using point estimates stored in `fit_result`.
    """
    params = fit_result["point_estimates"]
    y = params["intercept"] + params["slope"] * x_new
    for pk in params.get("peaks", []):
        y = y + pk["amplitude"] / (1 + ((x_new - pk["center"]) / pk["width"]) ** 2)
    return y


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    x = np.linspace(-5, 5, 400)
    y_true = 0.7 - 0.05 * x + 1.8 / (1 + ((x - (-1.0)) / 0.6) ** 2) + 0.9 / (1 + ((x - 1.8) / 0.4) ** 2)
    y = y_true + 0.08 * rng.normal(size=x.size)

    fitter = LorentzMixtureFitter(max_components=3, rng_seed=42)
    posterior, log_evidence = fitter.fit(jnp.asarray(x), jnp.asarray(y))
    details = fitter.get_last_result(flatten=True)

    print("Best model:", details["best_model"])
    print("Log evidence (approx):", log_evidence)
    print("Point estimates:")
    for k, v in details["point_estimates"].items():
        print(f"  {k}: {v}")

    x_pred = jnp.linspace(x.min(), x.max(), 800)
    y_pred = fitter.predict(x_pred)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=12, alpha=0.6, label="data")
    plt.plot(x_pred, y_pred, color="C3", lw=2.5, label="fit")

    for pk in details["point_estimates"].get("peaks", []):
        plt.plot(
            x_pred,
            pk["amplitude"] / (1 + ((x_pred - pk["center"]) / pk["width"]) ** 2),
            "--",
            lw=1.5,
            label=f"peak @ {pk['center']:.2f}",
        )

    plt.title(f"Best model: {details['best_model']['type']} (K={details['best_model']['K']})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()

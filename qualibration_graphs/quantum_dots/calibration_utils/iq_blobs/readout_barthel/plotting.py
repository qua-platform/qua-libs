import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, Tuple, Any

def plot_raw_data(x, bins=100, ax=None, show=True, save=None, label="data", figsize=(6, 4), **hist_kwargs):
    """
    Plot a histogram of the raw readout voltages.

    Args:
        x (array-like): 1D array of readout voltages.
        bins (int): Number of histogram bins.
        ax (matplotlib.Axes): Optional axis to draw on.
        show (bool): Whether to call plt.show().
        save (str|pathlib.Path|None): If set, path to save the figure (e.g. 'hist.png').
        label (str): Legend label for the data.
        figsize (tuple): Figure size for a new axes.
        **hist_kwargs: forwarded to ax.hist (e.g. alpha=0.6)
    Returns:
        ax (matplotlib.Axes)
    """

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # Default aesthetics but allow override through **hist_kwargs
    if "alpha" not in hist_kwargs:
        hist_kwargs["alpha"] = 0.6
    ax.hist(x, bins=bins, density=True, label=label, **hist_kwargs)

    ax.set_xlabel("Readout Voltage")
    ax.set_ylabel("Density")
    ax.set_title("Raw Readout Distribution")
    ax.legend()

    if save:
        ax.figure.savefig(save, bbox_inches="tight")
    if show and created_fig:
        plt.show()
    return ax


def plot_fit(x, samples, bins=100, ax=None, show=True, save=None,
             label_data="data", figsize=(6, 4), n_grid=400,
             component_kwargs=None, mixture_kwargs=None,
             posterior="ppd_mean", ppd_draws=64, agg="mean",
             hist_kwargs=None, **extra_hist_kwargs):
    """
    Plot the raw histogram plus fitted 2-component GMM curves and total density.

    Args:
        x (array-like): 1D readout data (used for histogram & x-range).
        samples (dict): Posterior samples from NumPyro MCMC with keys: 'pi', 'mu', 'tau'.
                        Each should be array-like with a samples dimension first.
        bins (int): Histogram bin count.
        ax (matplotlib.Axes): Optional axis to draw on.
        show (bool): Whether to call plt.show().
        save (str|pathlib.Path|None): If set, path to save the figure.
        label_data (str): Legend label for the histogram.
        figsize (tuple): Figure size for a new axes.
        n_grid (int): Number of x-points for curve plotting.
        component_kwargs (dict|None): kwargs for component lines (per-component).
        mixture_kwargs (dict|None): kwargs for total mixture line.
        posterior (str): 'ppd_mean' (recommended, label-invariant) or 'mean'/'median'.
        ppd_draws (int): Number of posterior draws for PPD mean mode.
        agg (str): Aggregation method when posterior != 'ppd_mean' ('mean' or 'median').
        hist_kwargs (dict|None): kwargs for histogram.
        **extra_hist_kwargs: additional kwargs forwarded to ax.hist (e.g. alpha=0.5)
    Returns:
        ax (matplotlib.Axes)
    """
    import numpy as np

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # Merge histogram kwargs
    if hist_kwargs is None:
        hist_kwargs = dict(alpha=0.5, density=True)
    hist_kwargs.update(extra_hist_kwargs)

    # Histogram
    x = np.asarray(x)
    ax.hist(x, bins=bins, label=label_data, **hist_kwargs)

    # Grid for plotting
    rng = np.ptp(x) or 1.0
    xs = np.linspace(x.min() - 0.1 * rng, x.max() + 0.1 * rng, n_grid)

    if component_kwargs is None:
        component_kwargs = dict(ls="--")
    if mixture_kwargs is None:
        mixture_kwargs = dict(lw=2)

    def _agg(key):
        if key not in samples: return None
        arr = np.asarray(samples[key])
        if posterior == "median" or agg == "median":
            return np.median(arr, axis=0)
        return arr.mean(axis=0)

    def _norm_pdf_local(x, mu, sigma):
        return (1.0 / (np.sqrt(2*np.pi)*sigma)) * np.exp(-(x - mu)**2 / (2*sigma**2))

    if posterior == "ppd_mean":
        # Average densities over posterior draws (label-invariant)
        S = len(np.asarray(samples["pi"]))
        idx = _choose_draw_indices(S, ppd_draws)

        total_pdf = np.zeros_like(xs)
        comp_pdfs = []

        # Determine number of components from first sample
        n_comp = len(np.asarray(samples["pi"])[0])
        for k in range(n_comp):
            comp_pdfs.append(np.zeros_like(xs))

        for i in idx:
            pi = np.asarray(samples["pi"])[i]
            mu = np.asarray(samples["mu"])[i]
            sigma = 1.0 / np.sqrt(np.asarray(samples["tau"])[i])

            dens_i = np.zeros_like(xs)
            for k in range(n_comp):
                comp_k = pi[k] * _norm_pdf_local(xs, mu[k], sigma[k])
                comp_pdfs[k] += comp_k
                dens_i += comp_k
            total_pdf += dens_i

        total_pdf /= float(len(idx))
        for k in range(n_comp):
            comp_pdfs[k] /= float(len(idx))

        # Plot
        ax.plot(xs, total_pdf, label="Total (PPD mean)", **mixture_kwargs)
        for k in range(n_comp):
            ax.plot(xs, comp_pdfs[k], label=f"Component {k+1}", **component_kwargs)

        # Annotate weights using posterior means
        pi_mean = _agg("pi")
        ax.text(0.02, 0.98, f"weights: " + "  ".join([f"w_{k+1}={pi_mean[k]:.2f}" for k in range(n_comp)]),
                transform=ax.transAxes, va="top", fontsize=9)

    else:
        # Point-estimate plug-in
        pi = _agg("pi")
        mu = _agg("mu")
        sigma = 1.0 / np.sqrt(_agg("tau"))

        total_pdf = np.zeros_like(xs)
        n_comp = len(pi)

        # Components
        for k in range(n_comp):
            comp_pdf = pi[k] * _norm_pdf_local(xs, mu[k], sigma[k])
            ax.plot(xs, comp_pdf, label=f"Component {k+1}", **component_kwargs)
            total_pdf += comp_pdf

        # Mixture total
        ax.plot(xs, total_pdf, label=f"Total ({posterior})", **mixture_kwargs)

    ax.set_xlabel("Readout Voltage")
    ax.set_ylabel("Density")
    ax.set_title("Raw Data + Fitted GMM")
    ax.legend()

    if save:
        ax.figure.savefig(save, bbox_inches="tight")
    if show and created_fig:
        plt.show()
    return ax

# --- leave your existing 1D functions as-is; append the following ---

def _get_plt():
    import matplotlib
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        return plt

# --- keep your existing imports and _get_plt() helper ---

def plot_raw_data_iq(
    X,
    labels=None,                 # <--- NEW
    ax=None,
    show=True,
    save=None,
    method="scatter",            # 'scatter' | 'hist2d' | 'hexbin'
    bins=100,
    gridsize=60,
    alpha=0.5,
    s=5,
    figsize=(6, 6),
    palette=None,                # <--- NEW: dict like {0:'C0', 1:'C1'}
    label_names=None,            # <--- NEW: dict like {0:'S', 1:'T'}
    legend=True,                 # <--- NEW
    legend_loc="best",           # <--- NEW
    **kwargs,
):
    """
    Plot raw IQ data (N,2). If `labels` is provided, points are colored by class (e.g., S/T).

    Args:
        X: (N,2) array of IQ data.
        labels: Optional (N,) array-like of {0,1} or booleans. 0→S, 1→T by default.
        method: 'scatter' | 'hist2d' | 'hexbin'. If labels is provided, will use 'scatter'.
        palette: Optional dict mapping class value -> color. If None, uses a stable cycle (C0, C1, ...).
        label_names: Optional dict mapping class value -> legend name. Defaults: {0:'S', 1:'T'}.
        legend: Show legend when labels is not None.
        legend_loc: Matplotlib legend location string.
    """
    plt = _get_plt()
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created = True

    X = np.asarray(X)
    I = X[:, 0]
    Q = X[:, 1]

    # If labels are given, draw class-colored scatter for clarity
    if labels is not None:
        labels = np.asarray(labels).astype(int).ravel()
        if labels.shape[0] != X.shape[0]:
            raise ValueError("labels must have the same length as X.")

        uniq = np.unique(labels)
        # Default label names
        if label_names is None:
            label_names = {0: "S", 1: "T"}

        # Default palette: stable mapping C0, C1, ...
        if palette is None:
            palette = {cls: f"C{i % 10}" for i, cls in enumerate(sorted(uniq))}

        # Always use scatter when coloring by class
        for i, cls in enumerate(sorted(uniq)):
            m = labels == cls
            ax.scatter(I[m], Q[m], s=s, alpha=alpha,
                       label=label_names.get(cls, f"class {cls}"),
                       color=palette.get(cls, f"C{i % 10}"),
                       **kwargs)

        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_title("Raw IQ Readout Distribution (colored by state)")
        ax.axis("equal")
        if legend:
            ax.legend(loc=legend_loc)

        if save:
            ax.figure.savefig(save, bbox_inches="tight")
        if show and created:
            plt.show()
        return ax

    # No labels: keep your original behavior
    if method == "scatter":
        ax.scatter(I, Q, s=s, alpha=alpha, **kwargs)
    elif method == "hist2d":
        h = ax.hist2d(I, Q, bins=bins, density=True, **kwargs)
        plt.colorbar(h[3], ax=ax)
    elif method == "hexbin":
        hb = ax.hexbin(I, Q, gridsize=gridsize, bins="log", **kwargs)
        plt.colorbar(hb, ax=ax)
    else:
        raise ValueError("method must be one of {'scatter','hist2d','hexbin'}")

    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title("Raw IQ Readout Distribution")
    ax.axis("equal")

    if save:
        ax.figure.savefig(save, bbox_inches="tight")
    if show and created:
        plt.show()
    return ax


def plot_fit_iq(
    X,
    samples,
    labels=None,                     # <--- NEW: pass through for colored raw layer
    ax=None,
    show=True,
    save=None,
    method="scatter",
    bins=100,
    gridsize=60,
    alpha=0.4,
    s=5,
    levels=8,
    grid_pts=200,
    figsize=(6, 6),
    draw_component_ellipses=True,
    ellipse_nsigma=1.0,
    **kwargs,
):
    """
    Plot raw IQ data plus fitted GMM (diagonal) as density contours and 1σ ellipses.
    If `labels` is provided, the raw layer is colored by class.
    """
    from matplotlib.patches import Ellipse
    plt = _get_plt()
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created = True

    # Base layer: raw data (now can be colored by labels)
    plot_raw_data_iq(
        X,
        labels=labels,          # <--- pass through
        ax=ax,
        show=False,
        method="scatter" if labels is not None else method,
        bins=bins,
        gridsize=gridsize,
        alpha=alpha,
        s=s,
        **kwargs,
    )

    X = np.asarray(X)
    Imin, Imax = np.min(X[:, 0]), np.max(X[:, 0])
    Qmin, Qmax = np.min(X[:, 1]), np.max(X[:, 1])
    dI = Imax - Imin
    dQ = Qmax - Qmin
    padI = 0.1 * dI if dI > 0 else 1.0
    padQ = 0.1 * dQ if dQ > 0 else 1.0
    I_lin = np.linspace(Imin - padI, Imax + padI, grid_pts)
    Q_lin = np.linspace(Qmin - padQ, Qmax + padQ, grid_pts)
    Ig, Qg = np.meshgrid(I_lin, Q_lin)

    # Posterior means for overlay
    pi = np.array(samples["pi"]).mean(axis=0)          # (K,)
    mu = np.array(samples["mu"]).mean(axis=0)          # (K,2)
    sigma = np.array(samples["sigma"]).mean(axis=0)    # (K,2)

    # Evaluate diagonal-covariance mixture density on grid
    two_pi = 2.0 * np.pi
    dens = np.zeros_like(Ig)
    for k in range(len(pi)):
        sI, sQ = sigma[k, 0], sigma[k, 1]
        mI, mQ = mu[k, 0], mu[k, 1]
        Z = two_pi * sI * sQ
        expo = ((Ig - mI) / sI) ** 2 + ((Qg - mQ) / sQ) ** 2
        dens += pi[k] * np.exp(-0.5 * expo) / Z

    cs = ax.contour(Ig, Qg, dens, levels=levels)
    ax.clabel(cs, inline=True, fontsize=8)

    if draw_component_ellipses:
        for k in range(len(pi)):
            width = 2.0 * ellipse_nsigma * sigma[k, 0]
            height = 2.0 * ellipse_nsigma * sigma[k, 1]
            e = Ellipse(xy=(mu[k, 0], mu[k, 1]), width=width, height=height,
                        angle=0.0, fill=False, lw=2)
            ax.add_patch(e)
            ax.plot(mu[k, 0], mu[k, 1], marker="x", ms=8)

    ax.set_title("IQ Data + Fitted 2D GMM (diag)")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.axis("equal")

    if save:
        ax.figure.savefig(save, bbox_inches="tight")
    if show and created:
        plt.show()
    return ax

# --- 1D Barthel plotting helpers ---

def _get_plt():
    import matplotlib
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        return plt

def _choose_draw_indices(n, draws):
    if draws >= n:
        return np.arange(n)
    step = n / float(draws)
    return (step * np.arange(draws)).astype(int)

def _norm_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)

def evaluate_barthel_density_1d_grid(xs, *, mu_S, mu_T, sigma, pT, T1, tauM, dec_nodes=48):
    """
    Evaluate the 1D Barthel mixture on xs. Returns (total, comp_dict)
    comp_dict has keys: 'S', 'T_no', 'T_dec'
    """
    p_no = np.exp(-tauM / T1) if T1 > 0 else 0.0
    w_S = max(1.0 - pT, 0.0)
    w_Tno = max(pT * p_no, 0.0)
    w_Tdec_tot = max(pT * (1.0 - p_no), 0.0)

    dens = np.zeros_like(xs, dtype=float)
    comps = {}

    # S and T-no
    S = _norm_pdf(xs, mu_S, sigma)
    Tno = _norm_pdf(xs, mu_T, sigma)
    dens += w_S * S + w_Tno * Tno
    comps["S"] = w_S * S
    comps["T_no"] = w_Tno * Tno

    # Decay-in-flight via inverse-CDF quadrature over t in [0, tauM)
    if w_Tdec_tot > 0 and dec_nodes > 0:
        u = (np.arange(dec_nodes, dtype=float) + 0.5) / float(dec_nodes)
        if T1 > 0:
            cdf_tau = 1.0 - np.exp(-tauM / T1)
            t = -T1 * np.log(1.0 - u * cdf_tau)
        else:
            t = np.zeros(dec_nodes, dtype=float)
        alpha = t / tauM
        w_each = w_Tdec_tot / float(dec_nodes)
        dec = np.zeros_like(xs)
        for a in alpha:
            mu = mu_S + a * (mu_T - mu_S)
            dec += _norm_pdf(xs, mu, sigma) * w_each
        dens += dec
        comps["T_dec"] = dec
    else:
        comps["T_dec"] = np.zeros_like(xs)

    return dens, comps

def plot_barthel_fit_1d(
    y,
    samples,
    *,
    v_rf: float | None = None,
    tauM_fixed: float | None = None,
    bins: int = 120,
    grid_pts: int = 800,
    dec_nodes: int = 64,
    posterior: str = "ppd_mean",   # 'ppd_mean' (recommended) or 'mean'/'median'
    ppd_draws: int = 64,
    show: bool = True,
    save: str | None = None,
    ax=None,
    agg: str = "mean",             # used when posterior != 'ppd_mean'
    hist_kwargs: dict | None = None,
    total_kwargs: dict | None = None,
    comp_kwargs: dict | None = None,
):
    """
    Plot histogram of y with the fitted 1D Barthel mixture curves.

    - posterior='ppd_mean' averages densities over posterior draws (label-invariant).
    - For point estimates, set posterior='mean' or 'median' (can mislead under label switching).
    """
    plt = _get_plt()
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        created = True

    if hist_kwargs is None: hist_kwargs = dict(alpha=0.5, density=True)
    if total_kwargs is None: total_kwargs = dict(lw=2)
    if comp_kwargs is None: comp_kwargs = dict(ls="--")

    y = np.asarray(y)
    ax.hist(y, bins=bins, **hist_kwargs)

    rng = np.ptp(y) or 1.0
    xs = np.linspace(y.min() - 0.1 * rng, y.max() + 0.1 * rng, grid_pts)

    def _agg(key):
        if key not in samples: return None
        arr = np.asarray(samples[key])
        if posterior == "median" or agg == "median":
            return np.median(arr, axis=0)
        return arr.mean(axis=0)

    if posterior == "ppd_mean":
        S = len(np.asarray(samples["T1"]))
        idx = _choose_draw_indices(S, ppd_draws)

        total = np.zeros_like(xs)
        comps_sum = dict(S=np.zeros_like(xs), T_no=np.zeros_like(xs), T_dec=np.zeros_like(xs))

        for i in idx:
            mu_S = float(np.asarray(samples["mu_S"])[i])
            mu_T = float(np.asarray(samples["mu_T"])[i])
            sigma = float(np.asarray(samples["sigma"])[i])
            pT    = float(np.asarray(samples["pT"])[i])
            T1    = float(np.asarray(samples["T1"])[i])
            if "tauM" in samples:
                tauM = float(np.asarray(samples["tauM"])[i])
            else:
                if tauM_fixed is None:
                    raise ValueError("tauM not in samples; pass tauM_fixed to plot.")
                tauM = float(tauM_fixed)

            dens_i, comps_i = evaluate_barthel_density_1d_grid(
                xs, mu_S=mu_S, mu_T=mu_T, sigma=sigma, pT=pT, T1=T1, tauM=tauM, dec_nodes=dec_nodes
            )
            total += dens_i
            for k in comps_sum:
                comps_sum[k] += comps_i[k]

        total /= float(len(idx))
        for k in comps_sum:
            comps_sum[k] /= float(len(idx))

        # plot
        ax.plot(xs, total, label="Total (PPD mean)", **total_kwargs)
        ax.plot(xs, comps_sum["S"], label="S component", **comp_kwargs)
        ax.plot(xs, comps_sum["T_no"], label="T (no decay)", **comp_kwargs)
        ax.plot(xs, comps_sum["T_dec"], label="T (decay)", **comp_kwargs)

        # annotate weights using posterior means
        pT_m = float(_agg("pT")); T1_m = float(_agg("T1"))
        tauM_m = float(_agg("tauM")) if "tauM" in samples else float(tauM_fixed)
        p_no = np.exp(-tauM_m / T1_m) if T1_m > 0 else 0.0
        ax.text(0.02, 0.98, f"w_S={1-pT_m:.2f}  w_T(no)={pT_m*p_no:.2f}  w_T(dec)={pT_m*(1-p_no):.2f}\n"
                            f"T1={T1_m:.3g}  tauM={tauM_m:g}",
                transform=ax.transAxes, va="top")

    else:
        # point-estimate plug-in
        mu_S = float(_agg("mu_S")); mu_T = float(_agg("mu_T"))
        sigma = float(_agg("sigma")); pT = float(_agg("pT")); T1 = float(_agg("T1"))
        tauM = float(_agg("tauM")) if "tauM" in samples else float(tauM_fixed)

        total, comps = evaluate_barthel_density_1d_grid(
            xs, mu_S=mu_S, mu_T=mu_T, sigma=sigma, pT=pT, T1=T1, tauM=tauM, dec_nodes=dec_nodes
        )
        ax.plot(xs, total, label=f"Total ({posterior})", **total_kwargs)
        ax.plot(xs, comps["S"], label="S component", **comp_kwargs)
        ax.plot(xs, comps["T_no"], label="T (no decay)", **comp_kwargs)
        ax.plot(xs, comps["T_dec"], label="T (decay)", **comp_kwargs)

    if v_rf:
        ax.axvline(v_rf, color='k', ls="--", lw=1, label="optimal threshold")

    ax.set_xlabel("Projected readout (PCA → 1D)")
    ax.set_ylabel("Density")
    ax.set_title("1D Barthel Fit on PCA-projected Data")
    ax.legend()
    if save:
        ax.figure.savefig(save, bbox_inches="tight")
    if show and created:
        plt.show()
    return ax

def plot_fidelity_and_visibility_barthel_1d(
    fidelity_result: Dict[str, Any],
    visibility_result: Dict[str, Any],
    *,
    show: bool = True,
    save: Optional[str] = None,
    ax=None,
    figsize: tuple = (6, 4),
    fidelity_color: str = "C0",
    visibility_color: str = "C1",
):
    """
    Plot fidelity and visibility curves on the same axes, highlighting their optima.
    """
    plt = _get_plt()
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created = True

    vrf_f = np.asarray(fidelity_result["vrf"])
    F_curve = np.asarray(fidelity_result["fidelity"])
    vrf_f_opt = float(fidelity_result["vrf_opt"])
    F_opt = float(fidelity_result["fidelity_opt"])

    vrf_v = np.asarray(visibility_result["vrf_grid"])
    V_curve = np.asarray(visibility_result["visibility_curve"])
    vrf_v_opt = float(visibility_result["vrf"])
    V_opt = float(visibility_result["visibility"])

    ax.plot(vrf_f, F_curve, color=fidelity_color, lw=2, label="Fidelity")
    ax.plot(vrf_v, V_curve, color=visibility_color, lw=2, label="Visibility")

    ax.axvline(vrf_f_opt, color=fidelity_color, ls="--", lw=1)
    ax.plot(vrf_f_opt, F_opt, "o", color=fidelity_color, ms=6, label=f"F*: {F_opt:.3f} @ {vrf_f_opt:.3f}")

    ax.axvline(vrf_v_opt, color=visibility_color, ls="--", lw=1)
    ax.plot(vrf_v_opt, V_opt, "s", color=visibility_color, ms=6, label=f"V*: {V_opt:.3f} @ {vrf_v_opt:.3f}")

    ax.set_xlabel("Threshold voltage  $V_{rf}$")
    ax.set_ylabel("Metric value")
    ax.set_title("Fidelity and Visibility vs Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save:
        ax.figure.savefig(save, bbox_inches="tight")
    if show and created:
        plt.show()
    return ax

def plot_iq_with_pca_and_threshold(
    X,
    proj,                # PCAProjection(mean, pc1, sign)
    v_rf,                # optimal threshold from fidelity (1D coordinate)
    *,
    labels=None,         # optional {0,1} labels (T=1) -> best for alignment
    samples=None,        # optional posterior dict; uses mean(mu_T - mu_S) if labels absent
    ax=None,
    show=True,
    save=None,
    figsize=(6, 6),
    raw_method="scatter",
    raw_alpha=0.5,
    raw_s=6,
    pca_kwargs=None,
    thr_kwargs=None,
    annotate=True,
    legend=True,
    align="auto",        # 'auto' | 'none' — auto flips v_rf if needed
):
    """
    Plot raw IQ with the PCA axis and the optimal threshold line (orthogonal to PCA axis).
    Auto-alignment ensures that the "T side" is the > v_rf side along the PCA axis.

    Decision rule in 1D: y = (x - mean)·(pc1 * sign); predict T if y > v_rf.
    The 2D threshold line is { x : (x - mean)·(pc1 * sign) = v_rf }.
    """
    plt = _get_plt()
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created = True

    # 0) Optional auto-alignment: ensure T projects to larger y than S
    # Compute the orientation indicator s_target = sign( E[y|T] - E[y|S] )
    # If s_target < 0, flip v_rf -> -v_rf so that T is on the y > v_rf side.
    a = np.asarray(proj.pc1, float) * float(proj.sign)  # oriented PCA axis (unit)
    mean = np.asarray(proj.mean, float)

    if align == "auto":
        s_target = None
        if labels is not None and np.any(labels == 1) and np.any(labels == 0):
            y_proj = (np.asarray(X, float) - mean[None, :]) @ a
            mT = y_proj[labels == 1].mean()
            mS = y_proj[labels == 0].mean()
            s_target = np.sign(mT - mS)
        elif samples is not None and "mu_S" in samples and "mu_T" in samples:
            muS_m = float(np.asarray(samples["mu_S"]).mean(axis=0))
            muT_m = float(np.asarray(samples["mu_T"]).mean(axis=0))
            s_target = np.sign(muT_m - muS_m)

        if s_target is not None and s_target < 0:
            v_rf = -float(v_rf)  # flip threshold to match PCA orientation

    # 1) Raw layer
    plot_raw_data_iq(
        X,
        labels=labels,
        ax=ax,
        show=False,
        method="scatter" if labels is not None else raw_method,
        alpha=raw_alpha,
        s=raw_s,
    )

    # 2) Build PCA axis and threshold line
    n = np.array([-a[1], a[0]], float)
    n /= (np.linalg.norm(n) + 1e-12)

    # Use current data limits to span the lines
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    corners = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]])

    # PCA axis segment
    t_vals = (corners - mean) @ a
    t_min, t_max = float(t_vals.min()), float(t_vals.max())
    p1 = mean + t_min * a
    p2 = mean + t_max * a

    # Threshold line segment (orthogonal to axis) through point p_thr on the axis
    p_thr = mean + float(v_rf) * a
    s_vals = (corners - p_thr) @ n
    s_min, s_max = float(s_vals.min()), float(s_vals.max())
    q1 = p_thr + s_min * n
    q2 = p_thr + s_max * n

    # 3) Draw
    if pca_kwargs is None: pca_kwargs = dict(color="C2", lw=2)
    if thr_kwargs is None: thr_kwargs = dict(color="C3", ls="--", lw=2)

    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], label="PCA axis", **pca_kwargs)
    ax.plot([q1[0], q2[0]], [q1[1], q2[1]], label=r"$V_{\rm rf}$ threshold", **thr_kwargs)

    if annotate:
        ax.plot(mean[0], mean[1], marker="x", ms=8, color=pca_kwargs.get("color", "C2"))
        ax.plot(p_thr[0], p_thr[1], marker="o", ms=6, color=thr_kwargs.get("color", "C3"))
        ax.text(p_thr[0], p_thr[1], f"  Vrf={float(v_rf):.3f}", va="center", fontsize=9)

    ax.set_xlabel("I"); ax.set_ylabel("Q")
    ax.set_title("Raw IQ with PCA axis and optimal threshold")
    ax.axis("equal")
    if legend: ax.legend()
    if save: ax.figure.savefig(save, bbox_inches="tight")
    if show and created: plt.show()
    return ax

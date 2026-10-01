import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit, least_squares


###################################################
# Sinusoidal fit related
###################################################
def _zero_crossings(t, y, min_separation=None, smooth_window=5):
    """
    Detect zero crossings of y(t) with optional rolling-mean smoothing
    and minimal time separation filtering.

    Parameters
    ----------
    t, y : array_like
        Time and signal arrays.
    min_separation : float, optional
        Minimum spacing between consecutive crossings.
        Defaults to 1% of total time span.
    smooth_window : int, optional
        Number of points for rolling mean smoothing (odd recommended).
        If None, no smoothing applied.
    """
    # y_orig = y.copy()
    y = np.asarray(y, float)
    t = np.asarray(t, float)

    # Apply rolling mean smoothing
    if smooth_window is not None and smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        y = np.convolve(y, kernel, mode="same")

    s = np.signbit(y)
    idx = np.where(s[:-1] != s[1:])[0]
    if idx.size == 0:
        return np.array([])

    # Linear interpolation for zero-crossing positions
    t0 = t[idx] + (t[idx + 1] - t[idx]) * (np.abs(y[idx]) / (np.abs(y[idx]) + np.abs(y[idx + 1])))

    # Default minimal spacing = 1% of time range
    if min_separation is None:
        min_separation = 0.01 * (t[-1] - t[0])

    keep = [0]
    for i in range(1, len(t0)):
        if t0[i] - t0[keep[-1]] > min_separation:
            keep.append(i)

    # plt.figure()
    # plt.plot(t, y_orig, label='Signal')
    # plt.plot(t, y, "--", label='Smoothed Signal' if smooth_window else 'Signal')
    # plt.plot(t0, np.zeros_like(t0), 'ro', label='Zero Crossings')
    # plt.legend()
    # plt.show()

    return t0[keep]


def _omega_tau_from_segments(t, y):
    """
    Period from median zero-crossing separation (between consecutive crossings).
    In each segment (between crossings), take abs(max) and its time; fit ln(amplitude) vs time.
    """
    zt = _zero_crossings(t, y)
    if len(zt) < 2:
        # fallback guesses
        print("Less than 1/2 period, fallback guesses used")
        abs_y = np.abs(y)
        T = 4 * abs(t[np.argmax(abs_y)] - t[np.argmin(abs_y)])
        omega = 2 * np.pi / T
        phase_sign = np.sign(y[abs_y.argmax()])
        return omega, None, phase_sign

    # median separation between consecutive zero crossings ~ T/2
    d = np.diff(zt)
    med_halfT = np.median(d)
    omega = np.pi / med_halfT  # since half-period = pi/omega

    # add zero crossing to zero if the first segment is close to median separation
    partial_segment_duration = zt[0] - t[0]
    if partial_segment_duration > 0.7 * med_halfT:
        zt = np.insert(zt, 0, t[0])
    elif partial_segment_duration > 0.3 * med_halfT:
        zt = np.insert(zt, 0, t[0] - med_halfT)

    # segment-wise envelope samples
    amps, times = [], []
    # use segments bounded by consecutive zero crossings
    for zt_start, zt_end in zip(zt[:-1], zt[1:]):
        msk = (t >= zt_start) & (t <= zt_end)
        if np.any(msk):
            t_seg = t[msk]
            y_seg = y[msk]
            abs_amp = abs(y_seg)
            max_abs_amp = np.argmax(abs_amp)
            amps.append(y_seg[max_abs_amp])
            times.append(t_seg[max_abs_amp])

    amps = np.asarray(amps)
    times = np.asarray(times)
    # plt.plot(times, amps, "go")
    # plt.plot(t, y, "k-", alpha=0.5)
    # for t in zt:
    #     plt.axvline(t, color="gray", ls=":", lw=0.8)
    # plt.show()
    # determine the sign of the phase from the first segment
    phase_sign = np.sign(amps[0])

    amps = abs(amps)

    if len(amps) < 3:
        tau = (t[-1] - t[0]) / 5.0
        return omega, tau, phase_sign

    # exponential envelope: A*exp(-t/tau) -> ln(amp) = ln A - t/tau
    slope, _ = np.polyfit(times, np.log(amps), 1)
    tau = -1.0 / slope if slope < 0 else (t[-1] - t[0]) / 5.0

    return omega, tau, phase_sign


def exp_decay_cosine(t, A, tau, omega, phi, c):
    return A * np.exp(-t / tau) * np.cos(omega * t + phi) + c


def fit_exp_decay_cosine(t, y, fix_offset=None, fix_phase=False, return_cov=True):
    """
    y(t) = A exp(-t/tau) cos(omega t + phi) + c
    - omega from median zero-crossing separation
    - tau from ln(abs(max) per zero-crossing segment) vs time
    - optional fixed offset and/or phase-from-first-sample
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    c0 = np.median(y) if fix_offset is None else float(fix_offset)
    yc = y
    # yc = y - c0
    # amplitude/phase from first sample (efficient, no FFT)
    A0 = np.max(np.abs(yc)) if np.any(yc) else 1.0
    A0 = max(A0, 1e-12)

    omega0, tau0, phase_sign = _omega_tau_from_segments(t, yc)
    if tau0 is None:
        if abs(np.mean(yc[:3])) < 0.1:
            phi = 0
        else:
            phi = np.pi / 2 * -phase_sign

        if return_cov:
            return dict(A=A0, tau=None, omega=omega0, phi=phi, offset=c0), None
        else:
            return dict(A=A0, tau=None, omega=omega0, phi=phi, offset=c0)
    # estimate phi from first value if requested; otherwise start at 0
    fit_phase = type(fix_phase) in (float, int)
    if fit_phase:
        # y0 = A e^{-t0/tau} cos(omega t0 + phi)
        # cos_arg = np.clip(yc[0]/(A0*np.exp(-t[0]/tau0)), -1.0, 1.0)
        phi0 = fix_phase
        # print("Initial phase fixed to:", phi0)
    else:
        init_pts = np.mean(yc[:2])
        # print("Initial points mean:", init_pts)
        if init_pts > 0.5 * max(yc):
            phi0 = 0
        elif init_pts < 0.5 * min(yc):
            phi0 = np.pi
        else:
            phi0 = np.pi / 2 * -phase_sign
    # print("Fit phase fixed:", fit_phase, fix_phase, "->", phi0)

    p0_full = [A0, tau0, omega0, phi0, c0]
    free = [True, True, True, not fit_phase, fix_offset is None]
    p0 = [p for p, f in zip(p0_full, free) if f]
    lbound = [0, -np.inf, 0, -2 * np.pi, -np.inf]
    ubound = [np.inf, np.inf, np.inf, 2 * np.pi, np.inf]
    lbound = [l for l, f in zip(lbound, free) if f]
    ubound = [u for u, f in zip(ubound, free) if f]

    def wrapped(t, *pars):
        full, j = [], 0
        for f, p in zip(free, p0_full):
            if f:
                full.append(pars[j])
                j += 1
            else:
                full.append(p)
        return exp_decay_cosine(t, *full)

    try:
        popt, pcov = curve_fit(wrapped, t, y, p0=p0, bounds=(lbound, ubound))
    except RuntimeError:
        print("Fit did not converge; returning initial guesses.")
        fit_params = dict(A=A0, tau=tau0, omega=omega0, phi=phi0, offset=c0)
        return (fit_params, None) if return_cov else fit_params

    # expand to full parameter set
    full, j = [], 0
    for f, p in zip(free, p0_full):
        full.append(popt[j] if f else p)
        if f:
            j += 1
    A, tau, omega, phi, c = full

    fit_results = dict(A=A, tau=tau, omega=omega, phi=phi, offset=c)

    return (fit_results, pcov) if return_cov else fit_results


def plot_exp_decay_fit(t, y, fit_params, show_envelope=True, show_zero_crossings=True, ax=None, override_color=None):
    """
    Plot data and fitted exponentially decaying oscillation.

    Parameters
    ----------
    t, y : array_like
        Time and signal arrays.
    fit_params : dict
        Output of fit_exp_decay_cosine() with keys A, tau, omega, phi, offset.
    show_envelope : bool, optional
        Plot ±A exp(-t/tau) envelopes.
    show_zero_crossings : bool, optional
        Mark zero crossings for reference.
    """
    A, tau, omega, phi, c = (fit_params[k] for k in ("A", "tau", "omega", "phi", "offset"))

    t_fit = np.linspace(0, t[-1], 100)
    y_fit = A * np.exp(-t_fit / tau) * np.cos(omega * t_fit + phi) + c

    if ax is None:
        plt.figure(figsize=(7, 4))
        ax = plt.gca()

    if override_color is None:
        ax.plot(t, y, "k.", ms=5, label="Data")
        ax.plot(t_fit, y_fit, "r-", lw=3, label="Fit", alpha=0.7)
    else:
        ax.plot(t, y, ".", color=override_color, ms=5, label="Data")
        ax.plot(t_fit, y_fit, "-", color=override_color, lw=3, label="Fit", alpha=0.7)

    if show_envelope:
        env = np.abs(A) * np.exp(-t_fit / tau) + c
        ax.plot(t_fit, env, "b--", lw=1, label="Envelope")
        ax.plot(t_fit, -env + 2 * c, "b--", lw=1)

    if show_zero_crossings:
        zt = _zero_crossings(t, y)
        for _z in zt:
            ax.axvline(_z, color="gray", ls=":", lw=0.8)

    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_ylim([-1, 1])
    ax.legend()


# def recover_delta_omegas(Cx, Cy, Cz, Ax, Ay, Az, phix, phiy, omega):
#     # """ """
#     # omega_sq = omega**2
#     # inv_omega_sq = 1 / omega_sq

#     # def model1(delta, Ox, Oy):
#     #     res1 = Cz - (delta**2) * inv_omega_sq
#     #     res2 = Az - inv_omega_sq * (1 - delta**2)
#     #     res3 = omega**2 - (delta**2 + Ox**2 + Oy**2)
#     #     return res1, res2, res3

#     # def model2(delta, Ox, Oy):
#     #     res1 = Cx - (delta * Ox) * inv_omega_sq
#     #     res2 = Ax * np.cos(phix) + (delta * Ox) * inv_omega_sq
#     #     res3 = Ax * np.sin(phix) + (omega * Oy) * inv_omega_sq
#     #     return res1, res2, res3

#     # def model3(delta, Ox, Oy):
#     #     res1 = Cy - (delta * Oy) * inv_omega_sq
#     #     res2 = Ay * np.cos(phiy) + (delta * Oy) * inv_omega_sq
#     #     res3 = Ay * np.sin(phiy) - (omega * Oy) * inv_omega_sq
#     #     return res1, res2, res3

#     # def residuals(params):
#     #     delta, Ox, Oy = params
#     #     r1 = model1(delta, Ox, Oy)
#     #     r2 = model2(delta, Ox, Oy)
#     #     r3 = model3(delta, Ox, Oy)
#     #     return np.concatenate([r1, r2, r3])

#     # x0 = [0, omega * np.sqrt(2), omega * np.sqrt(2)]  # initial guess

#     # res = least_squares(residuals, x0, method="trf")
#     # delta, Ox, Oy = res.x

#     # return delta, Ox, Oy, res
#     prefactor = 1 / (Ax + Ay)
#     if Ay > Ax:
#         sign_x = -np.sign(np.sin(phiy))
#         sign_y = np.sign(np.sin(phix))
#     else:
#         sign_x = np.sign(np.sin(phiy))
#         sign_y = -np.sign(np.sin(phix))
#     print(Ay, Ax, phix, phiy)
#     Ox = omega * np.sqrt(Ay * prefactor)
#     Oy = omega * np.sqrt(Ax * prefactor)
#     return 0, Ox, Oy, None


def recover_delta_omegas(params_x, params_y, params_z, t, Xd, Yd, Zd):
    Ax, Ay = params_x["A"], params_y["A"]
    omega = params_z["omega"]
    taux, tauy, tauz = params_x["tau"], params_y["tau"], params_z["tau"]

    Ox = omega * np.sqrt(Ay / (Ax + Ay))
    Oy = omega * np.sqrt(Ax / (Ax + Ay))

    best_rss = np.inf
    best_params = None

    for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        params = (0, sx * Ox, sy * Oy, taux, tauy, tauz)
        rss = residuals(params, t, Xd, Yd, Zd)
        rss = np.sum(rss**2)
        if rss < best_rss:
            best_rss = rss
            best_params = params

    return best_params[0], best_params[1], best_params[2], best_rss


###################################################
# Direct fit related
###################################################
def Bloch(t, delta, Ox, Oy):
    omega_sq = delta**2 + Ox**2 + Oy**2
    omega = np.sqrt(omega_sq)
    inv_omega_sq = 1 / omega_sq
    cos_omega_t = np.cos(omega * t)
    sin_omega_t = np.sin(omega * t)

    X = inv_omega_sq * (delta * Ox * (1 - cos_omega_t) + omega * Oy * sin_omega_t)
    Y = inv_omega_sq * (delta * Oy * (1 - cos_omega_t) - omega * Ox * sin_omega_t)
    Z = inv_omega_sq * (delta**2 * (1 - cos_omega_t) + omega_sq * cos_omega_t)

    return X, Y, Z


def exp_decay_Bloch(t, delta, Ox, Oy, taux=np.inf, tauy=np.inf, tauz=np.inf):
    X, Y, Z = Bloch(t, delta, Ox, Oy)

    if taux not in [np.inf, None]:
        X *= np.exp(-t / taux)
    if tauy not in [np.inf, None]:
        Y *= np.exp(-t / tauy)
    if tauz not in [np.inf, None]:
        Z *= np.exp(-t / tauz)

    return X, Y, Z


def model(params, t):
    if len(params) == 6:
        delta, Ox, Oy, taux, tauy, tauz = params
        return exp_decay_Bloch(t, delta, Ox, Oy, taux, tauy, tauz)
    elif len(params) == 3:
        delta, Ox, Oy = params
        return exp_decay_Bloch(t, delta, Ox, Oy)
    else:
        raise NotImplementedError("Parameters should either be 3 or 6.")


def residuals(params, t, Xd, Yd, Zd):
    X, Y, Z = model(params, t)
    reg = []
    if len(params) == 6:
        pass
    return np.concatenate([(Xd - X), (Yd - Y), (Zd - Z), reg])


def refine_guess_parameters(params_guess, t, Xd, Yd, Zd, n_omega=5, scale=2, plot_xyz=True, plot_hm=True):
    rss_list = []
    inv_scale = 1 / scale
    best_rss = np.inf
    best_params = None
    delta_guess, Ox_guess, Oy_guess, taux_guess, tauy_guess, tauz_guess = params_guess

    # prepare subplots for x, y, z
    if plot_xyz:
        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        colors = plt.cm.viridis(np.linspace(0, 1, n_omega**2))
        color_idx = 0

    for Oy in np.linspace(inv_scale * Oy_guess, scale * Oy_guess, n_omega):
        for Ox in np.linspace(inv_scale * Ox_guess, scale * Ox_guess, n_omega):
            params = (delta_guess, Ox, Oy, taux_guess, tauy_guess, tauz_guess)
            res = residuals(params, t, Xd, Yd, Zd)
            rss = np.sum(res**2)
            rss_list.append(rss)
            if rss < best_rss:
                best_rss = rss
                best_params = params

            if plot_xyz:
                X_fit, Y_fit, Z_fit = model(params, t)
                for i, (label, fit) in enumerate((("X", X_fit), ("Y", Y_fit), ("Z", Z_fit))):
                    color = colors[color_idx]
                    axs[i].plot(t, fit, "-", color=color, linewidth=3, alpha=0.1)  # line for fit
                    axs[i].set_ylabel(f"<{label}>")
                    axs[i].set_ylim([-1, 1])
                color_idx += 1

    # plot best
    if plot_xyz:
        X_init, Y_init, Z_init = model(params_guess, t)
        X_fit, Y_fit, Z_fit = model(best_params, t)
        for i, (label, data, init, fit) in enumerate(
            (("X", Xd, X_init, X_fit), ("Y", Yd, Y_init, Y_fit), ("Z", Zd, Z_init, Z_fit))
        ):
            axs[i].plot(t, data, "ko", ms=3, label=f"{label}(t) data")
            axs[i].plot(t, init, "r--", linewidth=2, alpha=0.7, label=f"{label}(t) initial guess")  # line for init
            axs[i].plot(t, fit, "r-", linewidth=2, alpha=0.7, label=f"{label}(t) best fit")  # line for fit
            axs[i].legend()

    # plot 2D residuals
    if plot_hm:
        plt.figure(figsize=(8, 6))
        plt.imshow(
            np.log10(np.array(rss_list).reshape((n_omega, n_omega))),
            extent=(inv_scale * Ox_guess, scale * Ox_guess, inv_scale * Oy_guess, scale * Oy_guess),
            origin="lower",
            aspect="auto",
        )
        plt.scatter([Ox_guess], [Oy_guess], color="black", label="Initial Guess")
        plt.scatter([best_params[1]], [best_params[2]], marker="x", color="red", label="Best Scan Guess")
        plt.legend()
        plt.colorbar(label="log10(RSS)")
        plt.xlabel("omega_x")
        plt.ylabel("omega_y")

    return best_params


def fit_exp_decay_Bloch(t, Xd, Yd, Zd, refine_guess=True, plot=False):
    cosine_params_x, _ = fit_exp_decay_cosine(t, Xd)
    cosine_params_y, _ = fit_exp_decay_cosine(t, Yd)
    cosine_params_z, _ = fit_exp_decay_cosine(t, Zd, fix_phase=0)
    for k, params in zip(["X", "Y", "Z"], [cosine_params_x, cosine_params_y, cosine_params_z]):
        print(f"Fitted exponential cosine parameters for <{k}>:")
        for key in ["A", "tau", "omega", "phi", "offset"]:
            print(f"  {key}: {params[key]}")
        print("")

    if plot:
        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        for i, (k, data, params) in enumerate(
            zip(["X", "Y", "Z"], [Xd, Yd, Zd], [cosine_params_x, cosine_params_y, cosine_params_z])
        ):
            plot_exp_decay_fit(
                t, data, params, show_envelope=True, show_zero_crossings=True, ax=axs[i], override_color="r"
            )
            axs[i].set_ylabel(f"<{k}>")
        plt.show()

    delta_guess, Ox_guess, Oy_guess, diag = recover_delta_omegas(
        cosine_params_x, cosine_params_y, cosine_params_z, t, Xd, Yd, Zd
    )
    taux_guess = cosine_params_x["tau"]
    tauy_guess = cosine_params_y["tau"]
    tauz_guess = cosine_params_z["tau"]

    params_guess = (delta_guess, Ox_guess, Oy_guess, taux_guess, tauy_guess, tauz_guess)
    if refine_guess:
        params_guess = refine_guess_parameters(
            params_guess, t, Xd, Yd, Zd, n_omega=5, scale=2, plot_xyz=plot, plot_hm=plot
        )

    lbound = [-np.inf, -np.inf, -np.inf, 0, 0, 0]
    ubound = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    if True:
        # if None in params_guess:
        params_guess = params_guess[:3]
        lbound = lbound[:3]
        ubound = ubound[:3]
    print("Final parameter guesses for least_squares:", params_guess)
    # clip to bounds
    params_guess = np.clip(params_guess, lbound, ubound)
    result = least_squares(
        residuals,
        x0=params_guess,
        bounds=(lbound, ubound),
        args=(t, Xd, Yd, Zd),
    )

    if result.success:
        return result.x
    else:
        raise ValueError(result.message)


def plot_Bloch_fit(
    t, Xd, Yd, Zd, fit_params, axs=None, color="r", line_style="-", alpha=0.7, legend_label="", ignore_decay=False
):
    if len(fit_params) == 3:
        delta, Ox, Oy = fit_params
        taux, tauy, tauz = np.inf, np.inf, np.inf
    elif len(fit_params) == 6:
        delta, Ox, Oy, taux, tauy, tauz = fit_params
    t_fit = np.linspace(t[0], t[-1], 100)
    if ignore_decay:
        X_fit, Y_fit, Z_fit = exp_decay_Bloch(t_fit, delta, Ox, Oy)
    else:
        X_fit, Y_fit, Z_fit = exp_decay_Bloch(t_fit, delta, Ox, Oy, taux, tauy, tauz)

    fs = 16
    if axs is None:
        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    if legend_label:
        legend_label = legend_label.strip() + " "
    for i, (label, data, fit) in enumerate((("X", Xd, X_fit), ("Y", Yd, Y_fit), ("Z", Zd, Z_fit))):
        axs[i].plot(t, data, "o", color=color, ms=3, label=f"{legend_label} data")
        axs[i].plot(
            t_fit, fit, line_style, color=color, linewidth=3, alpha=alpha, label=f"{legend_label} fit"
        )  # line for fit
        axs[i].set_ylabel(f"<{label}>", fontsize=fs)
        axs[i].set_ylim([-1.05, 1.05])
        axs[i].tick_params(axis="both", which="major", labelsize=fs)
    # legend outside plot
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1)
    plt.xlabel("t (ns)", fontsize=fs)

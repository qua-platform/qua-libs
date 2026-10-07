import numpy as np
import xarray as xr
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from calibration_utils.readout_optimal_integration_weights.analysis import has_variance
from calibration_utils.readout_optimal_integration_weights.weights import TOLERANCE


def _grid(ds: xr.Dataset, qubits: list[AnyTransmon], title: str):
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    grid.fig.suptitle(title)
    return grid


def _finish(grid, size=(15, 9)):
    grid.fig.set_size_inches(*size)
    grid.fig.tight_layout()
    return grid.fig


def _by_name(qubits: list[AnyTransmon]) -> dict:
    return {q.name: q for q in qubits}


def _iq_imbalance_label(qd: xr.Dataset) -> str:
    """Short panel-title fragment reporting the IQ-imbalance estimate and whether it was
    folded into the weight -- see analysis.OptimalWeightsFit.iq_imbalance_*."""
    if bool(qd.iq_imbalance_applied.values):
        return f"|b|={float(qd.iq_imbalance_abs):.3g} corrected"
    return f"|b|={float(qd.iq_imbalance_abs):.3g} not corrected"


# --- 1. variance -------------------------------------------------------------------------


def plot_variance(ds_fit: xr.Dataset, qubits: list[AnyTransmon]):
    """Per-sample variance for g and e, and the signal bound hb = min(|z| + SIGMA_NUM*sigma, ADC full scale) used by the
    overflow normalization (weights.normalization_factor). Only meaningful when the second
    moments were acquired (`measure_variance=True`); guarded here (not just by callers) since
    `ds_fit` lacks the `var_i_*`/`var_q_*` variables entirely otherwise."""
    if not has_variance(ds_fit):
        grid = _grid(ds_fit, qubits, "Per-sample variance and signal bound (hb) -- not acquired")
        for ax, qubit in grid_iter(grid):
            ax.text(0.5, 0.5, "measure_variance=False\n(no data)", ha="center", va="center")
            ax.set_title(qubit["qubit"])
        return _finish(grid)
    grid = _grid(ds_fit, qubits, "Per-sample variance and signal bound (hb)")
    for ax, qubit in grid_iter(grid):
        qd = ds_fit.sel(qubit=qubit["qubit"])
        t = qd.readout_time.values
        var_g = (qd.var_i_g + qd.var_q_g).values
        var_e = (qd.var_i_e + qd.var_q_e).values
        ax.plot(t, var_g, label="var g", color="tab:blue")
        ax.plot(t, var_e, label="var e", color="tab:red")
        ax2 = ax.twinx()
        ax2.plot(t, np.abs(qd.hb.values), label="hb", color="tab:green", linewidth=1)
        ax2.set_ylabel("hb [V]", color="tab:green")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Variance [V^2]")
        ax.set_title(qubit["qubit"])
    grid.fig.legend(*grid.fig.axes[0].get_legend_handles_labels(), loc="upper right", ncols=2)
    return _finish(grid)


# --- 2. envelopes (demod sanity check) ----------------------------------------------------


def plot_envelopes(ds_fit: xr.Dataset, qubits: list[AnyTransmon]):
    """Demodulated (baseband) envelopes for g and e -- IQ-imbalance-corrected when
    `correct_iq_imbalance` applied (see the panel title for `|b|` and whether it was applied).
    A residual fast oscillation here means the IF or the sign of omega is wrong -- no
    constant-phase rotation can fix that."""
    grid = _grid(ds_fit, qubits, "Demodulated envelopes env_g(t), env_e(t)")
    for ax, qubit in grid_iter(grid):
        qd = ds_fit.sel(qubit=qubit["qubit"])
        t = qd.readout_time.values
        env_g = qd.env_g.values
        env_e = qd.env_e.values
        ax.plot(t, env_g.real * 1e3, label="I g", color="tab:blue")
        ax.plot(t, env_g.imag * 1e3, label="Q g", color="tab:blue", linestyle="--")
        ax.plot(t, env_e.real * 1e3, label="I e", color="tab:red")
        ax.plot(t, env_e.imag * 1e3, label="Q e", color="tab:red", linestyle="--")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Envelope [mV]")
        ax.set_title(f"{qubit['qubit']} ({_iq_imbalance_label(qd)})")
    grid.fig.legend(*grid.fig.axes[0].get_legend_handles_labels(), loc="upper right", ncols=4)
    return _finish(grid)


# --- 3. demodulation debug: before/after, in time and in frequency ------------------------


def plot_demod_comparison(ds_fit: xr.Dataset, qubits: list[AnyTransmon], window_ns: int = 100):
    """Raw ADC I/Q (before software demodulation, thin) against the baseband envelope I/Q
    (after, thick) for both states, zoomed to the first `window_ns` samples so the IF carrier
    is resolvable at all -- over a full readout window the oscillation is far too dense to see.

    What correct demodulation looks like: the raw curves oscillate at f_IF while the envelope
    curves are smooth on this scale. Envelope curves that still oscillate mean `omega_t` is
    wrong; if they oscillate at roughly TWICE the raw rate, the SIGN of the IF is wrong (see
    `plot_demod_spectrum`, which separates those two cases unambiguously)."""
    grid = _grid(ds_fit, qubits, f"Before vs. after software demodulation (first {window_ns} ns)")
    for ax, qubit in grid_iter(grid):
        qd = ds_fit.sel(qubit=qubit["qubit"]).isel(readout_time=slice(0, window_ns))
        t = qd.readout_time.values
        for state, color in (("g", "tab:blue"), ("e", "tab:red")):
            env = qd[f"env_{state}"].values
            raw = [qd[f"a_i_{state}"].values, qd[f"a_q_{state}"].values]
            for comp, style, (pre, post) in zip(("I", "Q"), ("-", "--"), zip(raw, (env.real, env.imag))):
                ax.plot(t, pre * 1e3, style, color=color, linewidth=0.8, alpha=0.35, label=f"{comp} {state} raw")
                ax.plot(t, post * 1e3, style, color=color, linewidth=1.8, label=f"{comp} {state} demod")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Signal [mV]")
        ax.set_title(qubit["qubit"])
    grid.fig.legend(*grid.fig.axes[0].get_legend_handles_labels(), loc="upper right", ncols=4, fontsize="small")
    return _finish(grid)


def plot_demod_spectrum(ds_fit: xr.Dataset, qubits: list[AnyTransmon]):
    """Magnitude spectrum of the |g> trace before demodulation (z, thin) and after (env, thick),
    with the resonator's +/-f_IF marked (the signal bin and its mirror -- IQ-imbalance leakage
    shows up as residual power at the mirror, -f_IF, even after demodulation when uncorrected).
    This is the decisive test of `omega_t`: the raw trace peaks at +f_IF, and correct
    demodulation moves that peak to DC. A post-demod peak sitting at -2*f_IF instead means the
    sign of the IF (or of omega) is inverted; a peak at some other offset means the IF value
    itself is off by that much."""
    by_name = _by_name(qubits)
    grid = _grid(ds_fit, qubits, "Demodulation check in frequency: |FFT| before vs. after (|g>)")
    for ax, qubit in grid_iter(grid):
        qd = ds_fit.sel(qubit=qubit["qubit"])
        env = qd.env_g.values
        z = (qd.a_i_g.values + 1j * qd.a_q_g.values)[: env.size]
        valid = np.isfinite(z) & np.isfinite(env)
        z, env = z[valid], env[valid]
        freq_mhz = np.fft.fftshift(np.fft.fftfreq(z.size, d=1e-9)) / 1e6
        for sig, label, width in ((z, "before (z)", 0.8), (env, "after (env)", 1.8)):
            spectrum = np.abs(np.fft.fftshift(np.fft.fft(sig))) / max(sig.size, 1)
            ax.semilogy(freq_mhz, spectrum * 1e3, label=label, linewidth=width)
        f_if_mhz = by_name[qubit["qubit"]].resonator.intermediate_frequency / 1e6
        for f, style, seg_label in ((f_if_mhz, "--", "f_IF"), (-f_if_mhz, ":", "mirror")):
            ax.axvline(f, color="grey", linestyle=style, linewidth=1, label=seg_label)
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("|FFT| [mV]")
        ax.set_title(f"{qubit['qubit']} (f_IF={f_if_mhz:.1f} MHz, {_iq_imbalance_label(qd)})")
    grid.fig.legend(*grid.fig.axes[0].get_legend_handles_labels(), loc="upper right", ncols=2)
    return _finish(grid)


def plot_weight_spectrum(ds_fit: xr.Dataset, qubits: list[AnyTransmon]):
    """Magnitude spectrum of the ACTUAL deployed weight W = env_e - env_g with the IQ-imbalance
    fold applied, with +/-f_IF marked -- answers "does a residual carrier near +/-f_IF survive
    into the weight". A large peak at f_IF or its mirror means the residual is not common-mode
    between g and e and is worth chasing (e.g. per-shot timing jitter, or a genuinely
    state-dependent artifact)."""
    by_name = _by_name(qubits)
    grid = _grid(ds_fit, qubits, "Does a residual carrier survive into the deployed weight W?")
    for ax, qubit in grid_iter(grid):
        qd = ds_fit.sel(qubit=qubit["qubit"])
        W = qd.W.values
        W = W[np.isfinite(W)]
        freq_mhz = np.fft.fftshift(np.fft.fftfreq(W.size, d=1e-9)) / 1e6
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(W))) / max(W.size, 1)
        ax.semilogy(freq_mhz, spectrum * 1e3, label="W (deployed weight)", linewidth=1.8)
        f_if_mhz = by_name[qubit["qubit"]].resonator.intermediate_frequency / 1e6
        for f, style, seg_label in ((f_if_mhz, "--", "f_IF"), (-f_if_mhz, ":", "mirror")):
            ax.axvline(f, color="grey", linestyle=style, linewidth=1, label=seg_label)
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("|FFT| [mV]")
        ax.set_title(f"{qubit['qubit']} (f_IF={f_if_mhz:.1f} MHz, {_iq_imbalance_label(qd)})")
    grid.fig.legend(*grid.fig.axes[0].get_legend_handles_labels(), loc="upper right", ncols=2)
    return _finish(grid)


# --- 4. IQ trajectory (spec §8.1) ---------------------------------------------------------


def plot_iq_trajectory(ds_fit: xr.Dataset, qubits: list[AnyTransmon]):
    """env_g(t) and env_e(t) as points on a polar axis (radius = |env| in mV, angle = arg env).
    Dots only, with low alpha so regions where many samples overlap read darker."""
    grid = _grid(ds_fit, qubits, "IQ trajectory of the demodulated envelope")
    for ax, qubit in grid_iter(grid):
        # QubitGrid creates cartesian axes; swap this one for a polar axis in the same slot.
        polar_ax = grid.fig.add_subplot(ax.get_subplotspec(), projection="polar")
        ax.remove()
        qd = ds_fit.sel(qubit=qubit["qubit"])
        for state, color in (("g", "tab:blue"), ("e", "tab:red")):
            env = qd[f"env_{state}"].values
            env = env[np.isfinite(env)]
            polar_ax.scatter(np.angle(env), np.abs(env) * 1e3, s=6, alpha=0.12, linewidths=0, color=color, label=state)
        polar_ax.set_title(qubit["qubit"])
        polar_ax.set_rlabel_position(135)
    handles, labels = polar_ax.get_legend_handles_labels()
    grid.fig.legend(handles, labels, loc="upper right", ncols=2, markerscale=3)
    return _finish(grid)


# --- 5. weight magnitude and phase (spec §8.2) ---------------------------------------------


def plot_weight(ds_fit: xr.Dataset, qubits: list[AnyTransmon]):
    """|W_norm(t)| at 1 ns and chunked (so chunking loss is visible), and arg(W(t)) -- the
    time-varying phase a stock ReadoutPulse (real envelope + one global angle) cannot express.
    This is `W`, i.e. the IQ-imbalance correction already folded in when applied (see the panel
    title) -- the weight actually written to hardware, applied to the raw ADC."""
    grid = _grid(ds_fit, qubits, "Normalized weight magnitude and phase")
    for ax, qubit in grid_iter(grid):
        qd = ds_fit.sel(qubit=qubit["qubit"])
        t = qd.readout_time.values
        t_chunk = qd.chunk_time.values
        w_norm = qd.W_norm.values
        w_chunked = qd.W_chunked.values
        ax.plot(t, np.abs(w_norm), label="|W| (1 ns)", color="tab:purple", alpha=0.5)
        ax.step(t_chunk, np.abs(w_chunked), where="post", label="|W| (4 ns)", color="tab:purple")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("|W_norm|", color="tab:purple")
        ax2 = ax.twinx()
        ax2.plot(t, np.unwrap(np.angle(qd.W.values)), color="tab:orange", linewidth=1, label="arg(W)")
        ax2.set_ylabel("arg(W) [rad]", color="tab:orange")
        ax.set_title(f"{qubit['qubit']} ({_iq_imbalance_label(qd)})")
    grid.fig.legend(*grid.fig.axes[0].get_legend_handles_labels(), loc="upper right", ncols=2)
    return _finish(grid)


# --- 6. normalization headroom -------------------------------------------------------------


def plot_normalization(ds_fit: xr.Dataset, qubits: list[AnyTransmon]):
    """Which of the three OPX overflow limits binds, as a fraction of the DOCUMENTED hardware
    limit: 1.0 is the limit itself (overflow beyond it), TOLERANCE is the design target the
    weights are normalized to. The stored margins are relative to TOLERANCE*limit, so they are
    rescaled by TOLERANCE here."""
    grid = _grid(ds_fit, qubits, "Overflow headroom (fraction of documented limit)")
    for ax, qubit in grid_iter(grid):
        qd = ds_fit.sel(qubit=qubit["qubit"])
        labels = ["weight", "adc_mul", "adc_sum"]
        fractions = [
            TOLERANCE * float(qd.margin_weight),
            TOLERANCE * float(qd.margin_adc_mul),
            TOLERANCE * float(qd.margin_adc_sum),
        ]
        colors = ["tab:green" if f <= TOLERANCE else "tab:orange" if f <= 1.0 else "tab:red" for f in fractions]
        ax.bar(labels, fractions, color=colors)
        ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="documented limit")
        ax.axhline(TOLERANCE, color="tab:blue", linestyle=":", linewidth=1.5, label=f"TOLERANCE ({TOLERANCE:g})")
        ax.set_ylabel("value / documented limit")
        ax.set_title(f"{qubit['qubit']} (binding: {qd.binding_limit.values!s})")
        ax.legend(fontsize=7)
    return _finish(grid)


# --- 7. SNR breakdown ------------------------------------------------------------------------


def plot_snr(ds_fit: xr.Dataset, qubits: list[AnyTransmon]):
    """Predicted SNR for constant weights (today), a real envelope with the best single global
    phase (what a stock ReadoutPulse could do), and the full complex matched filter deployed."""
    grid = _grid(ds_fit, qubits, "Predicted SNR: constant vs. real-envelope vs. full complex")
    for ax, qubit in grid_iter(grid):
        qd = ds_fit.sel(qubit=qubit["qubit"])
        labels = ["constant", "real envelope", "full complex"]
        values = [
            float(qd.snr_constant),
            float(qd.snr_real_envelope),
            float(qd.snr),
        ]
        ax.bar(labels, values, color=["tab:gray", "tab:orange", "tab:green"])
        ax.set_ylabel("Predicted SNR")
        snr_const = float(qd.snr_constant)
        env_gain = float(qd.snr_real_envelope) / snr_const if snr_const > 0 else float("inf")
        ax.set_title(f"{qubit['qubit']}\nvs constant: envelope x{env_gain:.2f}, complex x{float(qd.snr_gain):.2f}")
    return _finish(grid)


def plot_raw_data_with_fit(ds_fit: xr.Dataset, qubits: list[AnyTransmon]):
    """The always-on plots: IQ trajectory, normalized weight, demodulated envelopes, weight
    spectrum and predicted SNR. The extra debug set (variance, before/after demodulation in time
    and frequency, normalization headroom) is produced separately by `plot_data` when
    `node.parameters.debug_plots` is True -- see the other `plot_*` functions above."""
    return {
        "iq_trajectory": plot_iq_trajectory(ds_fit, qubits),
        "weight": plot_weight(ds_fit, qubits),
        "envelopes": plot_envelopes(ds_fit, qubits),
        "weight_spectrum": plot_weight_spectrum(ds_fit, qubits),
        "snr": plot_snr(ds_fit, qubits),
    }

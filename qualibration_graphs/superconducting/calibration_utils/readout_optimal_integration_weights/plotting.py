import numpy as np
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from calibration_utils.readout_optimal_integration_weights.analysis import (
    complex_values,
    has_variance,
)
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
        env_g = complex_values(qd, "env_g")
        env_e = complex_values(qd, "env_e")
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
            env = complex_values(qd, f"env_{state}")
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
        env = complex_values(qd, "env_g")
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
    """Magnitude spectrum of the ideal complex weight W = env_e - env_g (variance-weighted when
    enabled) with the IQ-imbalance fold applied, with +/-f_IF marked -- answers "does a residual
    carrier near +/-f_IF survive into the weight". A large peak at f_IF or its mirror means the
    residual is not common-mode between g and e and is worth chasing (e.g. per-shot timing jitter,
    or a genuinely state-dependent artifact).

    When the IQ-imbalance correction was applied, W also carries the fold term
    -b*exp(-2i*omega*t)*conj(W) (weights.fold_iq_imbalance_into_weight), a copy of the weight
    shifted to -2*f_IF. That is expected, not a residual carrier, so -2*f_IF is marked too
    (wrapped into the +/-500 MHz band the 1 ns sampling resolves, where it aliases)."""
    by_name = _by_name(qubits)
    grid = _grid(ds_fit, qubits, "Carrier check: spectrum of the ideal complex weight W")
    nyquist_mhz = 500.0
    for ax, qubit in grid_iter(grid):
        qd = ds_fit.sel(qubit=qubit["qubit"])
        W = complex_values(qd, "W")
        W = W[np.isfinite(W)]
        freq_mhz = np.fft.fftshift(np.fft.fftfreq(W.size, d=1e-9)) / 1e6
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(W))) / max(W.size, 1)
        ax.semilogy(freq_mhz, spectrum * 1e3, linewidth=1.8)
        f_if_mhz = by_name[qubit["qubit"]].resonator.intermediate_frequency / 1e6
        for f, style in ((f_if_mhz, "--"), (-f_if_mhz, ":")):
            ax.axvline(f, color="grey", linestyle=style, linewidth=1)
        if bool(qd.iq_imbalance_applied.values):
            fold_mhz = (-2 * f_if_mhz + nyquist_mhz) % (2 * nyquist_mhz) - nyquist_mhz
            ax.axvline(fold_mhz, color="tab:red", linestyle="-.", linewidth=1)
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("|FFT| [mV]")
        ax.set_title(f"{qubit['qubit']} (f_IF={f_if_mhz:.1f} MHz, {_iq_imbalance_label(qd)})")
    handles = [
        Line2D([], [], color="tab:blue", linewidth=1.8, label="W (ideal complex weight)"),
        Line2D([], [], color="grey", linestyle="--", label="+f_IF"),
        Line2D([], [], color="grey", linestyle=":", label="-f_IF (mirror)"),
        Line2D([], [], color="tab:red", linestyle="-.", label="-2 f_IF (IQ-imbalance fold, aliased)"),
    ]
    grid.fig.legend(handles=handles, loc="upper left", ncols=1)  # the top-left grid slot is empty
    return _finish(grid)


# --- 4. IQ trajectory (spec §8.1) ---------------------------------------------------------


def plot_iq_trajectory(ds_fit: xr.Dataset, qubits: list[AnyTransmon]):
    """Mean env_g(t) and env_e(t) in the I-Q plane (mV, equal axes), as lines coloured by time:
    light = pulse start, dark = pulse end (Blues for g, Reds for e). The start is marked with a
    circle and the end with a square."""
    grid = _grid(ds_fit, qubits, "IQ trajectory of the mean demodulated envelope (light -> dark = time)")
    for ax, qubit in grid_iter(grid):
        qd = ds_fit.sel(qubit=qubit["qubit"])
        for state, cmap in (("g", "Blues"), ("e", "Reds")):
            env = complex_values(qd, f"env_{state}")
            env = env[np.isfinite(env)] * 1e3
            if env.size < 2:
                continue
            points = np.column_stack([env.real, env.imag])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            lines = LineCollection(segments, cmap=cmap, linewidth=2)
            # Start the colormap at 0.3 so the earliest samples stay visible on a white background.
            lines.set_array(0.3 + 0.7 * np.linspace(0, 1, len(segments)))
            lines.set_clim(0, 1)
            ax.add_collection(lines)
            colour = lines.get_cmap()(0.85)
            ax.plot(env.real[0], env.imag[0], "o", color=colour, markersize=6, markeredgecolor="k")
            ax.plot(env.real[-1], env.imag[-1], "s", color=colour, markersize=6, markeredgecolor="k")
        ax.autoscale_view()
        ax.set_aspect("equal", adjustable="datalim")
        ax.axhline(0, color="k", linewidth=0.4)
        ax.axvline(0, color="k", linewidth=0.4)
        ax.set_xlabel("I [mV]")
        ax.set_ylabel("Q [mV]")
        ax.set_title(qubit["qubit"])
    handles = [
        Line2D([], [], color="tab:blue", label="g"),
        Line2D([], [], color="tab:red", label="e"),
        Line2D([], [], marker="o", color="grey", linestyle="", markeredgecolor="k", label="start"),
        Line2D([], [], marker="s", color="grey", linestyle="", markeredgecolor="k", label="end"),
    ]
    grid.fig.legend(handles=handles, loc="upper right", ncols=4)
    return _finish(grid)


# --- 5. weight magnitude and phase (spec §8.2) ---------------------------------------------


def plot_weight(ds_fit: xr.Dataset, qubits: list[AnyTransmon]):
    """The DEPLOYED real weight (signed projection by default) at 1 ns and chunked to 4 ns (so
    chunking loss is visible), and arg(W(t)) -- the time-varying phase a stock ReadoutPulse (real
    weight + one global angle) cannot express, and which the signed projection partly turns into
    separation. W is the IQ-imbalance-folded matched filter (see the panel title); the deployed
    weight is applied to the raw ADC. arg(W) is drawn as unwrapped points (unwrapped within
    each stretch) only where |W| exceeds 10% of its maximum: the phase of a near-zero
    weight is noise, and unwrapping through it runs off to hundreds of radians."""
    grid = _grid(ds_fit, qubits, "Deployed real weight and phase of the ideal complex weight")
    handles = []
    for ax, qubit in grid_iter(grid):
        qd = ds_fit.sel(qubit=qubit["qubit"])
        t = qd.readout_time.values
        t_chunk = qd.chunk_time.values
        (h1,) = ax.plot(t, qd.w_deployed.values, label="deployed w (1 ns)", color="tab:purple", alpha=0.5)
        (h2,) = ax.step(
            t_chunk, qd.w_deployed_chunked.values, where="post", label="deployed w (4 ns)", color="tab:purple"
        )
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("weight (normalized)", color="tab:purple")
        W = complex_values(qd, "W")
        mag = np.abs(W)
        valid = mag >= 0.1 * np.nanmax(mag)
        ax2 = ax.twinx()
        # Phase unwrapped within each contiguous stretch where |W| is large: unwrapping across the
        # near-zero stretches would accumulate the phase noise of a vanishing weight into a runaway curve.
        edges = np.flatnonzero(np.diff(np.concatenate([[0], valid.astype(int), [0]])))
        for start, stop in zip(edges[::2], edges[1::2]):
            ax2.scatter(t[start:stop], np.unwrap(np.angle(W[start:stop])), s=4, color="tab:orange")
        ax2.set_ylabel("arg(W) [rad]", color="tab:orange")
        ax.set_title(f"{qubit['qubit']} ({_iq_imbalance_label(qd)}, phi={float(qd.phi_signed_rad):.2f} rad)")
        handles = [
            h1,
            h2,
            Line2D([], [], marker="o", markersize=3, linestyle="", color="tab:orange", label="arg(W) unwrapped"),
        ]
    grid.fig.legend(handles=handles, loc="upper left", ncols=1)  # the top-left grid slot is empty
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
    """Predicted SNR for four weight shapes: constant (today's boxcar), the real weight |W|
    (the previous deployment), the signed real weight (best for a stock ReadoutPulse: real weight
    and one global angle) and the full complex matched filter (the ceiling, not deployable yet).

    Bars are the bias-corrected ("net") values: a weight built from the averaged g/e difference
    also fits that difference's noise, so the raw SNR overstates every shape but the constant
    one; the black ticks mark the raw values. The title gives each net gain over the constant
    weight and the gain of the weight actually deployed (normalized, chunked to 4 ns;
    `Parameters.deployed_weight`), plus how many sigma the deployed SNR^2 sits above a weight
    fitted to pure noise."""
    grid = _grid(ds_fit, qubits, "Predicted SNR, noise-fit bias removed (ticks: raw)")
    for ax, qubit in grid_iter(grid):
        qd = ds_fit.sel(qubit=qubit["qubit"])
        labels = ["constant", "|W| real", "signed real", "full complex"]
        net = [float(qd.snr_constant_net), float(qd.snr_real_envelope_net), float(qd.snr_signed_net), float(qd.snr_net)]
        raw = [float(qd.snr_constant), float(qd.snr_real_envelope), float(qd.snr_signed), float(qd.snr)]
        ax.bar(labels, net, color=["tab:gray", "tab:orange", "tab:blue", "tab:green"])
        ax.plot(
            labels,
            raw,
            linestyle="",
            marker="_",
            color="k",
            markersize=22,
            markeredgewidth=2,
            label="raw (noise-fitted)",
        )
        ax.set_ylabel("Predicted SNR" + ("" if bool(has_variance(ds_fit)) else " [arb. units]"))
        snr_const = float(qd.snr_constant_net)
        gains = [v / snr_const if snr_const > 0 else float("inf") for v in net]
        ax.set_title(
            f"{qubit['qubit']}: deployed x{float(qd.snr_gain):.2f}, {float(qd.detection_sigma):.1f} sigma above noise fit\n"
            f"|W| x{gains[1]:.2f}, signed x{gains[2]:.2f}, complex x{gains[3]:.2f}"
        )
    handles, labels = ax.get_legend_handles_labels()
    grid.fig.legend(handles, labels, loc="upper right")
    return _finish(grid)


def plot_raw_data_with_fit(ds_fit: xr.Dataset, qubits: list[AnyTransmon]):
    """The always-on plots, in reading order (data -> weight -> check -> result): demodulated
    envelopes, normalized weight, weight spectrum (carrier check), predicted SNR, and the IQ
    trajectory last. The extra debug set (variance, before/after demodulation in time
    and frequency, normalization headroom) is produced separately by `plot_data` when
    `node.parameters.debug_plots` is True -- see the other `plot_*` functions above."""
    return {
        "envelopes": plot_envelopes(ds_fit, qubits),
        "weight": plot_weight(ds_fit, qubits),
        "weight_spectrum": plot_weight_spectrum(ds_fit, qubits),
        "snr": plot_snr(ds_fit, qubits),
        "iq_trajectory": plot_iq_trajectory(ds_fit, qubits),
    }

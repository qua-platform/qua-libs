"""Plots with drive rows, target columns and Power Rabi amplitude labels."""

import matplotlib.pyplot as plt
import numpy as np
from qualibration_libs.analysis import oscillation


def plot_rabi(ds, result):
    names = list(ds.qubit.values)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), squeeze=False)
    for ax, target in zip(axes.flat, names):
        for drive in ds.drive_qubit.values:
            selection = dict(qubit=target, drive_qubit=drive)
            x = ds.amp_prefactor * ds.drive_mV_per_prefactor.sel(drive_qubit=drive) * ds.pair_amp_scale.sel(selection)
            (line,) = ax.plot(x, ds.state.sel(selection), ".-.", ms=3, label=f"Drive {drive}")
            fit = result.fit.sel(selection)
            if np.all(np.isfinite(fit)):
                y = oscillation(
                    ds.amp_prefactor.values, *[float(fit.sel(fit_vals=k)) for k in ("a", "f", "phi", "offset")]
                )
                ax.plot(x, y, color=line.get_color(), lw=1)
        ax.set(xlabel="Pulse amplitude [mV]", ylabel="Excited-state population", title=f"Measure {target}")
        ax.legend()
    fig.suptitle("Drive-line crosstalk amplitude")
    fig.tight_layout()
    return fig


def plot_matrix(ds, kind):
    r = ds.amplitude_ratio.transpose("drive_qubit", "qubit")
    if kind == "amplitude":
        values, title, cmap = np.log10(r.values), "Crosstalk magnitude r (colour: log10 r)", "RdPu"
    elif kind == "phase":
        values, title, cmap = (
            np.rad2deg(ds.compensation_phase_rad.transpose(*r.dims).values),
            "Compensation phase [deg]",
            "twilight",
        )
    else:
        values, title, cmap = np.log10(r.values), "Added compensation (amp_ratio, θ°)", "RdPu"
        np.fill_diagonal(values, np.nan)
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(
        np.ma.masked_invalid(values),
        cmap=cmap,
        aspect="auto",
        vmin=0 if kind == "phase" else -3,
        vmax=360 if kind == "phase" else 0,
    )
    ax.set_xticks(range(len(r.qubit)), r.qubit.values)
    ax.set_yticks(range(len(r.drive_qubit)), [f"DL {name}" for name in r.drive_qubit.values])
    ax.set(xlabel="Target qubit", ylabel="Drive line", title=title)
    for (i, j), value in np.ndenumerate(values):
        if kind == "compensation" and i == j:
            label = "0"
        elif not np.isfinite(value):
            label = "N/A"
        elif kind == "amplitude":
            label = f"{r.values[i, j]:.5f}"
        elif kind == "phase":
            label = f"{value:.2f}°"
        else:
            theta = float(ds.compensation_phase_rad.sel(drive_qubit=r.drive_qubit[i], qubit=r.qubit[j]))
            passed = bool(ds.validation_passed.sel(drive_qubit=r.drive_qubit[i], qubit=r.qubit[j]))
            label = f"{r.values[i, j]:.5f}, {np.rad2deg(theta):.2f}°\n{'PASS' if passed else 'FAIL'}"
        ax.text(
            j,
            i,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="white" if kind != "phase" and np.isfinite(value) and value > -1.3 else "black",
        )
    fig.colorbar(image, ax=ax, label="Phase [deg]" if kind == "phase" else "log10(amp_ratio)")
    fig.tight_layout()
    return fig


def _amplitude_axis(ax, factor_to_mV):
    secondary = ax.secondary_xaxis("top", functions=(lambda x: x / factor_to_mV, lambda x: x * factor_to_mV))
    secondary.set_xlabel("amplitude prefactor")


def plot_phase(ds, result):
    pairs = [(d, t) for d in ds.drive_qubit.values for t in ds.qubit.values if d != t]
    fig, axes = (
        plt.subplots(3, 2, figsize=(12, 12), squeeze=False)
        if len(pairs) == 6
        else plt.subplots(len(pairs), 1, figsize=(7, 4 * len(pairs)), squeeze=False)
    )
    for ax, (drive, target) in zip(axes.flat, pairs):
        selection = dict(qubit=target, drive_qubit=drive)
        scale = float(ds.drive_mV_per_prefactor.sel(drive_qubit=drive))
        phases = ds.phase_turns.sel(selection).values * 2 * np.pi
        im = ax.pcolormesh(
            ds.amp_prefactor * scale,
            phases,
            ds.state.sel(selection).transpose("phase_index", "amp_prefactor"),
            shading="auto",
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        theta = float(result.compensation_phase_rad.sel(selection))
        r = float(result.amplitude_ratio.sel(selection))
        ax.axhline(theta, color="white", ls="--")
        ax.set(
            xlabel="Pulse amplitude [mV]",
            ylabel="Compensation phase [rad]",
            title=f"{drive} → {target}, r={r:.6f}, θ={np.rad2deg(theta):.2f}°",
        )
        _amplitude_axis(ax, scale)
        fig.colorbar(im, ax=ax, label="Excited-state population")
    fig.tight_layout()
    return fig


def plot_check(ds, result):
    pairs = [(d, t) for d in ds.drive_qubit.values for t in ds.qubit.values if d != t]
    fig, axes = (
        plt.subplots(3, 2, figsize=(12, 12), squeeze=False)
        if len(pairs) == 6
        else plt.subplots(len(pairs), 1, figsize=(7, 4 * len(pairs)), squeeze=False)
    )
    for ax, (drive, target) in zip(axes.flat, pairs):
        selection = dict(qubit=target, drive_qubit=drive)
        scale = float(ds.drive_mV_per_prefactor.sel(drive_qubit=drive))
        for mode in ("no_drive", "uncompensated", "compensated"):
            ax.plot(ds.amp_prefactor * scale, ds.state.sel(selection).sel(mode=mode), ".-.", ms=3, label=mode)
        passed = bool(result.validation_passed.sel(selection))
        ax.set(
            xlabel="Pulse amplitude [mV]",
            ylabel="Target excited-state population",
            ylim=(-0.02, 1.02),
            title=f"{drive} → {target}: {'passed' if passed else 'failed'}",
        )
        _amplitude_axis(ax, scale)
        ax.legend()
    fig.tight_layout()
    return fig


def plot_validation_table(results):
    """Numerical companion to the three measured validation curves."""
    columns = ["Pair", "r", "θ [deg]", "No drive", "Uncompensated", "Compensated", "Validation"]
    rows = [
        [
            pair,
            f"{row['amplitude_ratio']:.6f}",
            f"{row['compensation_phase_deg']:.3f}",
            f"{row['no_drive_mean_population']:.4f}",
            f"{row['uncompensated_mean_population']:.4f}",
            f"{row['compensated_mean_population']:.4f}",
            "PASS" if row["passed"] else "FAIL",
        ]
        for pair, row in results.items()
    ]
    fig, ax = plt.subplots(figsize=(12, max(3, len(rows) * 0.4)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    ax.set_title("Independent compensation validation — mean excited-state population")
    fig.tight_layout()
    return fig

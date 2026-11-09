"""Plot raw CKP maps, Ramsey diagnostics and matched photon occupations."""

import matplotlib.pyplot as plt
import numpy as np


def plot_ckp(ds, fit_results):
    """Plot state-dependent Stark maps, fitted ridges and photon occupations."""
    figures = {}
    for name in ds.qubit.values:
        selected = ds.sel(qubit=name)
        result = fit_results[str(name)]
        figure, axes = plt.subplots(ds.sizes["amp_factor"], 2, figsize=(12, 3 * ds.sizes["amp_factor"]), squeeze=False)
        for ai, amplitude in enumerate(ds.amp_factor.values):
            photon_label = ""
            if result["success"]:
                photon = float(selected.photon_number.isel(amp_factor=ai))
                uncertainty = float(selected.photon_number_uncertainty.isel(amp_factor=ai))
                photon_label = rf"; $n_g$={photon:.2f} ± {uncertainty:.2f}"
            for si in range(2):
                axis = axes[ai, si]
                line = selected.isel(amp_factor=ai, prepared_state=si)
                probability = line.state if si == 0 else 1 - line.state
                axis.pcolormesh(ds.resonator_detuning, ds.qubit_detuning, probability.T, shading="auto", vmin=0, vmax=1)
                axis.plot(ds.resonator_detuning, line.stark_ridge_mhz, "w.", ms=3)
                axis.plot(ds.resonator_detuning, line.stark_ridge_fit_mhz, color="orange")
                axis.set(
                    title=f"{name}: amp={amplitude:g}, prepare |{si}>" + photon_label,
                    xlabel="resonator detuning (MHz)",
                    ylabel="qubit probe detuning (MHz)",
                )
        if result["success"]:
            figure.suptitle(
                f"CKP: linewidth={result['linewidth_mhz']:.3g} MHz; 2chi={result['dispersive_shift_mhz']:.3g} MHz\n"
                r"$n_g$: ground-state steady-state photon number at comparison drive frequency"
            )
        else:
            figure.suptitle(f"CKP FIT FAILED: {result.get('reason', 'parameters unresolved')}")

        figure.tight_layout()
        figures[f"ckp_{name}"] = figure
    return figures


def plot_ramsey(ds, fit_results):
    """Plot Ramsey contrast, phase and the matched CKP photon comparison."""
    figures = {}
    for name in ds.qubit.values:
        selected = ds.sel(qubit=name)
        result = fit_results[str(name)]
        figure, axes = plt.subplots(1, 3, figsize=(15, 4))
        reference_phase = selected.ramsey_phase_rad.isel(amp_factor=0)
        for ai, amplitude in enumerate(ds.amp_factor.values):
            color = plt.get_cmap("tab10")(ai)
            valid = selected.ramsey_fit_mask.isel(amp_factor=ai).astype(bool)
            axes[0].plot(
                ds.duration, selected.ramsey_contrast.isel(amp_factor=ai), ".-", label=f"amp={amplitude:g}", color=color
            )
            delta = selected.ramsey_phase_rad.isel(amp_factor=ai) - reference_phase
            axes[1].plot(ds.duration.where(valid), np.angle(np.exp(1j * delta)), ".", color=color)
            model = selected.coherence_ratio_fit_real.isel(amp_factor=ai) + 1j * selected.coherence_ratio_fit_imag.isel(
                amp_factor=ai
            )
            axes[1].plot(ds.duration.where(valid), np.angle(model), "-", color=color)
            axes[0].plot(ds.duration, selected.ramsey_contrast.isel(amp_factor=0) * abs(model), "--", color=color)
        axes[0].set(xlabel="drive duration (ns)", ylabel="Ramsey contrast", ylim=(0, 1.05))
        axes[0].legend(fontsize=8)
        axes[1].set(xlabel="drive duration (ns)", ylabel="phase relative to zero drive (rad)")
        axes[2].errorbar(
            ds.amp_factor, selected.photon_number, yerr=selected.photon_number_uncertainty, fmt="o", label="Ramsey"
        )
        axes[2].errorbar(
            ds.amp_factor,
            selected.ckp_photon_number,
            yerr=selected.ckp_photon_number_uncertainty,
            fmt="s-",
            label="CKP",
        )
        axes[2].set(xlabel="readout amplitude factor", ylabel="ground-state steady-state photon number")
        axes[2].legend()
        for axis in axes:
            axis.grid(alpha=0.25)
        figure.suptitle(f"{name}: matched Ramsey / CKP photon calibration; resolved={result['amp_success']}")
        figure.tight_layout()
        figures[f"photon_comparison_{name}"] = figure
    return figures

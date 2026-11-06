"""Fit CKP Stark ridges and finite-pulse Ramsey coherence.

CKP follows Sank et al., Phys. Rev. Applied 23, 024055 (2025), Eq. (7):
https://doi.org/10.1103/PhysRevApplied.23.024055
Frequencies are in MHz in fits; chi is half the signed e-minus-g separation.
"""

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit, least_squares
from quam.components.pulses import SquareReadoutPulse


def validate_drive(qubits, parameters):
    """Require the same calibrated square voltage for both protocols."""
    metadata = {}
    for q in qubits:
        pulse = q.resonator.operations[parameters.operation]
        if not isinstance(pulse, SquareReadoutPulse):
            raise ValueError("Use a SquareReadoutPulse for CKP and Ramsey (e.g. readout_square)")
        metadata[q.name] = {
            "operation": parameters.operation,
            "base_amplitude_v": float(pulse.amplitude),
            "readout_frequency_hz": float(q.resonator.RF_frequency),
            "drive_frequency_hz": float(q.resonator.RF_frequency + parameters.drive_detuning_mhz * 1e6),
            "qubit_frequency_hz": float(q.xy.RF_frequency),
        }
    return metadata


def _line(x, offset, height, center, width):
    return offset + height * np.exp(-0.5 * ((x - center) / width) ** 2)


def fit_ckp(ds, parameters):
    """Fit each probe line, then jointly fit the two state-dependent Stark ridges."""
    ds = ds.copy()
    dims = ("qubit", "amp_factor", "prepared_state", "resonator_detuning")
    shape = tuple(ds.sizes[d] for d in dims)
    centers, errors = np.full(shape, np.nan), np.full(shape, np.nan)
    fitted = np.full(shape, np.nan)
    photon = np.full((shape[0], shape[1]), np.nan)
    photon_error = np.full_like(photon, np.nan)
    x = np.asarray(ds.qubit_detuning, float)
    r = np.asarray(ds.resonator_detuning, float)
    amps = np.asarray(ds.amp_factor, float)
    results = {}
    for qi, name in enumerate(ds.qubit.values):
        for ai in range(len(amps)):
            for si in range(2):
                for ri in range(len(r)):
                    y = np.asarray(ds.state.isel(qubit=qi, amp_factor=ai, prepared_state=si, resonator_detuning=ri))
                    if si == 1:
                        y = 1 - y
                    height = np.ptp(y)
                    if not np.all(np.isfinite(y)) or height < parameters.min_line_contrast:
                        continue
                    try:
                        popt, pcov = curve_fit(
                            _line,
                            x,
                            y,
                            p0=(y.min(), height, x[np.argmax(y)], 0.5),
                            bounds=(
                                [-0.1, 0, x[0], parameters.qubit_step_mhz / 3],
                                [1.1, 1.5, x[-1], (x[-1] - x[0]) / 2],
                            ),
                            maxfev=10000,
                        )
                        sigma = np.sqrt(pcov[2, 2])
                        # Reject peaks truncated by the sweep and unresolved spectral lines.
                        if (
                            popt[1] >= parameters.min_line_contrast
                            and x[0] + popt[3] < popt[2] < x[-1] - popt[3]
                            and np.isfinite(sigma)
                        ):
                            centers[qi, ai, si, ri] = popt[2]
                            errors[qi, ai, si, ri] = max(sigma, parameters.qubit_step_mhz / 20)
                    except (ValueError, RuntimeError, np.linalg.LinAlgError):
                        continue
        observed = centers[qi]
        valid = np.isfinite(observed)
        result = {"success": False, "reason": "Insufficient resolved CKP probe lines"}
        results[str(name)] = result
        if any(np.count_nonzero(valid[ai, si]) < 5 for ai in range(len(amps)) for si in range(2)):
            continue
        baseline = np.nanmedian(observed[0], axis=1)

        # center, signed chi, common FWHM, zero-drive qubit offset, one n_peak per nonzero amplitude.
        def model(p):
            middle, chi, kappa, offset = p[:4]
            ng = np.r_[0.0, p[4:]]
            resonance = middle + np.array([-chi, chi])
            profile = 1 / (1 + (2 * (r[None, :] - resonance[:, None]) / kappa) ** 2)
            return offset + 2 * chi * ng[:, None, None] * profile[None, :, :]

        def residual(p):
            return ((model(p) - observed) / errors[qi])[valid]

        best = None
        for sign in [-1, 1]:
            chi_guess = sign * 0.5
            peak = np.maximum(0.1, np.nanmax(np.abs(observed[1:] - baseline.mean()), axis=(1, 2)))
            initial = [
                r[np.nanargmax(np.abs(observed[-1, 0] - baseline[0]))] + chi_guess,
                chi_guess,
                0.6,
                baseline.mean(),
                *(peak / (2 * abs(chi_guess))),
            ]
            lower = [
                r[0],
                -parameters.resonator_span_mhz,
                parameters.resonator_step_mhz / 2,
                x[0],
                *([0] * (len(amps) - 1)),
            ]
            upper = [r[-1], parameters.resonator_span_mhz, 2 * (r[-1] - r[0]), x[-1], *([1000] * (len(amps) - 1))]
            initial = np.clip(initial, np.array(lower) + 1e-6, np.array(upper) - 1e-6)
            try:
                fit = least_squares(residual, initial, bounds=(lower, upper), max_nfev=10000)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue
            if best is None or fit.cost < best.cost:
                best = fit
        if best is None:
            result["reason"] = "CKP joint fit did not converge"
            continue
        p = best.x
        dof = np.count_nonzero(valid) - len(p)
        covariance = np.linalg.pinv(best.jac.T @ best.jac) * (2 * best.cost / max(dof, 1))
        stderr = np.sqrt(np.maximum(0, np.diag(covariance)))
        middle, chi, kappa, offset = p[:4]
        detuning = parameters.drive_detuning_mhz - (middle - chi)
        factor = 1 / (1 + (2 * detuning / kappa) ** 2)
        photon[qi] = np.r_[0.0, p[4:]] * factor
        # Include covariance between linewidth, detuning, chi and peak occupation.
        gradients = np.zeros((len(amps), len(p)))
        for pi in range(len(p)):
            step = max(abs(p[pi]) * 1e-5, 1e-6)
            plus, minus = p.copy(), p.copy()
            plus[pi] += step
            minus[pi] -= step

            def occupation(v):
                return np.r_[0.0, v[4:]] / (1 + (2 * (parameters.drive_detuning_mhz - v[0] + v[1]) / v[2]) ** 2)

            gradients[:, pi] = (occupation(plus) - occupation(minus)) / (2 * step)
        photon_error[qi] = np.sqrt(np.maximum(0, np.einsum("ij,jk,ik->i", gradients, covariance, gradients)))
        fitted[qi] = model(p)
        resolved = (
            abs(chi) > 3 * stderr[1]
            and kappa > 3 * stderr[2]
            and all(r[0] + kappa / 2 < center < r[-1] - kappa / 2 for center in [middle - chi, middle + chi])
            and best.optimality < 1
            and np.all(best.active_mask == 0)
        )
        result.update(
            success=bool(best.success and resolved),
            reason="" if resolved else "CKP parameters unresolved or sweep too narrow",
            chi_mhz=float(chi),
            chi_uncertainty_mhz=float(stderr[1]),
            dispersive_shift_mhz=float(2 * chi),
            dispersive_shift_uncertainty_mhz=float(2 * stderr[1]),
            linewidth_mhz=float(kappa),
            linewidth_uncertainty_mhz=float(stderr[2]),
            kappa_rad_per_ns=float(2 * np.pi * kappa * 1e-3),
            resonator_ground_detuning_mhz=float(middle - chi),
            resonator_excited_detuning_mhz=float(middle + chi),
            qubit_zero_drive_detuning_mhz=float(offset),
            amp_factors=amps.tolist(),
            reduced_chi_square=float(2 * best.cost / max(dof, 1)),
            peak_photon_number=np.r_[0.0, p[4:]].tolist(),
            photon_flux_per_second=(np.r_[0.0, p[4:]] * 2 * np.pi * kappa * 1e6 / 4).tolist(),
            photon_number=photon[qi].tolist(),
            photon_number_uncertainty=photon_error[qi].tolist(),
            photon_definition="ground-state steady-state occupation at the comparison drive frequency",
        )
    ds["stark_ridge_mhz"] = (dims, centers)
    ds["stark_ridge_uncertainty_mhz"] = (dims, errors)
    ds["stark_ridge_fit_mhz"] = (dims, fitted)
    ds["photon_number"] = (("qubit", "amp_factor"), photon)
    ds["photon_number_uncertainty"] = (("qubit", "amp_factor"), photon_error)
    return ds, results


def ramsey_kernel(durations_ns, idle_ns, chi_mhz, linewidth_mhz, ground_detuning_mhz, drive_detuning_mhz):
    """Log QUA fringe phasor per steady-state ground photon, including ring-up and ring-down.

    Integrate alpha_e * conj(alpha_g) analytically for a square drive starting
    with an empty cavity. Re gives Stark phase; Im gives measurement dephasing.
    """
    chi = 2 * np.pi * chi_mhz * 1e-3
    kappa = 2 * np.pi * linewidth_mhz * 1e-3
    delta_g = 2 * np.pi * (ground_detuning_mhz - drive_detuning_mhz) * 1e-3
    bg, be = kappa / 2 + 1j * delta_g, kappa / 2 + 1j * (delta_g + 2 * chi)
    bgc = np.conj(bg)
    t = np.asarray(durations_ns, float)
    scale = abs(bg) ** 2 / (be * bgc)
    integral = scale * (
        t - (1 - np.exp(-be * t)) / be - (1 - np.exp(-bgc * t)) / bgc + (1 - np.exp(-(be + bgc) * t)) / (be + bgc)
    )
    beta_end = scale * (1 - np.exp(-be * t)) * (1 - np.exp(-bgc * t))
    integral += beta_end * (1 - np.exp(-(be + bgc) * idle_ns)) / (be + bgc)
    # Positive QUA frame rotation adds lab RF phase. For the fitted
    # cos(theta) - i*sin(theta) phasor, Stark phase is minus the rho_ge phase.
    return 2 * chi * (integral.imag - 1j * integral.real)


def fit_ramsey(ds, parameters, ckp_results):
    """Estimate photon number from complex Ramsey fringes, excluding lost contrast."""
    ds = ds.copy()
    frames = np.asarray(ds.frame)
    design = np.column_stack([np.ones(len(frames)), np.cos(2 * np.pi * frames), np.sin(2 * np.pi * frames)])
    dims = ("qubit", "amp_factor", "duration")
    signal = np.asarray(ds.state.transpose(*dims, "frame"))
    coefficients = np.linalg.lstsq(design, signal.reshape(-1, len(frames)).T, rcond=None)[0]
    phasor = (coefficients[1] - 1j * coefficients[2]).reshape(signal.shape[:-1])
    contrast = 2 * np.abs(phasor)
    fit_mask = np.zeros(phasor.shape, dtype=np.int8)
    photon = np.full(signal.shape[:2], np.nan)
    error = np.full_like(photon, np.nan)
    predicted = np.full(phasor.shape, np.nan + 1j * np.nan)
    results = {}
    for qi, name in enumerate(ds.qubit.values):
        calibration = ckp_results[str(name)]
        if not calibration["success"]:
            raise ValueError(f"CKP calibration failed for {name}")
        kernel = ramsey_kernel(
            np.asarray(ds.duration),
            parameters.post_drive_idle_ns,
            calibration["chi_mhz"],
            calibration["linewidth_mhz"],
            calibration["resonator_ground_detuning_mhz"],
            parameters.drive_detuning_mhz,
        )
        if np.exp(-calibration["kappa_rad_per_ns"] * parameters.post_drive_idle_ns) > 0.01:
            raise ValueError("Increase post_drive_idle_ns: residual photons would perturb the second x90 gate")
        fit_mask[qi, 0] = contrast[qi, 0] >= parameters.min_contrast
        reference = phasor[qi, 0]
        ratio = np.divide(phasor[qi], reference, out=np.full_like(phasor[qi], np.nan), where=np.abs(reference) > 0)
        photon[qi, 0], error[qi, 0] = 0, 0
        predicted[qi, 0] = 1
        success = [bool(np.count_nonzero(contrast[qi, 0] >= parameters.min_contrast) >= 5)]
        residual_rms = [0.0]
        for ai in range(1, ds.sizes["amp_factor"]):
            valid = (contrast[qi, ai] >= parameters.min_contrast) & (contrast[qi, 0] >= parameters.min_contrast)
            valid &= np.isfinite(ratio[ai])
            fit_mask[qi, ai] = valid
            if np.count_nonzero(valid) < 5:
                success.append(False)
                residual_rms.append(float("nan"))
                continue

            def residual(n):
                difference = (np.exp(kernel[valid] * n[0]) - ratio[ai, valid]) * np.abs(reference[valid])
                return np.r_[difference.real, difference.imag]

            candidates = [
                least_squares(residual, [n], bounds=(0, parameters.max_photon_number))
                for n in np.linspace(0.001, parameters.max_photon_number * 0.99, 25)
            ]
            best = min(candidates, key=lambda fit: fit.cost)
            n = best.x[0]
            variance = 2 * best.cost / max(2 * np.count_nonzero(valid) - 1, 1)
            sigma = np.sqrt(variance / max(float((best.jac.T @ best.jac)[0, 0]), 1e-20))
            ambiguous = any(
                abs(other.x[0] - n) > max(3 * sigma, 0.1) and other.cost <= best.cost * 1.1 + 1e-10
                for other in candidates
            )
            rms = float(np.sqrt(np.mean(residual([n]) ** 2)))
            residual_rms.append(rms)
            # Conservative shot-noise limit includes uncertainty in the reference fringe.
            noise_limit = 3 * np.sqrt(1 / (parameters.num_shots * len(frames)))
            resolved = (
                best.success
                and not ambiguous
                and n < parameters.max_photon_number * 0.99
                and sigma < max(n, 0.1)
                and rms <= noise_limit
            )
            success.append(bool(resolved))
            if resolved:
                photon[qi, ai], error[qi, ai] = n, sigma
                predicted[qi, ai] = np.exp(kernel * n)
        ckp_n = np.asarray(calibration["photon_number"])
        ckp_error = np.asarray(calibration["photon_number_uncertainty"])
        combined = np.sqrt(error[qi] ** 2 + ckp_error**2)
        difference = photon[qi] - ckp_n
        results[str(name)] = {
            "success": bool(all(success)),
            "amp_success": success,
            "fringe_residual_rms": residual_rms,
            "amp_factors": np.asarray(ds.amp_factor).tolist(),
            "photon_number": photon[qi].tolist(),
            "photon_number_uncertainty": error[qi].tolist(),
            "ckp_photon_number": ckp_n.tolist(),
            "ckp_photon_number_uncertainty": ckp_error.tolist(),
            "difference": difference.tolist(),
            "consistent_within_2sigma": [
                bool(ok and abs(delta) <= 2 * sigma) for ok, delta, sigma in zip(success, difference, combined)
            ],
            "uncertainty_note": "Ramsey statistical errors are conditional on CKP chi/kappa; methods share calibration uncertainty",
            "photon_definition": calibration["photon_definition"],
        }
    ds["ramsey_contrast"] = (dims, contrast)
    ds["ramsey_fit_mask"] = (dims, fit_mask)
    ds["ramsey_phase_rad"] = (dims, np.angle(phasor))
    ds["coherence_ratio_fit_real"] = (dims, predicted.real)
    ds["coherence_ratio_fit_imag"] = (dims, predicted.imag)
    ds["photon_number"] = (("qubit", "amp_factor"), photon)
    ds["photon_number_uncertainty"] = (("qubit", "amp_factor"), error)
    ds["ckp_photon_number"] = (("qubit", "amp_factor"), [ckp_results[str(q)]["photon_number"] for q in ds.qubit.values])
    ds["ckp_photon_number_uncertainty"] = (
        ("qubit", "amp_factor"),
        [ckp_results[str(q)]["photon_number_uncertainty"] for q in ds.qubit.values],
    )
    return ds, results

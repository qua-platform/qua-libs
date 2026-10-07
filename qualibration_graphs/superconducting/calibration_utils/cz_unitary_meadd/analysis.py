"""Analysis of the CZ unitary reconstruction (MEADD + Floquet).

Gate model (basis |b_L b_R>, L = control, R = target), up to a global phase:

    W = [[e^{iγ},  0,                0,                0          ],
         [0,       e^{-iζ} cosθ,    -i e^{iχ} sinθ,   0          ],
         [0,       -i e^{-iχ} sinθ,  e^{iζ} cosθ,     0          ],
         [0,       0,                0,                e^{-i(γ+ϕ)}]]

Methods: J. A. Gross et al., arXiv:2404.12550 (MEADD) for ϕ, θ and χ; F. Arute et al., arXiv:2010.07965
(Floquet characterization) for γ and ζ.

Virtual-Z corrections Z(a) on L and Z(b) on R after the gate map γ -> γ - (a + b) / 2, ζ -> ζ + (a - b) / 2 and
χ -> χ - (a - b) / 2, and leave ϕ and θ unchanged. With ϕ = π + δ, the process fidelity to CZ,
|Tr(CZ† W)|² / 16 = |e^{iγ} + 2 cosθ cosζ + e^{-i(γ+δ)}|² / 16, is largest at ζ = 0 and γ = -δ/2, where it equals
(1 + cos²θ + 2 cosθ cos(δ/2)) / 4. χ does not enter the fidelity.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

from calibration_utils.common_utils.confusion_matrix import recover_prepared_probs
from calibration_utils.cz_unitary_meadd.circuits import (
    EXP_FLOQUET,
    EXP_PHI,
    EXP_THETA_XX,
    EXP_THETA_YX,
    PREP_0P,
    PREP_1P,
    PREP_10,
    PREP_P0,
    PREP_P1,
    RO_XODD,
    RO_XX,
    RO_YODD,
    RO_YY,
    RO_ZZ,
    build_circuit_table,
)

OUTCOMES = ["00", "01", "10", "11"]
# A pair fails if the smallest odd-parity kept fraction of the theta circuits is below this value
KEPT_FRACTION_MIN = 0.5
# A pair fails if a phase line fit has a residual above this value, which means the unwrap slipped by 2π
UNWRAP_RESIDUAL_MAX = np.pi / 2

CZ = np.diag([1, 1, 1, -1])

_PAULIS = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
}
# Two-qubit Pauli basis, control (L) first
PAULI_LABELS = [a + b for a in _PAULIS for b in _PAULIS]
_TWO_QUBIT_PAULIS = [np.kron(_PAULIS[a], _PAULIS[b]) for a in _PAULIS for b in _PAULIS]


@dataclass
class FitResults:
    """Fitted CZ angles (radians) and quality flags for one qubit pair.

    Attributes:
        phi, theta, chi, gamma, zeta: Angles of the gate model, with standard errors in the *_err fields.
        chi_sign_margin: Confidence in the signs used for chi (0 to 1); below about 0.2, chi is unreliable.
        kept_fraction_min: Smallest odd-parity fraction in the theta circuits (low values mean leakage).
        max_unwrap_residual: Largest residual of the phase line fits.
        zeta_in_range: False if cos(Ω)/cos(θ) exceeds 1 by more than the measurement error allows.
        phi_floquet: Optional |11>-reference estimate of phi (known modulo π), None if not measured.
        fidelity: Process fidelity to CZ of the fitted gate, as measured (coherent errors only).
        corrected_fidelity: Process fidelity expected after applying the suggested phase_shift values.
        suggested_phase_shift_control/target: phase_shift values (units of 2π) that set zeta to 0 and gamma to
            the target chosen by the correction_target parameter.
        success: True if all quality checks passed.
    """

    phi: float
    phi_err: float
    theta: float
    theta_err: float
    chi: float
    chi_sign_margin: float
    gamma: float
    gamma_err: float
    zeta: float
    zeta_err: float
    kept_fraction_min: float
    max_unwrap_residual: float
    zeta_in_range: bool
    phi_floquet: Optional[float]
    phi_floquet_err: Optional[float]
    fidelity: float
    corrected_fidelity: float
    suggested_phase_shift_control: float
    suggested_phase_shift_target: float
    success: bool


def log_fitted_results(fit_results: Dict[str, Dict], log_callable=None):
    """Log the fitted angles and quality flags of every qubit pair."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, r in fit_results.items():
        lines = [
            f"Results for qubit pair {qp_name}: {'SUCCESS!' if r['success'] else 'FAIL!'}",
            f"\tphi - pi = {1e3 * (r['phi'] - np.pi):.2f} ± {1e3 * r['phi_err']:.2f} mrad",
            f"\ttheta    = {1e3 * r['theta']:.2f} ± {1e3 * r['theta_err']:.2f} mrad",
            f"\tchi      = {r['chi']:.3f} rad (sign margin {r['chi_sign_margin']:.2f})",
            f"\tgamma    = {1e3 * r['gamma']:.2f} ± {1e3 * r['gamma_err']:.2f} mrad",
            f"\tzeta     = {1e3 * r['zeta']:.2f} ± {1e3 * r['zeta_err']:.2f} mrad",
        ]
        if r["phi_floquet"] is not None:
            lines.append(
                f"\tphi - pi (Floquet) = {1e3 * (r['phi_floquet'] - np.pi):.2f} ± {1e3 * r['phi_floquet_err']:.2f} mrad"
            )
        lines += [
            f"\tprocess fidelity = {r['fidelity']:.5f} ({r['corrected_fidelity']:.5f} after suggested corrections)",
            f"\tmin kept fraction = {r['kept_fraction_min']:.3f} (fail below {KEPT_FRACTION_MIN})",
            f"\tmax unwrap residual = {r['max_unwrap_residual']:.3f} rad (fail above {UNWRAP_RESIDUAL_MAX:.3f})",
            f"\tzeta in range = {r['zeta_in_range']}",
            f"\tsuggested phase_shift_control = {r['suggested_phase_shift_control']:.5f}",
            f"\tsuggested phase_shift_target  = {r['suggested_phase_shift_target']:.5f}",
        ]
        log_callable("\n".join(lines))


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Label the circuits, store the CZ corrections used, and compute the joint outcome probabilities.

    The QUA program saves the averages of state_control, state_target and state_both (= control AND target).
    These give the four joint probabilities P(b_L b_R), stored as ``probs`` with an ``outcome`` dimension,
    corrected with the pair's confusion matrix if ``use_readout_mitigation`` is set.
    """
    params = node.parameters
    qubit_pairs = node.namespace["qubit_pairs"]
    rows = build_circuit_table(
        params.max_cz_meadd, params.step_cz_meadd, params.max_cz_floquet, params.include_floquet_phi
    )
    coord_names = {"experiment": "exp", "preparation": "prep", "readout": "ro", "ncz": "ncz"}
    ds = ds.assign_coords({name: ("circuit", [r[key] for r in rows]) for name, key in coord_names.items()})

    # The chi sign needs the virtual-Z corrections that were active during the run
    if "phase_shift_control" not in ds:
        macros = [qp.macros[params.operation] for qp in qubit_pairs]
        ds = ds.assign(
            phase_shift_control=("qubit_pair", [m.phase_shift_control for m in macros]),
            phase_shift_target=("qubit_pair", [m.phase_shift_target for m in macros]),
        )

    p11 = ds.state_both
    p10 = ds.state_control - p11
    p01 = ds.state_target - p11
    p00 = 1 - p01 - p10 - p11
    probs = xr.concat([p00, p01, p10, p11], dim=xr.DataArray(OUTCOMES, dims="outcome"))
    probs = probs.transpose("qubit_pair", "circuit", "outcome")

    if params.use_readout_mitigation:
        for i, qp in enumerate(qubit_pairs):
            conf = np.asarray(qp.confusion)
            probs[i] = np.array([recover_prepared_probs(conf, p) for p in probs[i].values])

    probs.attrs = {"long_name": "joint outcome probability", "units": ""}
    return ds.assign(probs=probs)


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """Extract the five CZ angles of every qubit pair.

    Returns a dataset with the intermediate curves used for plotting (dimensions ``cz_pairs`` for the MEADD
    circuits and ``ncz_floquet`` for the Floquet circuits) and a dictionary of FitResults per pair.
    """
    rows = [
        dict(exp=int(e), prep=int(p), ro=int(r), ncz=int(n))
        for e, p, r, n in zip(ds.experiment.values, ds.preparation.values, ds.readout.values, ds.ncz.values)
    ]
    s = -1 if node.parameters.invert_frame_sign else 1
    fits, fit_results = [], {}
    for qp in node.namespace["qubit_pairs"]:
        ds_qp = ds.sel(qubit_pair=qp.name)
        x_control = float(ds_qp.phase_shift_control)
        x_target = float(ds_qp.phase_shift_target)
        # zeta of the gate without virtual-Z corrections, minus zeta of the gate as used
        zeta_raw_offset = -s * np.pi * (x_control - x_target)

        values, curves = analyze_pair(rows, ds_qp.probs.values, zeta_raw_offset)
        success = bool(
            all(np.isfinite(v) for v in values.values() if v is not None)
            and values["kept_fraction_min"] >= KEPT_FRACTION_MIN
            and values["max_unwrap_residual"] <= UNWRAP_RESIDUAL_MAX
            and values["zeta_in_range"]
        )
        # Z(a) on L and Z(b) on R after the gate, with a + b = 2(γ - γ_target) and a - b = -2ζ
        gamma, zeta = values["gamma"], values["zeta"]
        gamma_target = optimal_gamma(values["phi"]) if node.parameters.correction_target == "max_fidelity" else 0.0
        a = gamma - gamma_target - zeta
        b = gamma - gamma_target + zeta
        fit_results[qp.name] = FitResults(
            **values,
            fidelity=process_fidelity(gate_unitary(values["phi"], values["theta"], values["chi"], gamma, zeta)),
            corrected_fidelity=process_fidelity(
                gate_unitary(values["phi"], values["theta"], values["chi"] + zeta, gamma_target, 0.0)
            ),
            suggested_phase_shift_control=float((x_control + s * a / (2 * np.pi)) % 1),
            suggested_phase_shift_target=float((x_target + s * b / (2 * np.pi)) % 1),
            success=success,
        )
        fits.append(curves.assign(success=success))

    ds_fit = xr.concat(fits, dim=xr.DataArray([qp.name for qp in node.namespace["qubit_pairs"]], dims="qubit_pair"))
    return ds_fit, fit_results


# ---------------------------------------------------------------------------------------------------------------
# Pure-numpy analysis of one pair. P has shape (n_circuits, 4) with columns [P00, P01, P10, P11].
# ---------------------------------------------------------------------------------------------------------------


def analyze_pair(rows: List[Dict[str, int]], P: np.ndarray, zeta_raw_offset: float) -> Tuple[Dict, xr.Dataset]:
    """Run the MEADD-ϕ, MEADD-θ and Floquet analyses in order (ζ needs θ, and the χ sign needs ζ)."""
    lookup = {(r["exp"], r["prep"], r["ro"], r["ncz"]): i for i, r in enumerate(rows)}
    max_residual = []

    # MEADD-ϕ: arg det M_n = 2 n ϕ (mod 2π), with n = ncz / 2 CZ pairs
    ncz_meadd, M_phi = _odd_block_matrices(rows, P, lookup, EXP_PHI)
    n = ncz_meadd / 2
    det_phi = _unwrapped_det_phase(M_phi)
    slope, slope_err, det_phi_fit = _line_fit(n, det_phi)
    phi, phi_err = np.pi + slope / 2, slope_err / 2
    max_residual.append(np.max(np.abs(det_phi - det_phi_fit)))

    # MEADD-θ: signed rotation angle away from the |10> pole, for both dynamical-decoupling flavours
    theta_proj = {}
    for exp in (EXP_THETA_XX, EXP_THETA_YX):
        x, y, z_pole, kept = _odd_bloch_vector(P, lookup, exp, ncz_meadd)
        c = np.sum(n * (x + 1j * y))  # rotation direction from all depths
        equatorial = np.real((x + 1j * y) * np.exp(-1j * np.angle(c)))
        alpha = np.arctan2(equatorial, z_pole)
        slope, slope_err, alpha_fit = _line_fit(n, alpha)
        theta_proj[exp] = dict(
            mag=slope / 4, err=slope_err / 4, c=c, x=x, y=y, z=z_pole, alpha=alpha, alpha_fit=alpha_fit, kept=kept
        )
    a, a_err = theta_proj[EXP_THETA_XX]["mag"], theta_proj[EXP_THETA_XX]["err"]  # |θ cos χ|
    b, b_err = theta_proj[EXP_THETA_YX]["mag"], theta_proj[EXP_THETA_YX]["err"]  # |θ sin χ|
    theta, theta_err = np.hypot(a, b), np.hypot(a_err, b_err)

    # Floquet: det M_n = e^{-2inγ}; the |10> eigenphase of the odd block grows as n Ω, with cos Ω = cos θ cos ζ
    ncz_floquet, M_floquet = _odd_block_matrices(rows, P, lookup, EXP_FLOQUET)
    det_floquet = _unwrapped_det_phase(M_floquet)
    slope_det, slope_det_err, det_floquet_fit = _line_fit(ncz_floquet, det_floquet)
    gamma, gamma_err = -slope_det / 2, slope_det_err / 2
    eigphase = _unwrapped_eigenphase(ncz_floquet, M_floquet, gamma)
    omega, omega_err, eigphase_fit = _line_fit(ncz_floquet, eigphase)
    max_residual += [np.max(np.abs(det_floquet - det_floquet_fit)), np.max(np.abs(eigphase - eigphase_fit))]

    zeta, zeta_in_range = _zeta_from_omega(omega, omega_err, theta, theta_err)

    # χ: the θ circuits run without virtual-Z corrections, so their rotation axis is tilted by -ζ_bare.
    # Undo the tilt, then compare with +Y_odd (X⊗X) and +X_odd (Y⊗X) to get the signs of θcosχ and θsinχ.
    zeta_bare = zeta + zeta_raw_offset
    axis_xx = theta_proj[EXP_THETA_XX]["c"] * np.exp(1j * zeta_bare) * (-1j)
    axis_yx = theta_proj[EXP_THETA_YX]["c"] * np.exp(1j * zeta_bare)
    chi = np.arctan2(np.sign(axis_yx.real) * b, np.sign(axis_xx.real) * a)
    chi_sign_margin = min(abs(np.cos(np.angle(axis_xx))), abs(np.cos(np.angle(axis_yx))))

    values = dict(
        phi=float(phi),
        phi_err=float(phi_err),
        theta=float(theta),
        theta_err=float(theta_err),
        chi=float(chi),
        chi_sign_margin=float(chi_sign_margin),
        gamma=float(gamma),
        gamma_err=float(gamma_err),
        zeta=float(zeta),
        zeta_err=float(omega_err),
        kept_fraction_min=float(min(theta_proj[e]["kept"].min() for e in theta_proj)),
        max_unwrap_residual=float(np.max(max_residual)),
        zeta_in_range=bool(zeta_in_range),
        phi_floquet=None,
        phi_floquet_err=None,
    )

    # Optional |11>-reference cross-check: det N_n = e^{-2in(γ+ϕ)}, so ϕ is known only modulo π
    if (EXP_FLOQUET, PREP_P1, RO_XX, 0) in lookup:
        _, N_floquet = _odd_block_matrices(rows, P, lookup, EXP_FLOQUET, preps=(PREP_P1, PREP_1P))
        slope_N, slope_N_err, _ = _line_fit(ncz_floquet, _unwrapped_det_phase(N_floquet))
        delta = np.mod(-(slope_N - slope_det) / 2 + np.pi / 2, np.pi) - np.pi / 2
        values["phi_floquet"] = float(np.pi + delta)
        values["phi_floquet_err"] = float(np.hypot(slope_N_err, slope_det_err) / 2)

    meadd = {"cz_pairs": n}
    floquet = {"ncz_floquet": ncz_floquet}
    curves = xr.Dataset(
        {
            # Diagonal of M_n: target qubit for prep 0P, control qubit for prep P0
            "phi_target_x": ("cz_pairs", M_phi[:, 0, 0].real),
            "phi_target_y": ("cz_pairs", M_phi[:, 0, 0].imag),
            "phi_control_x": ("cz_pairs", M_phi[:, 1, 1].real),
            "phi_control_y": ("cz_pairs", M_phi[:, 1, 1].imag),
            "phi_det_phase": ("cz_pairs", det_phi),
            "phi_det_phase_fit": ("cz_pairs", det_phi_fit),
            **{
                f"theta_{name}_{key}": ("cz_pairs", theta_proj[exp][key])
                for exp, name in ((EXP_THETA_XX, "xx"), (EXP_THETA_YX, "yx"))
                for key in ("x", "y", "z", "alpha", "alpha_fit", "kept")
            },
            # Control qubit <X>, <Y> for prep P0, and the off-diagonal entries of M_n
            "floquet_control_x": ("ncz_floquet", M_floquet[:, 1, 1].real),
            "floquet_control_y": ("ncz_floquet", M_floquet[:, 1, 1].imag),
            "floquet_offdiag_01": ("ncz_floquet", np.abs(M_floquet[:, 0, 1])),
            "floquet_offdiag_10": ("ncz_floquet", np.abs(M_floquet[:, 1, 0])),
            "floquet_det_phase": ("ncz_floquet", det_floquet),
            "floquet_det_phase_fit": ("ncz_floquet", det_floquet_fit),
            "floquet_eigphase": ("ncz_floquet", eigphase),
            "floquet_eigphase_fit": ("ncz_floquet", eigphase_fit),
        },
        coords={**meadd, **floquet},
    )
    return values, curves


def gate_unitary(phi: float, theta: float, chi: float, gamma: float, zeta: float) -> np.ndarray:
    """The 4x4 gate model W of the module docstring for the given angles, in the basis |b_L b_R>."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [np.exp(1j * gamma), 0, 0, 0],
            [0, np.exp(-1j * zeta) * c, -1j * np.exp(1j * chi) * s, 0],
            [0, -1j * np.exp(-1j * chi) * s, np.exp(1j * zeta) * c, 0],
            [0, 0, 0, np.exp(-1j * (gamma + phi))],
        ]
    )


def process_fidelity(unitary: np.ndarray) -> float:
    """Process fidelity |Tr(CZ† U)|² / 16 of a two-qubit unitary to the ideal CZ."""
    return float(abs(np.trace(CZ.conj().T @ unitary)) ** 2 / 16)


def optimal_gamma(phi: float) -> float:
    """Value of γ that maximizes the process fidelity to CZ for a conditional phase ϕ: it splits ϕ - π over |00>
    and |11>."""
    return -(phi - np.pi) / 2


def pauli_chi_matrix(unitary: np.ndarray) -> np.ndarray:
    """Process matrix χ of a two-qubit unitary in the Pauli basis, ordered as PAULI_LABELS.

    With U = Σ_m c_m P_m and c_m = Tr(P_m U) / 4, the process is ρ -> Σ_mn χ_mn P_m ρ P_n with χ_mn = c_m c_n*.
    χ does not depend on the global phase of U, and its trace is 1.
    """
    coeffs = np.array([np.trace(P @ unitary) / 4 for P in _TWO_QUBIT_PAULIS])
    return np.outer(coeffs, coeffs.conj())


def _z_expectations(p: np.ndarray) -> Tuple[float, float]:
    """<Z_L>, <Z_R> from the joint probabilities [P00, P01, P10, P11]."""
    return (p[0] + p[1]) - (p[2] + p[3]), (p[0] + p[2]) - (p[1] + p[3])


def _odd_block_matrices(rows, P, lookup, exp, preps=(PREP_0P, PREP_P0)) -> Tuple[np.ndarray, np.ndarray]:
    """Build the 2x2 matrix M_n of odd-parity matrix elements for every depth of a sub-experiment.

    With one qubit prepared in |+> and the other in |0>, <X> + i<Y> of each qubit is one matrix element of the
    odd-parity block times the reference phase of |00> (preps 0P, P0) or |11> (preps P1, 1P).
    Returns the depths (number of CZ gates) and an array of shape (n_depths, 2, 2).
    """
    depths = np.array(sorted({r["ncz"] for r in rows if r["exp"] == exp and r["prep"] == preps[0]}))
    matrices = []
    for ncz in depths:
        xy = {}
        for prep in preps:
            zl_x, zr_x = _z_expectations(P[lookup[(exp, prep, RO_XX, ncz)]])
            zl_y, zr_y = _z_expectations(P[lookup[(exp, prep, RO_YY, ncz)]])
            xy[prep] = dict(L=zl_x + 1j * zl_y, R=zr_x + 1j * zr_y)
        a, b = preps
        if preps == (PREP_0P, PREP_P0):  # |00> reference: row <01| from R, row <10| from L
            matrices.append([[xy[a]["R"], xy[b]["R"]], [xy[a]["L"], xy[b]["L"]]])
        else:  # |11> reference: row <01| from L, row <10| from R
            matrices.append([[xy[a]["L"], xy[b]["L"]], [xy[a]["R"], xy[b]["R"]]])
    return depths, np.array(matrices)


def _odd_bloch_vector(P, lookup, exp, depths):
    """Parity-postselected odd-parity Bloch vector for the |10> start.

    Returns x, y, the z component along the start pole, and the kept (odd-parity) fraction for each depth.
    """
    pz = np.array([P[lookup[(exp, PREP_10, RO_ZZ, ncz)]] for ncz in depths])
    px = np.array([P[lookup[(exp, PREP_10, RO_XODD, ncz)]] for ncz in depths])
    py = np.array([P[lookup[(exp, PREP_10, RO_YODD, ncz)]] for ncz in depths])
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (pz[:, 1] - pz[:, 2]) / (pz[:, 1] + pz[:, 2])  # Z_odd = +1 for |01>
        x = (px[:, 1] - px[:, 3]) / (px[:, 1] + px[:, 3])  # keep R = 1 (outcomes 01, 11); L = 0 means +1
        y = (py[:, 1] - py[:, 3]) / (py[:, 1] + py[:, 3])
    return x, y, -z, pz[:, 1] + pz[:, 2]  # |10> starts at Z_odd = -1


def _unwrapped_det_phase(matrices: np.ndarray) -> np.ndarray:
    return np.unwrap(np.angle(np.linalg.det(matrices)))


def _unwrapped_eigenphase(depths, matrices, gamma) -> np.ndarray:
    """Eigenphase of the eigenvector closest to |10>, after removing the reference phase e^{-inγ}."""
    phases = []
    for ncz, M in zip(depths, matrices):
        if not np.all(np.isfinite(M)):
            phases.append(np.nan)
            continue
        U, _, Vh = np.linalg.svd(M)
        unitary = (U @ Vh) * np.exp(1j * ncz * gamma)
        eigvals, eigvecs = np.linalg.eig(unitary)
        phases.append(np.angle(eigvals[np.argmax(np.abs(eigvecs[1, :]))]))
    return np.unwrap(phases)


def _zeta_from_omega(omega, omega_err, theta, theta_err) -> Tuple[float, bool]:
    """ζ = sgn(Ω) arccos(cos Ω / cos θ).

    When ζ ≈ 0, Ω ≈ θ and noise can push cos Ω / cos θ slightly above 1. Then ζ is set to 0, and it is flagged
    out of range only if |Ω| is below θ by more than twice the combined error.
    """
    ratio = np.cos(omega) / np.cos(theta)
    if abs(ratio) <= 1:
        return np.sign(omega) * np.arccos(ratio), True
    return 0.0, theta - abs(omega) <= 2 * np.hypot(omega_err, theta_err)


def _line_fit(x, y) -> Tuple[float, float, np.ndarray]:
    """Least-squares line. Returns the slope, its standard error and the fitted values (NaN if y is not finite)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if not np.all(np.isfinite(y)):
        return np.nan, np.nan, np.full_like(y, np.nan)
    (slope, intercept), cov = np.polyfit(x, y, 1, cov=True)
    return slope, np.sqrt(cov[0, 0]), slope * x + intercept

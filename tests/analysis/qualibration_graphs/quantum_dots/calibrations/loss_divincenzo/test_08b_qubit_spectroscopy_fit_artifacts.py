"""Visual artifact tests for 08b_qubit_spectroscopy Lorentzian fitting.

Generates synthetic spectroscopy data (single peak, single dip, two peaks),
runs the full fit_raw_data pipeline, and saves plots + fit summaries to
``tests/analysis/artifacts/08b_qubit_spectroscopy/``.

Run with::

    pytest tests/analysis/.../test_08b_qubit_spectroscopy_fit_artifacts.py -s

Inspect the generated PNGs under ``tests/analysis/artifacts/08b_qubit_spectroscopy/``.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from calibration_utils.qubit_spectroscopy_parity_diff.analysis import fit_raw_data

# ── Paths ─────────────────────────────────────────────────────────────────────

ARTIFACTS_DIR = (
    Path(__file__).resolve().parents[4] / "artifacts" / "08b_qubit_spectroscopy"
)

# ── Constants ─────────────────────────────────────────────────────────────────

RF_FREQ = 10.0e9
SPAN_MHZ = 20.0
STEP_MHZ = 0.25
DETUNINGS = np.arange(-SPAN_MHZ / 2 * 1e6, SPAN_MHZ / 2 * 1e6, STEP_MHZ * 1e6)
NOISE_STD = 0.012


# ── Helpers ───────────────────────────────────────────────────────────────────


def _lorentzian(x, center, fwhm, amp):
    hwhm = fwhm / 2.0
    return amp / (1.0 + ((x - center) / hwhm) ** 2)


def _make_node(qubit_names):
    qubits = [
        SimpleNamespace(name=n, xy=SimpleNamespace(RF_frequency=RF_FREQ))
        for n in qubit_names
    ]
    return SimpleNamespace(
        parameters=SimpleNamespace(
            analysis_signal="E_p2_given_p1_0",
            frequency_span_in_mhz=SPAN_MHZ,
        ),
        namespace={"qubits": qubits},
    )


def _make_ds(signals: dict[str, np.ndarray], sig="E_p2_given_p1_0") -> xr.Dataset:
    return xr.Dataset(
        {f"{sig}_{q}": xr.DataArray(y, dims="detuning") for q, y in signals.items()},
        coords={
            "detuning": xr.DataArray(
                DETUNINGS, dims="detuning",
                attrs={"long_name": "drive frequency", "units": "Hz"},
            )
        },
    )


def _plot_fit(
    ax, detunings, y_data, fit_curve, fit_result, title,
):
    det_mhz = detunings / 1e6
    ax.plot(det_mhz, y_data, "b.", ms=2, alpha=0.6, label="data")
    ax.plot(det_mhz, fit_curve, "r-", lw=1.5, label="fit")
    ax.axvline(
        fit_result["relative_freq"] / 1e6, color="green", ls="--", alpha=0.7,
        label=f"qubit = {fit_result['relative_freq']/1e6:.2f} MHz",
    )
    if fit_result.get("readout_qubit_relative_freq") is not None:
        ax.axvline(
            fit_result["readout_qubit_relative_freq"] / 1e6,
            color="orange", ls="--", alpha=0.7,
            label=f"readout = {fit_result['readout_qubit_relative_freq']/1e6:.2f} MHz",
        )
    ax.set_xlabel("Detuning [MHz]")
    ax.set_ylabel("Signal")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)


def _save(fig, name):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Scenario definitions ──────────────────────────────────────────────────────

SCENARIOS = {
    "single_peak": {
        "q1": {"center": 3e6, "fwhm": 2e6, "amp": 0.4, "offset": 0.0},
        "q2": {"center": -4e6, "fwhm": 1.5e6, "amp": 0.3, "offset": 0.0},
    },
    "single_dip": {
        "q1": {"center": 2e6, "fwhm": 2e6, "amp": -0.35, "offset": 0.5},
        "q2": {"center": -3e6, "fwhm": 1.5e6, "amp": -0.4, "offset": 0.5},
    },
    "two_peaks": {
        "q1": {
            "peaks": [
                {"center": 1e6, "fwhm": 1e6, "amp": 0.4},
                {"center": 7e6, "fwhm": 1e6, "amp": 0.35},
            ],
            "offset": 0.0,
        },
        "q2": {
            "peaks": [
                {"center": -2e6, "fwhm": 1.2e6, "amp": 0.38},
                {"center": -8e6, "fwhm": 1e6, "amp": 0.32},
            ],
            "offset": 0.0,
        },
    },
}


def _generate_signal(spec, rng):
    n = len(DETUNINGS)
    offset = spec.get("offset", 0.0)
    y = np.full(n, offset)
    if "peaks" in spec:
        for pk in spec["peaks"]:
            y += _lorentzian(DETUNINGS, pk["center"], pk["fwhm"], pk["amp"])
    else:
        y += _lorentzian(DETUNINGS, spec["center"], spec["fwhm"], spec["amp"])
    y += rng.normal(0, NOISE_STD, n)
    return y


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.fixture(params=["single_peak", "single_dip", "two_peaks"])
def scenario(request):
    return request.param


def test_fit_and_plot_artifact(scenario):
    """Run fit_raw_data on synthetic data and save a plot artifact for review."""
    rng = np.random.default_rng(42)
    spec = SCENARIOS[scenario]
    qubit_names = sorted(spec.keys())

    signals = {q: _generate_signal(spec[q], rng) for q in qubit_names}
    ds = _make_ds(signals)
    node = _make_node(qubit_names)

    ds_fit, fit_results = fit_raw_data(ds, node)
    fit_dicts = {k: asdict(v) for k, v in fit_results.items()}

    n_qubits = len(qubit_names)
    fig, axes = plt.subplots(1, n_qubits, figsize=(7 * n_qubits, 5), squeeze=False)

    for i, qname in enumerate(qubit_names):
        r = fit_dicts[qname]
        y_data = signals[qname]
        curve = ds_fit.fit_curve.sel(qubit=qname).values

        subtitle = (
            f"{qname}: n_peaks={r['num_peaks']}, "
            f"FWHM={r['fwhm']/1e6:.2f} MHz, "
            f"freq={r['frequency']/1e9:.4f} GHz"
        )
        _plot_fit(axes[0, i], DETUNINGS, y_data, curve, r, subtitle)

    fig.suptitle(f"08b qubit spectroscopy — {scenario}", fontsize=13)
    fig.tight_layout()
    artifact_path = _save(fig, f"{scenario}.png")

    summary_lines = [f"# {scenario}\n"]
    for qname in qubit_names:
        r = fit_dicts[qname]
        summary_lines.append(f"## {qname}")
        summary_lines.append(f"- Peaks found: {r['num_peaks']}")
        summary_lines.append(f"- Frequency: {r['frequency']/1e9:.6f} GHz")
        summary_lines.append(f"- Relative freq: {r['relative_freq']/1e6:.3f} MHz")
        summary_lines.append(f"- FWHM: {r['fwhm']/1e6:.3f} MHz")
        summary_lines.append(f"- Success: {r['success']}")
        if r.get("readout_qubit_frequency") is not None:
            summary_lines.append(
                f"- Readout qubit freq: {r['readout_qubit_frequency']/1e9:.6f} GHz"
            )
        summary_lines.append("")

    readme_path = ARTIFACTS_DIR / f"{scenario}_summary.md"
    readme_path.write_text("\n".join(summary_lines), encoding="utf-8")

    assert artifact_path.exists(), f"Artifact not saved: {artifact_path}"
    assert readme_path.exists(), f"Summary not saved: {readme_path}"

    for qname in qubit_names:
        r = fit_dicts[qname]
        assert r["success"], f"{qname} fit should succeed: {r}"

        s = SCENARIOS[scenario][qname]
        if "peaks" in s:
            primary_center = min(s["peaks"], key=lambda p: abs(p["center"]))["center"]
        else:
            primary_center = s["center"]
        assert abs(r["relative_freq"] - primary_center) < 1e6, (
            f"{qname}: expected ~{primary_center/1e6:.1f} MHz, "
            f"got {r['relative_freq']/1e6:.2f} MHz"
        )

    if scenario == "two_peaks":
        for qname in qubit_names:
            r = fit_dicts[qname]
            assert r["num_peaks"] == 2, f"{qname} should detect 2 peaks"
            assert r["readout_qubit_frequency"] is not None

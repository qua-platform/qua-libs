"""Tests for the complex drive-line crosstalk fit (27_driveline_crosstalk_matrix)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_utils.driveline_crosstalk_matrix.analysis import (  # noqa: E402
    FRAMES,
    PHASE_MODE,
    POPULATION_MODE,
    fit_bowl,
    fit_raw_data,
    population,
    ramsey_phase,
)
from calibration_utils.driveline_crosstalk_matrix.sequence import (  # noqa: E402
    add_pair_info,
    comb_factor,
    grid_points,
    near_second_transition,
    pair_mode,
    pulse_rotation,
    stark_factor,
    train_pulses,
)

NODE = SimpleNamespace(parameters=None)  # fit_raw_data only reads the dataset
C_TRUE = 0.03 * np.exp(1j * np.deg2rad(120))


def _dataset(mode, c=C_TRUE, center=0j, r_max=0.1, scale=60.0, offset=0.4, shots=400, seed=0, readout=(0.0, 1.0)):
    """Synthetic raw dataset of one pair: Stark phase (or P_e) = scale * |c + z|^2 + offset."""
    rng = np.random.default_rng(seed)
    z = np.append(grid_points(center, r_max, [0.5, 1.0], 8), 0j)
    on = np.append(np.ones(len(z) - 1, dtype=int), 0)
    signal = scale * np.abs(np.where(on, c, 0) + z) ** 2
    pe = np.full((len(z), len(FRAMES)), np.nan)
    if mode == PHASE_MODE:
        phi = signal + offset
        p = 0.5 * (1 + 0.9 * np.cos(phi[:, None] - 2 * np.pi * np.asarray(FRAMES)[None, :]))
        pe[:] = rng.binomial(shots, p) / shots
    else:
        # readout = (a, b): measured P_e = a + b * true P_e (offset, contrast, inversion)
        p = np.clip(readout[0] + readout[1] * np.clip(0.02 + signal, 0, 1), 0, 1)
        pe[:] = rng.binomial(shots, np.repeat(p[:, None], len(FRAMES), axis=1)) / shots
    ds = xr.Dataset(
        {
            "pe": (("qubit_pair", "point", "frame"), pe[None]),
            "z_re": (("qubit_pair", "point"), z.real[None]),
            "z_im": (("qubit_pair", "point"), z.imag[None]),
            "source_on": (("qubit_pair", "point"), on[None]),
            "center_re": ("qubit_pair", [center.real]),
            "center_im": ("qubit_pair", [center.imag]),
            "source_frequency": ("qubit_pair", [5.5e9]),
        },
        coords={
            "qubit_pair": ["qA-qB"],
            "point": np.arange(len(z)),
            "frame": list(FRAMES),
            "source_qubit": ("qubit_pair", ["qA"]),
            "victim_qubit": ("qubit_pair", ["qB"]),
            "mode": ("qubit_pair", [mode]),
        },
    )
    ds.attrs.update(num_shots=shots, r_max=r_max, operation="x180")
    return ds


def test_grid_points_centre_and_rings():
    z = grid_points(0.1 + 0.2j, 0.2, [0.5, 1.0], 8)
    assert len(z) == 17
    assert z[0] == pytest.approx(0.1 + 0.2j)
    assert np.allclose(np.abs(z[1:9] - z[0]), 0.1)
    assert np.allclose(np.abs(z[9:] - z[0]), 0.2)


def test_fit_bowl_noiseless_recovers_c():
    z = grid_points(0j, 0.1, [0.5, 1.0], 8)
    obs = 37.0 * np.abs(C_TRUE + z) ** 2 - 1.3
    fit = fit_bowl(z, obs, np.full(len(z), 1e-3))
    assert fit["c"] == pytest.approx(C_TRUE, abs=1e-9)
    assert fit["a"] == pytest.approx(37.0)


def test_fit_bowl_sign_of_curvature_does_not_matter():
    z = grid_points(0j, 0.1, [0.5, 1.0], 8)
    obs = -37.0 * np.abs(C_TRUE + z) ** 2
    assert fit_bowl(z, obs, np.ones(len(z)))["c"] == pytest.approx(C_TRUE, abs=1e-9)


def test_ramsey_phase_extracts_phase_without_wrapping():
    phi = np.array([3.0, 3.1, -3.1, 2.9])  # around +/- pi
    p = 0.5 * (1 + np.cos(phi[:, None] - 2 * np.pi * np.asarray(FRAMES)[None, :]))
    phase, sigma, visibility = ramsey_phase(p, 1000)
    unwrapped = np.unwrap(phi)
    assert np.allclose(np.diff(phase), np.diff(unwrapped), atol=1e-9) or np.allclose(
        np.diff(phase), -np.diff(unwrapped), atol=1e-9
    )
    assert np.allclose(visibility, 1.0)


@pytest.mark.parametrize("mode", [PHASE_MODE, POPULATION_MODE])
def test_analysis_recovers_c_with_shot_noise(mode):
    scale = 60.0 if mode == PHASE_MODE else 15.0
    ds = _dataset(mode, scale=scale)
    _, fits = fit_raw_data(ds, NODE)
    fit = fits["qA-qB"]
    c = complex(fit.c_re, fit.c_im)
    assert fit.success, fit.reason
    assert abs(c - C_TRUE) < 4 * fit.c_err + 2e-3


def test_compensated_grid_gives_small_residual():
    estimate = C_TRUE * 0.95 * np.exp(1j * 0.05)
    ds = _dataset(PHASE_MODE, center=-estimate, r_max=0.06, scale=150.0)
    _, fits = fit_raw_data(ds, NODE)
    fit = fits["qA-qB"]
    assert abs(complex(fit.c_re, fit.c_im) - C_TRUE) < 4 * fit.c_err + 2e-3


def test_null_outside_grid_fails():
    ds = _dataset(PHASE_MODE, c=0.3, r_max=0.05, scale=5.0)
    _, fits = fit_raw_data(ds, NODE)
    assert not fits["qA-qB"].success
    assert "beyond 1.5 r_max" in fits["qA-qB"].reason


def test_population_not_small_angle_fails():
    ds = _dataset(POPULATION_MODE, scale=80.0)
    _, fits = fit_raw_data(ds, NODE)
    assert not fits["qA-qB"].success
    assert "small-angle" in fits["qA-qB"].reason


def test_population_readout_offset_and_inversion_pass():
    """|0> read as 0.57 and |1> as 0.07 (as qA2 on 2026-11-24): the check is relative to the reference point, and the
    fit is invariant to an affine readout."""
    ds = _dataset(POPULATION_MODE, scale=15.0, readout=(0.57, -0.5), shots=1600)
    _, fits = fit_raw_data(ds, NODE)
    fit = fits["qA-qB"]
    assert fit.success, fit.reason
    assert abs(complex(fit.c_re, fit.c_im) - C_TRUE) < 4 * fit.c_err + 2e-3


def test_pair_mode_threshold():
    assert pair_mode(50e6, 10e6) == PHASE_MODE
    assert pair_mode(-3e6, 10e6) == POPULATION_MODE


def test_train_pulses_is_even_and_capped():
    tone = 0.2 * np.hanning(40)
    n = train_pulses(PHASE_MODE, tone, 2.5e8, 200e6, 0.2, 1.0, 1.0, 400)
    assert n % 2 == 0 and 2 <= n <= 400
    assert train_pulses(PHASE_MODE, tone, 2.5e8, 200e6, 1e-6, 1.0, 1.0, 401) == 400
    assert train_pulses(POPULATION_MODE, tone, 2.5e8, 1e6, 0.2, 1.0, 1.0, 400, multiple=4) % 4 == 0


TONE = 0.2 * np.hanning(48)  # 48 ns raised-cosine pulse at 1 GS/s
RABI = 2.5e8  # Hz per unit amplitude


def test_stark_factor():
    assert stark_factor(-50e6, None) == 1.0
    # Delta = f_victim - f_source = +23.6 MHz, alpha = -207.5 MHz -> |alpha / (Delta + alpha)| = 1.128
    assert stark_factor(-23.6e6, 207.5e6) == pytest.approx(207.5 / (207.5 - 23.6))
    # source near the 1-2 transition: the factor diverges and is capped
    assert stark_factor(-207.5e6, 207.5e6) == 10.0


def test_near_second_transition():
    assert near_second_transition(-205e6, 207.5e6, 10e6)
    assert not near_second_transition(-23.6e6, 207.5e6, 10e6)
    assert not near_second_transition(-205e6, None, 10e6)


def test_comb_factor_and_pulse_rotation():
    assert comb_factor(8, 0.0, 48e-9) == pytest.approx(8)
    # half a cycle per period: pairs of pulses cancel
    assert comb_factor(8, 0.5 / 48e-9, 48e-9) == pytest.approx(0, abs=1e-9)
    on_resonance = pulse_rotation(TONE, RABI, 0.2, 0.0)
    assert on_resonance == pytest.approx(2 * np.pi * RABI * 0.2 * np.sum(TONE) * 1e-9)
    assert pulse_rotation(TONE, RABI, 0.2, 30e6) < on_resonance


def test_train_pulses_population_on_resonance_matches_area():
    first = pulse_rotation(TONE, RABI, 0.2, 0.0)
    n = train_pulses(POPULATION_MODE, TONE, RABI, 0.0, 0.2, 1.0, 1.0, 400)
    expected = int(np.ceil(1.0 / first))
    assert n == expected + expected % 2


def test_train_pulses_population_detuned_reaches_target_or_best():
    detuning, period = 3e6, 68e-9
    n = train_pulses(POPULATION_MODE, TONE, RABI, detuning, 0.2, 1.0, 1.0, 400, period_ns=68)
    rotation = pulse_rotation(TONE, RABI, 0.2, detuning) * comb_factor(n, detuning, period)
    best = max(pulse_rotation(TONE, RABI, 0.2, detuning) * comb_factor(k, detuning, period) for k in range(2, 401, 2))
    assert n % 2 == 0
    assert rotation >= 1.0 or rotation >= 0.9 * best


def test_train_pulses_population_unreachable_takes_shortest_near_best():
    # Delta T ~ 0.43: the comb never builds up, so a long train gains nothing
    n = train_pulses(POPULATION_MODE, TONE, RABI, 9e6, 0.01, 1.0, 1.0, 400)
    first = pulse_rotation(TONE, RABI, 0.01, 9e6)
    rotations = [first * comb_factor(k, 9e6, 48e-9) for k in range(2, 401, 2)]
    assert n == 2 * (1 + next(i for i, r in enumerate(rotations) if r >= 0.9 * max(rotations)))
    assert n < 20


def test_train_pulses_phase_includes_transmon_factor():
    two_level = train_pulses(PHASE_MODE, TONE, RABI, -23.6e6, 0.2, 1.0, 1.0, 4000)
    transmon = train_pulses(PHASE_MODE, TONE, RABI, -23.6e6, 0.2, 1.0, 1.0, 4000, anharmonicity_hz=207.5e6)
    assert transmon <= two_level


def test_population_averages_the_repeated_frames():
    pe = np.array([[0.1, 0.2, 0.1, 0.2], [0.5, 0.5, 0.5, 0.5]])
    p, sigma = population(pe, 100)
    assert np.allclose(p, [0.15, 0.5])
    assert sigma[1] == pytest.approx(np.sqrt(0.25 / 400))


def _fake_qubit(name, f, lo):
    return SimpleNamespace(name=name, xy=SimpleNamespace(RF_frequency=f, LO_frequency=lo))


def test_add_pair_info_and_fit_round_trip():
    """The fetched dataset (qubit_pair, point, frame) plus the plan gives the dataset the analysis expects."""
    raw = _dataset(PHASE_MODE)
    z = raw.z_re.values[0] + 1j * raw.z_im.values[0]
    pairs = [
        dict(
            source=_fake_qubit("qA", 5.5e9, 5.6e9),
            victim=_fake_qubit("qB", 5.45e9, 5.55e9),
            mode=PHASE_MODE,
            z=z,
            source_on=raw.source_on.values[0],
            center=0j,
            detuning=50e6,
            train_pulses=10,
        )
    ]
    fetched = raw[["pe"]].reset_coords(drop=True)
    ds = add_pair_info(fetched, pairs, SimpleNamespace(num_shots=400, r_max=0.1))
    assert ds.source_qubit.values.tolist() == ["qA"] and float(ds.victim_lo[0]) == 5.55e9
    ds_fit, fits = fit_raw_data(ds, NODE)
    fit = fits["qA-qB"]
    assert fit.success, fit.reason
    assert abs(complex(fit.c_re, fit.c_im) - C_TRUE) < 4 * fit.c_err + 2e-3
    assert ds_fit.obs.dims == ("qubit_pair", "point") and np.isnan(ds_fit.obs.values[0, -1])


def _state_qubit(name, f, lo, extras=None):
    return SimpleNamespace(name=name, xy=SimpleNamespace(RF_frequency=f, LO_frequency=lo), extras=extras or {})


def test_state_matrix_merges_state_and_run():
    from calibration_utils.driveline_crosstalk_matrix.analysis import CrosstalkFit, matrix_names, state_matrix

    key = "drive_crosstalk_complex"
    entry = lambda c, f, lo: {"c_re": c.real, "c_im": c.imag, "c_err": 1e-3, "frequency_hz": f, "victim_lo_hz": lo}
    qubits = {
        "q1": _state_qubit(
            "q1", 5.0e9, 5.1e9, {key: {"q2": entry(0.01j, 5.0e9, 5.2e9), "q3": entry(0.02, 5.0e9, 9e9)}}
        ),
        "q2": _state_qubit("q2", 5.1e9, 5.2e9),
        "q3": _state_qubit("q3", 5.2e9, 5.3e9),
        "q4": _state_qubit("q4", 5.3e9, 5.4e9),
    }

    def fit(source, victim, c, success):
        return CrosstalkFit(source, victim, "phase", c.real, c.imag, 1e-3, 1.0, 10.0, 0.0, 1.0, 5e9, success, "")

    fits = {"q2-q1": fit("q2", "q1", 0.03 + 0j, True), "q1-q2": fit("q1", "q2", 0.5 + 0j, False)}
    names = matrix_names(qubits, ["q1", "q2"], key)
    assert names == ["q1", "q2", "q3"]  # q3 has a stored entry, q4 nothing
    m = state_matrix(qubits, names, key, fits)
    # q1 -> q2: stored value kept although this run failed
    assert m["c"][1, 0] == pytest.approx(0.01j) and m["failed"][1, 0] and not m["updated"][1, 0]
    # q1 -> q3: stored with another victim LO, so stale
    assert m["stale"][2, 0]
    # q2 -> q1: written by this run
    assert m["c"][0, 1] == pytest.approx(0.03) and m["updated"][0, 1]
    assert np.isnan(m["c"][2, 1])


def test_unmeasurable_pairs_reports_if_limit():
    from calibration_utils.driveline_crosstalk_matrix.sequence import unmeasurable_pairs

    p = SimpleNamespace(min_stark_detuning_mhz=10)
    q1, q2 = _state_qubit("q1", 5.0e9, 5.1e9), _state_qubit("q2", 6.0e9, 6.1e9)
    for q, port in ((q1, object()), (q2, object())):
        q.xy.opx_output = port
        q.xy.sampling_rate = 1e9
        q.anharmonicity = 200e6
    reasons = unmeasurable_pairs([q1, q2], p)
    assert set(reasons) == {("q1", "q2"), ("q2", "q1")} and "IF" in reasons[("q1", "q2")]


def test_tone_range_shrinks_for_a_strong_source_pulse():
    from calibration_utils.driveline_crosstalk_matrix.sequence import tone_range, train_signal

    strong = 5 * TONE  # e.g. a weak source line whose x180 has 5x the area of the victim's
    r, n = tone_range(POPULATION_MODE, strong, RABI, 0.7e6, 0.2, 0.9, 1.0, 400)
    assert n == 2 and r < 0.2
    assert train_signal(POPULATION_MODE, strong, RABI, 0.7e6, r, n) == pytest.approx(1.0, rel=0.05)


def test_tone_range_grows_when_the_train_is_too_long():
    from calibration_utils.driveline_crosstalk_matrix.sequence import tone_range, train_signal

    weak = 0.05 * TONE
    r, n = tone_range(PHASE_MODE, weak, RABI, 30e6, 0.2, 1.5, 1.0, 400, max_duration_s=5e-6)
    assert r > 0.2 and n * len(weak) * 1e-9 <= 5e-6
    assert train_signal(PHASE_MODE, weak, RABI, 30e6, r, n) >= 1.0 / 1.5
    # capped by the output range
    r_capped, _ = tone_range(PHASE_MODE, weak, RABI, 30e6, 0.2, 0.3, 1.0, 400, max_duration_s=5e-6)
    assert r_capped == pytest.approx(0.3)


def test_tone_range_unchanged_when_the_signal_is_reached():
    from calibration_utils.driveline_crosstalk_matrix.sequence import tone_range, train_pulses

    r, n = tone_range(PHASE_MODE, TONE, RABI, 30e6, 0.2, 1.5, 1.0, 400)
    assert r == 0.2 and n == train_pulses(PHASE_MODE, TONE, RABI, 30e6, 0.2, 1.0, 1.0, 400)


def _plan_qubit(name, f, lo):
    pulse = SimpleNamespace(calculate_waveform=lambda: TONE)
    return SimpleNamespace(
        name=name,
        xy=SimpleNamespace(
            RF_frequency=f, LO_frequency=lo, opx_output=object(), sampling_rate=1e9, operations={"x180": pulse}
        ),
        extras={},
        anharmonicity=200e6,
        T2ramsey=50e-6,
    )


def test_plan_pairs_explicit_pairs():
    from calibration_utils.driveline_crosstalk_matrix import Parameters, plan_pairs

    qubits = [_plan_qubit(n, f, f + 100e6) for n, f in (("q1", 5.00e9), ("q2", 5.03e9), ("q3", 5.06e9))]
    params = Parameters(qubits=["q1", "q2", "q3"], pairs=["q1-q2", "q3 - q1", "q1-q9", "q1->q3"])
    logs = []
    pairs = plan_pairs(qubits, params, "key", log_callable=logs.append)
    assert sorted(p["key"] for p in pairs) == ["q1-q2", "q3-q1"]
    assert any("Ignoring pair 'q1-q9'" in line for line in logs)
    assert any("Ignoring pair 'q1->q3'" in line for line in logs)
    assert all(len(p["z"]) == 18 and p["r_max"] > 0 for p in pairs)


@pytest.mark.parametrize(
    "mode, center",
    [(PHASE_MODE, 0j), (PHASE_MODE, -C_TRUE * 0.9), (POPULATION_MODE, 0j)],
    ids=["zero", "previous", "pop"],
)
def test_plot_pair_fits_draws_the_map_around_minus_c(mode, center):
    import matplotlib

    matplotlib.use("Agg")
    from calibration_utils.driveline_crosstalk_matrix.plotting import plot_pair_fits

    ds = _dataset(mode, center=complex(center), scale=60.0 if mode == PHASE_MODE else 15.0)
    ds_fit, fits = fit_raw_data(ds, NODE)
    fig = plot_pair_fits(ds_fit)
    ax = fig.axes[0]
    assert "|c| =" in ax.get_title() and "arg c" in ax.get_title()
    # the red x sits at -c, inside the plotted (measured) region
    cross = [line for line in ax.get_lines() if line.get_marker() == "x"][0]
    c = complex(fits["qA-qB"].c_re, fits["qA-qB"].c_im)
    assert np.allclose([cross.get_xdata()[0], cross.get_ydata()[0]], [-c.real, -c.imag])
    x0, x1 = ax.get_xlim()
    assert x0 < -c.real < x1
    # the 17 measured points, coloured on the map's scale
    assert ax.collections and len(ax.collections[-1].get_offsets()) == 17


def test_old_datasets_are_renamed_before_the_fit():
    """Datasets written before the source/victim rename (drive_qubit, target_qubit, ...) still fit."""
    from calibration_utils.driveline_crosstalk_matrix.analysis import upgrade_names

    old = _dataset(PHASE_MODE).rename(
        {"source_qubit": "drive_qubit", "victim_qubit": "target_qubit", "source_frequency": "drive_frequency"}
    )
    assert "source_qubit" in upgrade_names(old).coords
    _, fits = fit_raw_data(old, NODE)
    assert fits["qA-qB"].source_qubit == "qA" and fits["qA-qB"].victim_qubit == "qB"


def test_stale_reads_entries_written_before_the_rename():
    from calibration_utils.driveline_crosstalk_matrix.analysis import _is_stale

    source, victim = _state_qubit("q1", 5.0e9, 5.1e9), _state_qubit("q2", 5.1e9, 5.2e9)
    assert not _is_stale({"frequency_hz": 5.0e9, "target_lo_hz": 5.2e9}, source, victim)
    assert _is_stale({"frequency_hz": 5.0e9, "target_lo_hz": 5.9e9}, source, victim)

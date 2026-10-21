"""Tests for the dynamical decoupling sequences, sweep schedule and fit (06c_dynamical_decoupling)."""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_utils.dynamical_decoupling import (  # noqa: E402
    DD_SEQUENCES,
    Parameters,
    assign_schedule_coords,
    get_dd_sequence,
    get_sweep_schedule,
    get_window_counts,
)
from calibration_utils.dynamical_decoupling.analysis import fit_dd_data  # noqa: E402


def _qubit(name="q1", t_pi=48, readout=2000):
    ops = {op: SimpleNamespace(length=t_pi) for op in ("x90", "x180", "y180")}
    return SimpleNamespace(
        name=name,
        T1=40e-6,
        grid_location="0,0",
        xy=SimpleNamespace(operations=ops),
        resonator=SimpleNamespace(operations={"readout": SimpleNamespace(length=readout)}),
    )


def _waits(steps):
    return [s[1] for s in steps if s[0] == "wait"]


def _plays(steps):
    return [(s[1], s[2]) for s in steps if s[0] == "play"]


# --- Sequences ---------------------------------------------------------------------------------------------------


def test_sequence_block_sizes():
    assert {name: s.pulses_per_block for name, s in DD_SEQUENCES.items()} == {
        "CPMG": 2,
        "XY4": 4,
        "XY8": 8,
        "XY16": 16,
    }


def test_sequence_pulse_order():
    assert DD_SEQUENCES["CPMG"].pulse_train(4) == [("y180", 0.0)] * 4
    assert DD_SEQUENCES["XY4"].pulse_train(8) == [("x180", 0.0), ("y180", 0.0)] * 4
    xy8 = [op for op, _ in DD_SEQUENCES["XY8"].pulse_train(8)]
    assert xy8 == ["x180", "y180", "x180", "y180", "y180", "x180", "y180", "x180"]
    xy16 = DD_SEQUENCES["XY16"].pulse_train(16)
    assert [op for op, _ in xy16] == xy8 * 2
    assert [ph for _, ph in xy16] == [0.0] * 8 + [180.0] * 8


def test_pulse_train_rejects_partial_blocks():
    with pytest.raises(ValueError, match="multiple of 4"):
        DD_SEQUENCES["XY4"].pulse_train(6)


def test_unknown_sequence():
    with pytest.raises(ValueError, match="Unknown DD sequence"):
        get_dd_sequence("UDD")


@pytest.mark.parametrize("name", list(DD_SEQUENCES))
@pytest.mark.parametrize("free_cc", [400, 401, 403, 457])
def test_window_spacing_is_uniform_including_boundaries(name, free_cc):
    """Gaps between pulses (also across two consecutive windows) differ by at most one clock cycle, and the waits add
    up to exactly the free evolution time."""
    sequence = DD_SEQUENCES[name]
    n = 16
    steps = sequence.window_steps(n, free_cc)
    waits = _waits(steps)
    assert sum(waits) == free_cc
    assert [s[0] for s in steps] == ["wait", "play"] * n + ["wait"]
    gaps = waits[1:-1] + [waits[-1] + waits[0]]  # inner gaps + gap across the window boundary
    assert max(gaps) - min(gaps) <= 1
    assert min(waits) >= 4


def test_window_rejects_too_short_spacing():
    with pytest.raises(ValueError, match="minimum"):
        DD_SEQUENCES["CPMG"].window_steps(4, 4 * 7)  # 7 cycles per gap < 8


def test_play_generates_strict_timing_qua():
    """Build the QUA code with a stand-in qubit and check the loop body and phase handling."""
    from qm import generate_qua_script
    from qm.qua import declare, frame_rotation_2pi, play, program, wait

    xy = SimpleNamespace(
        play=lambda op: play(op, "q1.xy"),
        wait=lambda cc: wait(cc, "q1.xy"),
        frame_rotation_2pi=lambda angle: frame_rotation_2pi(angle, "q1.xy"),
    )
    with program() as prog:
        n_windows, j = declare(int, value=3), declare(int)
        DD_SEQUENCES["XY16"].play(SimpleNamespace(xy=xy), n_windows, j, n_pulses=16, free_cc=150)
    script = generate_qua_script(prog)
    assert "strict_timing_()" in script
    plays = re.findall(r"play\(['\"]([^'\"]+)['\"], ['\"]q1.xy['\"]\)", script)
    assert plays == ["x90"] + [op for op, _ in DD_SEQUENCES["XY16"].pulse_train(16)] + ["x90"]
    # Each phase-shifted pulse is wrapped in a +/- 0.5 (180 deg) frame rotation
    assert script.count("frame_rotation_2pi(0.5,") == 8
    assert script.count("frame_rotation_2pi(-0.5,") == 8
    waits = [int(w) for w in re.findall(r"wait\((\d+), ['\"]q1.xy", script)]
    assert sum(waits) == 150


# --- Schedule ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(DD_SEQUENCES))
def test_schedule_window_is_exact(name):
    qubits = [_qubit("q1", 48), _qubit("q2", 40)]
    sched = get_sweep_schedule(Parameters(sequence={"sequence": name}), qubits)
    ppb = DD_SEQUENCES[name].pulses_per_block
    assert sched["window_ns"] == 2000
    assert np.all(sched["pulses_per_window"] % ppb == 0)
    for q in qubits:
        t_pi = sched["pi_lengths"][q.name]
        for n, free in zip(sched["pulses_per_window"], sched["free_cc"][q.name]):
            steps = DD_SEQUENCES[name].window_steps(int(n), int(free))
            assert 4 * sum(_waits(steps)) + n * t_pi == 2000


def test_window_counts_shared_axis():
    counts = get_window_counts(60, 25)
    expected = np.unique(np.concatenate([np.round(np.linspace(0, 60, 25)).astype(int), [1]]))
    assert list(counts) == list(expected)
    assert counts[0] == 0 and counts[1] == 1 and counts[-1] == 60
    # Fewer windows than points: every window count once, no extension past max_num_windows
    assert list(get_window_counts(10, 25)) == list(range(11))
    # The num_rounds point is always measured, and must fit in the sweep
    assert 7 in get_window_counts(60, 25, num_rounds=7)
    with pytest.raises(ValueError, match="num_rounds"):
        get_window_counts(10, 25, num_rounds=20)


def test_time_axis_is_whole_windows():
    sched = get_sweep_schedule(Parameters(sequence={"sequence": "XY4"}, sweep={"max_num_windows": 60}), [_qubit()])
    ds = xr.Dataset(
        coords={
            "qubit": ["q1"],
            "pulses_per_window": sched["pulses_per_window"],
            "point": np.arange(len(sched["window_counts"])),
        }
    )
    ds = assign_schedule_coords(ds, sched)
    assert ds.time.dims == ("point",)
    assert np.all(ds.time.values == 2000 * ds.windows.values)
    assert ds.time.values[-1] == 120_000
    assert np.all(ds.total_pulses.values == ds.pulses_per_window.values[:, None] * ds.windows.values[None, :])


def test_schedule_respects_minimum_spacing():
    sched = get_sweep_schedule(Parameters(sequence={"sequence": "CPMG"}), [_qubit(t_pi=48)])
    # 2000 // (32 + 48) = 25 -> 24 pulses max for CPMG
    assert sched["pulses_per_window"].max() == 24
    sched = get_sweep_schedule(Parameters(sequence={"sequence": "XY8"}), [_qubit(t_pi=48)])
    assert list(sched["pulses_per_window"]) == [8, 16, 24]
    with pytest.raises(ValueError, match="too short"):
        get_sweep_schedule(Parameters(sequence={"sequence": "XY16"}), [_qubit(t_pi=48, readout=760)])


def test_schedule_rejects_bad_pulses_per_window():
    with pytest.raises(ValueError, match="multiples of 4"):
        get_sweep_schedule(Parameters(sequence={"sequence": "XY4", "pulses_per_window": [4, 6]}), [_qubit()])


def test_schedule_rejects_missing_operation():
    q = _qubit()
    del q.xy.operations["x180"]
    with pytest.raises(ValueError, match="x180"):
        get_sweep_schedule(Parameters(sequence={"sequence": "XY4"}), [q])


# --- Fit ---------------------------------------------------------------------------------------------------------


def _synthetic(sequence="XY4", noise=0.005, seed=0, alpha=1.0, max_num_windows=40):
    sched = get_sweep_schedule(
        Parameters(sequence={"sequence": sequence}, sweep={"max_num_windows": max_num_windows}),
        [_qubit("q1", 40, readout=1500)],
    )
    n_values = sched["pulses_per_window"]
    ds = xr.Dataset(
        coords={"qubit": ["q1"], "pulses_per_window": n_values, "point": np.arange(len(sched["window_counts"]))}
    )
    ds = assign_schedule_coords(ds, sched)
    # Ground truth: 1/f-like dephasing filtered by more pulses + pulse-error term -> optimum in the middle
    true_T2_ns = 1e9 / np.array([1 / 80e-6 + 1e5 / n + 1500 * n for n in n_values])
    t = ds.time.values[None, None, :]
    clean = 0.5 + 0.45 * np.exp(-((t / true_T2_ns[None, :, None]) ** alpha))
    noisy = clean + np.random.default_rng(seed).normal(0, noise, clean.shape)
    # Average error per round over M = 10 rounds (the default num_rounds)
    true_p = (1 - np.exp(-((10 * 1500 / true_T2_ns) ** alpha) / 10)) / 2
    return ds.assign(state=(("qubit", "pulses_per_window", "point"), noisy)).state, 1e-3 * true_T2_ns, true_p


def test_fit_recovers_exponential_decay():
    signal, true_T2, true_p = _synthetic()
    ds_fit, fit_results = fit_dd_data(signal)
    f = ds_fit.sel(qubit="q1")
    T2_us, T2_err_us = 1e-3 * f.T2.values, 1e-3 * f.T2_error.values
    assert np.all(np.abs(T2_us - true_T2) < 3 * T2_err_us) and np.max(np.abs(T2_us / true_T2 - 1)) < 0.08
    assert np.all(f.alpha.values < 1.15)
    # All curves share the same dephased level here: no floor shift flagged
    assert not np.any(f.floor_shifted.values)
    # Error per round taken from the measured 10-window point, within 3 sigma of the truth, and agreeing with the fit
    assert np.all(f.error_per_round_is_measured.values)
    assert np.all(np.abs(f.error_per_round.values - true_p) < 3 * f.error_per_round_error.values)
    assert np.max(np.abs(f.error_per_round_fit.values / true_p - 1)) < 0.05
    r = fit_results["q1"]
    assert r.success and r.sequence == "XY4" and r.fit_check_ok


def test_fit_recovers_gaussian_decay():
    signal, true_T2, true_p = _synthetic(alpha=2.0, max_num_windows=60)
    ds_fit, _ = fit_dd_data(signal)
    f = ds_fit.sel(qubit="q1")
    assert np.max(np.abs(f.alpha.values - 2.0)) < 0.25
    assert np.max(np.abs(1e-3 * f.T2.values / true_T2 - 1)) < 0.05
    assert np.all(np.abs(f.error_per_round.values - true_p) < 3 * f.error_per_round_error.values)
    assert np.all(np.abs(f.error_per_round_fit.values - true_p) < 3 * f.error_per_round_fit_error.values + 1e-5)


def test_selection_follows_error_budget():
    signal, _, true_p = _synthetic()
    n_values = signal.pulses_per_window.values
    # Zero budget and no margin: the N with the lowest fitted error per window
    ds_fit, r = fit_dd_data(signal, max_extra_error_per_round=0, uncertainty_margin_sigma=0)
    p = ds_fit.error_per_round.sel(qubit="q1").values
    assert r["q1"].pulses_per_window == n_values[np.nanargmin(p)]
    assert r["q1"].extra_error_per_round == 0
    # 0.1% budget: fewest pulses whose true extra error is small, never more pulses than the best
    _, r = fit_dd_data(signal, max_extra_error_per_round=1e-3)
    i_sel = list(n_values).index(r["q1"].pulses_per_window)
    assert true_p[i_sel] - true_p.min() < 2e-3
    assert r["q1"].pulses_per_window <= n_values[np.nanargmin(p)]
    # Huge budget: the fewest pulses scanned, flagged as at the boundary
    _, r = fit_dd_data(signal, max_extra_error_per_round=1.0)
    assert r["q1"].pulses_per_window == n_values[0] and r["q1"].at_boundary


def test_error_per_round_falls_back_to_fit_without_measured_point():
    signal, _, _ = _synthetic()
    assert 20 not in signal.windows.values
    ds_fit, r = fit_dd_data(signal, num_rounds=20)
    f = ds_fit.sel(qubit="q1")
    assert not np.any(f.error_per_round_is_measured.values)
    assert np.allclose(f.error_per_round.values, f.error_per_round_fit.values)
    assert r["q1"].success and r["q1"].num_rounds == 20


def test_fit_accepts_legacy_n_window_dataset():
    """Datasets saved by the former 06c_cpmg node use 'n_window' and a per-curve time coordinate."""
    from calibration_utils.dynamical_decoupling.analysis import process_raw_dataset

    signal, _, _ = _synthetic("CPMG")
    legacy = signal.to_dataset(name="state").drop_vars(["sequence", "windows"]).rename(pulses_per_window="n_window")
    legacy = legacy.assign_coords(
        time=(("qubit", "n_window", "point"), np.broadcast_to(legacy.time.values, legacy.state.shape).copy())
    )
    node = SimpleNamespace(parameters=SimpleNamespace(use_state_discrimination=True))
    ds = process_raw_dataset(legacy, node)
    _, fit_results = fit_dd_data(ds.state)
    assert fit_results["q1"].success and fit_results["q1"].sequence == "CPMG"


def test_noise_spectrum_first_order():
    """S_f(f0) = (1/T2 - 1/(2 T1)) / 8 (one-sided, Hz^2/Hz) at f0 = N / (2 window)."""
    signal, true_T2, _ = _synthetic()
    ds_fit, _ = fit_dd_data(signal, T1={"q1": 40e-6})
    f = ds_fit.sel(qubit="q1")
    n_values = signal.pulses_per_window.values
    assert np.allclose(f.noise_frequency.values, n_values / (2 * 1500e-9))
    true_psd = (1 / (1e-6 * true_T2) - 1 / (2 * 40e-6)) / 8
    assert np.max(np.abs(f.noise_psd.values / true_psd - 1)) < 0.15
    assert bool(f.T1_subtracted)
    # Without T1 the raw T2 is used
    ds_fit, _ = fit_dd_data(signal)
    assert not bool(ds_fit.T1_subtracted.sel(qubit="q1"))
    assert np.allclose(ds_fit.noise_psd.sel(qubit="q1").values, 1e9 / ds_fit.T2.sel(qubit="q1").values / 8)


def test_noise_power_law_fit():
    from calibration_utils.dynamical_decoupling.analysis import fit_noise_power_law

    f = np.linspace(0.5e6, 6e6, 12)
    psd = 2000 * (1e6 / f) ** 0.9 + 400
    rng = np.random.default_rng(0)
    noisy = psd * (1 + 0.03 * rng.standard_normal(f.size))
    res = fit_noise_power_law(f, noisy, 0.03 * noisy, alpha=np.ones(f.size))
    assert abs(res["amplitude"] / 2000 - 1) < 0.15
    assert abs(res["exponent"] - 0.9) < 0.15
    assert abs(res["floor"] / 400 - 1) < 0.3
    # Unreliable (alpha > 1.5) and invalid points are excluded; too few points -> no fit
    alpha = np.array([2.0] * 9 + [1.0] * 3)
    assert fit_noise_power_law(f, noisy, 0.03 * noisy, alpha=alpha) is None
    # A rising spectrum cannot be described by a decaying power law: no fit instead of a degenerate one
    rising = 500 * (f / 1e6) ** 1.5
    assert fit_noise_power_law(f, rising, 0.03 * rising) is None


def test_noise_fit_can_be_disabled():
    # Decays whose dephasing spectrum is a clean power law + floor, S_f = 3000 (1 MHz / f)^0.8 + 300 Hz
    sched = get_sweep_schedule(
        Parameters(sequence={"sequence": "CPMG"}, sweep={"max_num_windows": 60}), [_qubit("q1", 48)]
    )
    n_values = sched["pulses_per_window"]
    ds = xr.Dataset(
        coords={"qubit": ["q1"], "pulses_per_window": n_values, "point": np.arange(len(sched["window_counts"]))}
    )
    ds = assign_schedule_coords(ds, sched)
    psd = 3000 * (1e6 / (n_values / (2 * 2e-6))) ** 0.8 + 300
    T2_ns = 1e9 / (8 * psd + 1 / (2 * 40e-6))
    y = 0.5 + 0.45 * np.exp(-ds.time.values[None, None, :] / T2_ns[None, :, None])
    y = y + np.random.default_rng(1).normal(0, 0.003, y.shape)
    signal = ds.assign(state=(("qubit", "pulses_per_window", "point"), y)).state

    ds_fit, r = fit_dd_data(signal, T1={"q1": 40e-6}, fit_noise_spectrum=False)
    assert np.isnan(r["q1"].noise_exponent)
    assert np.all(np.isfinite(ds_fit.noise_psd.sel(qubit="q1").values))  # spectrum values still computed
    _, r = fit_dd_data(signal, T1={"q1": 40e-6})
    assert abs(r["q1"].noise_exponent - 0.8) < 0.2
    assert abs(r["q1"].noise_amplitude / 3000 - 1) < 0.2


def test_decayed_points_use_fit_and_noisy_points_are_not_selected():
    """If almost no coherence is left after M windows, the measured point is noise: use the fit instead, and never
    select an N whose error per round is not resolved."""
    signal, _, _ = _synthetic()
    # M = 40 windows = 60 us is several T2: little coherence left for most N
    ds_fit, r = fit_dd_data(signal, num_rounds=40)
    f = ds_fit.sel(qubit="q1")
    c_last = f.coherence.isel(point=-1).values
    assert np.all(f.error_per_round_is_measured.values == (c_last >= 0.2))
    i_sel = list(f.pulses_per_window.values).index(r["q1"].pulses_per_window)
    assert f.error_per_round_error.values[i_sel] <= 0.5 * f.error_per_round.values[i_sel]


def test_floor_shift_is_flagged_and_excluded():
    """Curves that settle at a lower level (leakage / heating at many pulses) get their own offset, so the T2 of the
    other curves is not biased, and they are flagged and excluded from the decision."""
    signal, true_T2, _ = _synthetic()
    shifted = signal.copy()
    shifted[0, -2:, :] -= 0.15 * (1 - np.exp(-shifted.time.values / 20_000))[None, :]  # last two N drift lower
    ds_fit, r = fit_dd_data(shifted)
    f = ds_fit.sel(qubit="q1")
    n_values = list(signal.pulses_per_window.values)
    assert list(f.floor_shifted.values) == [False] * (len(n_values) - 2) + [True, True]
    assert r["q1"].floor_shifted_pulses_per_window == n_values[-2:]
    assert r["q1"].pulses_per_window not in n_values[-2:]
    # Unshifted curves keep an unbiased T2
    T2_us = 1e-3 * f.T2.values[:-2]
    assert np.max(np.abs(T2_us / true_T2[:-2] - 1)) < 0.08

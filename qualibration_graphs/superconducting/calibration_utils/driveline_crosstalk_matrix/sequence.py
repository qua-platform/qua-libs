"""Planning of the drive-line crosstalk measurement: which pairs, in which mode, over which grid and how long a train.

Every directed pair (source -> victim) among the selected qubits is planned independently: the coefficient c of a
pair does not depend on the other pairs, so the pairs measured in a run are only a matter of time.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from .analysis import PHASE_MODE, POPULATION_MODE, stored_coefficient

OPERATION = "x180"
"""Gate of the source qubit played in the train. c is a ratio of transfer functions, so it does not depend on the gate;
an even number of x180 leaves the source qubit in |0>."""
RING_FRACTIONS = (0.5, 1.0)
NUM_ANGLES = 8
"""Grid of tone amplitudes z: the centre and two rings (r_max / 2 and r_max) of 8 points, 17 points in all."""
TARGET_SIGNAL_RAD = 1.0
"""Largest expected signal over the grid (Stark phase, or victim rotation in population mode): well above the shot
noise, well below a phase wrap and inside the small-angle regime."""
MAX_TRAIN_PULSES = 400
"""Upper limit of the automatic train length."""
MAX_TRAIN_FRACTION_OF_T2 = 0.3
"""Longest train, as a fraction of the victim's T2*, before the Ramsey visibility suffers."""
MAX_ADAPTED_R_MAX = 0.5
"""Largest radius the per-pair adaptation may reach: a large tone on the victim line also leaks back onto the source
qubit (at its own frequency) and over-rotates it."""
MAX_IF_HZ = 400e6
"""Largest |f_source - victim LO| the victim line can play (MW-FEM IF range)."""

__all__ = [
    "OPERATION",
    "grid_points",
    "pair_mode",
    "rabi_hz_per_amplitude",
    "stark_factor",
    "near_second_transition",
    "comb_factor",
    "pulse_rotation",
    "train_pulses",
    "train_signal",
    "tone_range",
    "unmeasurable_pairs",
    "plan_pairs",
    "add_pair_info",
]


# --------------------------------------------------------------------------------------------- physics
def grid_points(center: complex, r_max: float, ring_fractions: Sequence[float], num_angles: int) -> np.ndarray:
    """Complex tone amplitudes z: the centre plus rings of radius fraction * r_max, num_angles points each.

    Successive rings are rotated by half an angular step so that the grid covers the plane evenly.
    """
    points = [complex(center)]
    for k, fraction in enumerate(ring_fractions):
        angles = 2 * np.pi * (np.arange(num_angles) + 0.5 * (k % 2)) / num_angles
        points.extend(center + fraction * r_max * np.exp(1j * angles))
    return np.asarray(points, dtype=complex)


def pair_mode(detuning_hz: float, min_stark_detuning_hz: float) -> str:
    """Ramsey (Stark phase) for detuned pairs, excited population for close pairs."""
    return PHASE_MODE if abs(detuning_hz) >= min_stark_detuning_hz else POPULATION_MODE


def rabi_hz_per_amplitude(x180_waveform: np.ndarray, sample_ns: float = 1.0) -> float:
    """Rabi frequency (Hz) per unit output amplitude of a line, from the pulse area of its calibrated x180."""
    area = abs(np.sum(x180_waveform)) * sample_ns * 1e-9  # amplitude * s
    return 0.5 / area


def stark_factor(detuning_hz: float, anharmonicity_hz: Optional[float]) -> float:
    """Transmon correction |alpha / (Delta + alpha)| of the 0-1 AC Stark shift, with Delta = f_victim - f_source
    and alpha = -|anharmonicity| (1 for a two-level system). ``detuning_hz`` is f_source - f_victim."""
    if not anharmonicity_hz or not np.isfinite(anharmonicity_hz):
        return 1.0
    alpha = -abs(float(anharmonicity_hz))
    denominator = -float(detuning_hz) + alpha
    return abs(alpha / denominator) if abs(denominator) > 1e6 else 10.0


def near_second_transition(detuning_hz: float, anharmonicity_hz: Optional[float], window_hz: float) -> bool:
    """True when f_source lies within ``window_hz`` of the victim's 1-2 transition f_victim - |alpha|.

    There the Stark model fails, and a victim in |0> barely responds, so neither mode applies."""
    if not anharmonicity_hz or not np.isfinite(anharmonicity_hz):
        return False
    return abs(float(detuning_hz) + abs(float(anharmonicity_hz))) < window_hz


def comb_factor(n: int, detuning_hz: float, period_s: float) -> float:
    """|sum_k exp(i 2 pi Delta k T)| = |sin(n pi Delta T) / sin(pi Delta T)| for n pulses spaced by T."""
    s = np.sin(np.pi * detuning_hz * period_s)
    return float(n) if abs(s) < 1e-12 else abs(np.sin(n * np.pi * detuning_hz * period_s) / s)


def pulse_rotation(tone_waveform: np.ndarray, rabi_hz: float, field: float, detuning_hz: float, sample_ns=1.0):
    """First-order rotation [rad] of the victim by one pulse detuned by Delta: 2 pi |sum Omega(t) e^{i 2 pi Delta t}| dt.

    On resonance it is the pulse area; off resonance it is the pulse spectrum at Delta."""
    dt = sample_ns * 1e-9
    t = np.arange(len(tone_waveform)) * dt
    omega = rabi_hz * field * np.asarray(tone_waveform)
    return float(2 * np.pi * abs(np.sum(omega * np.exp(2j * np.pi * detuning_hz * t))) * dt)


def train_pulses(
    mode: str,
    tone_waveform: np.ndarray,
    rabi_per_amplitude_hz: float,
    detuning_hz: float,
    field_max: float,
    target_phase_rad: float,
    target_rotation_rad: float,
    max_pulses: int,
    multiple: int = 2,
    sample_ns: float = 1.0,
    anharmonicity_hz: Optional[float] = None,
    period_ns: Optional[float] = None,
) -> int:
    """Number of source pulses such that the largest expected signal over the grid reaches its target value.

    ``field_max`` is the largest |c + z| expected over the grid. The result is a multiple of ``multiple`` (2 for an
    x180, so that the source qubit ends in |0>) between ``multiple`` and ``max_pulses``.

    - phase mode: the Stark phase adds up linearly with the number of pulses. Per pulse it is
      2 pi * sum(Omega^2 / (2 |Delta|)) dt, times the transmon factor |alpha / (Delta + alpha)|.
    - population mode: the rotation of a detuned victim builds up coherently only near Delta = m / T, so the train
      length is searched: one pulse's spectrum at Delta times the comb factor of n pulses spaced by the period T
      (``period_ns``, the pulse length if None; the QUA loop can add a gap between pulses). The first n reaching
      ``target_rotation_rad`` is taken; if none reaches it, the shortest n within 90 % of the largest rotation.
    """
    n_min = multiple
    n_max = max(n_min, max_pulses - max_pulses % multiple)
    if mode == PHASE_MODE:
        omega_hz = rabi_per_amplitude_hz * field_max * np.abs(tone_waveform)
        per_pulse = 2 * np.pi * np.sum(omega_hz**2 / (2 * abs(detuning_hz))) * sample_ns * 1e-9
        per_pulse *= stark_factor(detuning_hz, anharmonicity_hz)
        n = n_max if per_pulse <= 0 else int(np.ceil(target_phase_rad / per_pulse))
        n += -n % multiple  # round up to a multiple
        return int(np.clip(n, n_min, n_max))
    period_s = (period_ns if period_ns else len(tone_waveform) * sample_ns) * 1e-9
    first = pulse_rotation(tone_waveform, rabi_per_amplitude_hz, field_max, detuning_hz, sample_ns)
    counts = np.arange(n_min, n_max + 1, multiple)
    rotation = np.array([first * comb_factor(int(n), detuning_hz, period_s) for n in counts])
    reached = np.flatnonzero(rotation >= target_rotation_rad)
    if reached.size:
        return int(counts[reached[0]])
    # Not reachable: the shortest train within 90 % of the best rotation (a longer one only adds decoherence)
    return int(counts[np.flatnonzero(rotation >= 0.9 * np.max(rotation))[0]])


def train_signal(
    mode: str,
    tone_waveform: np.ndarray,
    rabi_per_amplitude_hz: float,
    detuning_hz: float,
    field: float,
    n: int,
    sample_ns: float = 1.0,
    anharmonicity_hz: Optional[float] = None,
    period_ns: Optional[float] = None,
) -> float:
    """Expected signal of a train of n pulses for a field |c + z|: the Stark phase [rad] in phase mode, the
    first-order rotation [rad] of the victim in population mode (same model as train_pulses)."""
    if mode == PHASE_MODE:
        omega_hz = rabi_per_amplitude_hz * field * np.abs(tone_waveform)
        per_pulse = 2 * np.pi * np.sum(omega_hz**2 / (2 * abs(detuning_hz))) * sample_ns * 1e-9
        return float(n * per_pulse * stark_factor(detuning_hz, anharmonicity_hz))
    period_s = (period_ns if period_ns else len(tone_waveform) * sample_ns) * 1e-9
    first = pulse_rotation(tone_waveform, rabi_per_amplitude_hz, field, detuning_hz, sample_ns)
    return float(first * comb_factor(int(n), detuning_hz, period_s))


def tone_range(
    mode: str,
    tone_waveform: np.ndarray,
    rabi_per_amplitude_hz: float,
    detuning_hz: float,
    r_start: float,
    r_limit: float,
    target_signal: float,
    max_pulses: int,
    multiple: int = 2,
    max_duration_s: Optional[float] = None,
    sample_ns: float = 1.0,
    anharmonicity_hz: Optional[float] = None,
    period_ns: Optional[float] = None,
) -> Tuple[float, int]:
    """Radius of the tone grid and train length of a pair, so that the largest expected signal is close to
    ``target_signal``.

    The train length is first chosen for r_start (train_pulses), with at most max_pulses pulses and, if given,
    max_duration_s. If even the shortest train gives more than 1.5x the target signal (a strong source pulse on a
    sensitive victim), the radius is shrunk. If the longest allowed train gives less than the target signal / 1.5 (a
    weak source pulse, or a victim line that needs a large amplitude), the radius is enlarged, up to r_limit (the output range). The
    signal scales with r^2 in phase mode and with r in population mode."""
    kwargs = dict(sample_ns=sample_ns, anharmonicity_hz=anharmonicity_hz, period_ns=period_ns)
    pulse_s = (period_ns if period_ns else len(tone_waveform) * sample_ns) * 1e-9
    n_allowed = max_pulses
    if max_duration_s:
        n_allowed = max(multiple, min(max_pulses, int(max_duration_s / pulse_s)))

    def length(r):
        return train_pulses(
            mode,
            tone_waveform,
            rabi_per_amplitude_hz,
            detuning_hz,
            r,
            target_signal,
            target_signal,
            n_allowed,
            multiple,
            **kwargs,
        )

    def signal(r, n):
        return train_signal(mode, tone_waveform, rabi_per_amplitude_hz, detuning_hz, r, n, **kwargs)

    exponent = 0.5 if mode == PHASE_MODE else 1.0
    r = r_start
    n = length(r)
    s = signal(r, n)
    if s > 1.5 * target_signal:
        # Keep the shortest train, with the radius that brings its signal to the target
        return r * (target_signal / s) ** exponent, n
    if s < target_signal / 1.5 and r < r_limit:
        r = min(r_limit, r * (target_signal / max(s, 1e-12)) ** exponent)
        return r, length(r)
    return r, n


# --------------------------------------------------------------------------------------------- planning
def _skip_reason(source, victim, detuning: float, tone_if: float, anharmonicity, parameters) -> Optional[str]:
    """Why the pair cannot be measured, or None."""
    if abs(tone_if) > MAX_IF_HZ:
        return f"tone IF {tone_if / 1e6:.0f} MHz outside +/-{MAX_IF_HZ / 1e6:.0f} MHz"
    if source.xy.opx_output is victim.xy.opx_output:
        return "source and victim share the same output port"
    if getattr(source.xy, "sampling_rate", None) != getattr(victim.xy, "sampling_rate", None):
        return "the source and victim ports have different sampling rates"
    window_mhz = parameters.min_stark_detuning_mhz
    if near_second_transition(detuning, anharmonicity, window_mhz * 1e6):
        return (
            f"f_source is within {window_mhz} MHz of {victim.name}'s 1-2 transition, where the Stark model fails and "
            "a victim in |0> barely responds"
        )
    return None


def unmeasurable_pairs(qubits, parameters) -> Dict[Tuple[str, str], str]:
    """The directed pairs (source, victim) among ``qubits`` that this node cannot measure, with the reason."""
    reasons = {}
    for victim in qubits:
        for source in qubits:
            if source is victim:
                continue
            f_source = float(source.xy.RF_frequency)
            tone_if = f_source - float(victim.xy.LO_frequency)
            detuning = f_source - float(victim.xy.RF_frequency)
            anharmonicity = getattr(victim, "anharmonicity", None)
            reason = _skip_reason(source, victim, detuning, tone_if, anharmonicity, parameters)
            if reason:
                reasons[(source.name, victim.name)] = reason
    return reasons


def _requested_pairs(names: List[str], entries: Optional[List[str]], log_callable) -> Optional[set]:
    """The (source, victim) names of the "source-victim" entries, or None for every pair."""
    if not entries:
        return None
    requested = set()
    for entry in entries:
        parts = entry.replace(" ", "").split("-")
        if len(parts) != 2 or parts[0] not in names or parts[1] not in names or parts[0] == parts[1]:
            log_callable(f"WARNING Ignoring pair '{entry}': expected 'source-victim' among the selected qubits")
            continue
        requested.add((parts[0], parts[1]))
    return requested


def plan_pairs(qubits, parameters, state_key: str, log_callable=print) -> List[dict]:
    """The measurable directed pairs among ``qubits`` (the given ``pairs``, or all of them), each with its mode, grid
    of tone amplitudes z and train length.

    Unless train_pulses is set, the grid radius of every pair starts at r_max and is adapted (tone_range): shrunk when
    even the shortest train overshoots the target signal (a strong source pulse on a sensitive victim), enlarged up to
    MAX_ADAPTED_R_MAX when the train would be too long (a weak source pulse, or a victim line needing a large amplitude).

    Every pair is a dict with the keys: key ("source-victim"), source, victim, mode, tone (complex waveform of the
    source operation), tone_op (name of the temporary tone operation on victim.xy), tone_if, detuning, center, r_max
    (radius of the pair's grid), z (the grid followed by a reference point with the source and the tone off),
    source_on and train_pulses.
    """
    p = parameters
    requested = _requested_pairs([q.name for q in qubits], p.pairs, log_callable)
    multiple = 2  # even number of x180, so that the source qubit ends in |0>
    pairs = []
    for victim in qubits:
        rabi_victim = rabi_hz_per_amplitude(victim.xy.operations["x180"].calculate_waveform())
        anharmonicity = getattr(victim, "anharmonicity", None)
        t2 = getattr(victim, "T2ramsey", None)
        for source in qubits:
            if source is victim or (requested is not None and (source.name, victim.name) not in requested):
                continue
            key = f"{source.name}-{victim.name}"
            f_source = float(source.xy.RF_frequency)
            tone_if = f_source - float(victim.xy.LO_frequency)
            detuning = f_source - float(victim.xy.RF_frequency)
            reason = _skip_reason(source, victim, detuning, tone_if, anharmonicity, p)
            if reason:
                log_callable(f"Skipping {key}: {reason}")
                continue
            mode = pair_mode(detuning, p.min_stark_detuning_mhz * 1e6)

            # Grid centre: 0, or the null -c of the coefficient stored by a previous run
            stored = stored_coefficient(source.extras, victim.name, state_key)
            center = -stored if (p.center == "previous" and stored is not None) else 0j
            tone = np.asarray(source.xy.operations[OPERATION].calculate_waveform(), dtype=complex)

            r = p.r_max
            if p.train_pulses:
                n_train = p.train_pulses + (-p.train_pulses % multiple)
            else:
                # Largest radius: the output range (|z| < 2 and |z| * peak < 1) and MAX_ADAPTED_R_MAX
                r_limit = min(1.99, 0.99 / np.max(np.abs(tone))) - abs(center)
                r_limit = min(r_limit, max(MAX_ADAPTED_R_MAX, p.r_max))
                r, n_train = tone_range(
                    mode,
                    tone,
                    rabi_victim,
                    detuning,
                    p.r_max,
                    r_limit,
                    TARGET_SIGNAL_RAD,
                    MAX_TRAIN_PULSES,
                    multiple,
                    max_duration_s=MAX_TRAIN_FRACTION_OF_T2 * t2 if t2 else None,
                    anharmonicity_hz=anharmonicity,
                )
                if not np.isclose(r, p.r_max):
                    log_callable(f"{key}: tone range r_max {p.r_max} -> {r:.3f} (adapted to the target signal)")
            z_max = abs(center) + r
            if z_max >= 2 or z_max * np.max(np.abs(tone)) >= 1:
                log_callable(f"Skipping {key}: |z| up to {z_max:.3f} exceeds the output range")
                continue
            duration = n_train * len(tone) * 1e-9
            if mode == PHASE_MODE and t2 and duration > MAX_TRAIN_FRACTION_OF_T2 * t2:
                log_callable(
                    f"WARNING {key}: train of {duration * 1e6:.1f} us > {MAX_TRAIN_FRACTION_OF_T2} T2 "
                    f"({t2 * 1e6:.1f} us), visibility will suffer"
                )

            # Last point: source and tone both off (reference, not used by the fit)
            z = np.append(grid_points(center, r, RING_FRACTIONS, NUM_ANGLES), 0j)
            source_on = np.append(np.ones(len(z) - 1, dtype=int), 0)
            pairs.append(
                dict(
                    key=key,
                    source=source,
                    victim=victim,
                    mode=mode,
                    tone=tone,
                    tone_op=f"_xtalk_tone_{source.name}",
                    tone_if=tone_if,
                    detuning=detuning,
                    center=center,
                    r_max=r,
                    z=z,
                    source_on=source_on,
                    train_pulses=n_train,
                )
            )
            log_callable(
                f"{key}: {mode} mode, detuning {detuning / 1e6:+.1f} MHz, tone IF {tone_if / 1e6:+.1f} MHz, "
                f"{n_train} x {OPERATION} ({duration * 1e6:.2f} us), r_max {r:.3f}, {len(z)} points"
            )
    return pairs


def add_pair_info(ds: xr.Dataset, pairs: List[dict], parameters) -> xr.Dataset:
    """Attach the plan of every pair (grid, mode, frequencies, train length) to the raw dataset, so that the
    analysis needs nothing but the dataset (also when reloaded with load_data_id)."""
    ds = ds.assign(
        z_re=(("qubit_pair", "point"), np.array([pair["z"].real for pair in pairs])),
        z_im=(("qubit_pair", "point"), np.array([pair["z"].imag for pair in pairs])),
        source_on=(("qubit_pair", "point"), np.array([pair["source_on"] for pair in pairs])),
        center_re=("qubit_pair", [float(np.real(pair["center"])) for pair in pairs]),
        center_im=("qubit_pair", [float(np.imag(pair["center"])) for pair in pairs]),
        r_max=("qubit_pair", [float(pair.get("r_max", parameters.r_max)) for pair in pairs]),
        # Stored as floats: integer frequencies in Hz overflow the int32 of the netCDF3 files
        source_frequency=("qubit_pair", [float(pair["source"].xy.RF_frequency) for pair in pairs]),
        victim_lo=("qubit_pair", [float(pair["victim"].xy.LO_frequency) for pair in pairs]),
        detuning=("qubit_pair", [float(pair["detuning"]) for pair in pairs]),
        train_pulses=("qubit_pair", [int(pair["train_pulses"]) for pair in pairs]),
    )
    ds = ds.assign_coords(
        source_qubit=("qubit_pair", [pair["source"].name for pair in pairs]),
        victim_qubit=("qubit_pair", [pair["victim"].name for pair in pairs]),
        mode=("qubit_pair", [pair["mode"] for pair in pairs]),
    )
    ds.attrs.update(num_shots=parameters.num_shots, r_max=parameters.r_max, operation=OPERATION)
    return ds

"""Hybrid simulator for barrier-virtualization analysis tests.

This simulator is test-only and generates synthetic 2D pair scans
(``drive barrier`` x ``detuning``) for the barrier-barrier virtualization
pipeline:

1. Tunnel couplings follow the paper-style local exponential model
   ``t_i = t0_i * exp(sum_j Gamma_ij * dB_j)``.
2. Per-drive detuning traces are generated from a finite-temperature
   two-level transition model.
3. Optional ``qarray`` sensor output is mixed in as a realistic
   background component.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

try:
    from qarray import ChargeSensedDotArray

    _QARRAY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    ChargeSensedDotArray = None
    _QARRAY_AVAILABLE = False


def finite_temperature_excess_charge(
    detuning: np.ndarray,
    tunnel_coupling: float,
    thermal_energy: float,
    center: float,
) -> np.ndarray:
    """Finite-temperature charge-transition model used for synthetic traces."""
    eps = np.asarray(detuning, dtype=float) - float(center)
    t = max(abs(float(tunnel_coupling)), 1e-18)
    kbt = max(abs(float(thermal_energy)), 1e-18)
    omega = np.sqrt(eps * eps + (2.0 * t) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(omega > 0.0, eps / omega, 0.0)
        thermal = np.tanh(omega / (2.0 * kbt))
    return 0.5 * (1.0 - ratio * thermal)


def _normalize_map(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    centered = arr - np.nanmean(arr)
    std = float(np.nanstd(centered))
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(centered)
    return centered / std


@dataclass
class HybridBarrierSimulationConfig:
    """Configuration for synthetic barrier-pair scan generation."""

    barrier_names: Sequence[str]
    barrier_exponent_matrix: np.ndarray
    base_tunnel_couplings: np.ndarray
    drive_values: np.ndarray
    detuning_values: np.ndarray
    barrier_dc_offsets: Optional[np.ndarray] = None
    thermal_energy: float = 0.14
    detuning_center: float = 0.0
    detuning_center_drive_factor: float = 0.05
    linearize_exponential_locally: bool = False
    enforce_nearest_neighbor_gamma: bool = True
    nearest_neighbor_gamma_ratio_max: float = 0.35
    zero_non_nearest_neighbor_gamma: bool = True
    use_paper_signal_model: bool = True
    paper_signal_v0: float = 0.25
    paper_signal_delta_v: float = 1.10
    paper_signal_s0: float = 0.0
    paper_signal_s1: float = 0.0
    signal_offset: float = 0.25
    signal_scale: float = 1.10
    qarray_background_weight: float = 0.03
    analytic_background_weight: float = 0.02
    noise_std: float = 0.0015
    random_seed: int = 1337
    use_qarray_background: bool = True


class HybridBarrierVirtualizationSimulator:
    """Generate synthetic per-pair scans for barrier virtualization tests."""

    def __init__(self, config: HybridBarrierSimulationConfig):
        self.config = self._validate_config(config)
        self.barrier_names = list(self.config.barrier_names)
        self._index = {name: idx for idx, name in enumerate(self.barrier_names)}
        self._effective_gamma = self._build_effective_gamma_matrix(self.config.barrier_exponent_matrix)
        self._qarray_model = self._build_qarray_model() if self.config.use_qarray_background else None

    def _validate_config(self, config: HybridBarrierSimulationConfig) -> HybridBarrierSimulationConfig:
        names = list(config.barrier_names)
        n = len(names)
        if n == 0:
            raise ValueError("barrier_names must be non-empty.")

        gamma = np.asarray(config.barrier_exponent_matrix, dtype=float)
        if gamma.shape != (n, n):
            raise ValueError(f"barrier_exponent_matrix must have shape {(n, n)}, got {gamma.shape}.")

        t0 = np.asarray(config.base_tunnel_couplings, dtype=float)
        if t0.shape != (n,):
            raise ValueError(f"base_tunnel_couplings must have shape {(n,)}, got {t0.shape}.")
        if np.any(t0 <= 0.0):
            raise ValueError("base_tunnel_couplings must be strictly positive.")

        x = np.asarray(config.drive_values, dtype=float)
        y = np.asarray(config.detuning_values, dtype=float)
        offsets = (
            np.zeros((n,), dtype=float)
            if config.barrier_dc_offsets is None
            else np.asarray(config.barrier_dc_offsets, dtype=float)
        )
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("drive_values and detuning_values must be 1D arrays.")
        if x.size < 3:
            raise ValueError("drive_values must contain at least 3 points.")
        if y.size < 21:
            raise ValueError("detuning_values must contain at least 21 points.")
        if offsets.shape != (n,):
            raise ValueError(f"barrier_dc_offsets must have shape {(n,)}, got {offsets.shape}.")

        config.barrier_exponent_matrix = gamma
        config.base_tunnel_couplings = t0
        config.drive_values = x
        config.detuning_values = y
        config.barrier_dc_offsets = offsets
        return config

    def _build_effective_gamma_matrix(self, gamma_matrix: np.ndarray) -> np.ndarray:
        """Apply Eq. (2)-compatible locality assumptions to Γ."""
        gamma = np.asarray(gamma_matrix, dtype=float).copy()
        n = gamma.shape[0]

        if not self.config.enforce_nearest_neighbor_gamma:
            return gamma

        out = np.zeros_like(gamma)
        ratio_max = max(float(self.config.nearest_neighbor_gamma_ratio_max), 0.0)

        for i in range(n):
            diag = float(gamma[i, i])
            out[i, i] = diag
            diag_abs = abs(diag)

            for j in range(n):
                if j == i:
                    continue
                distance = abs(i - j)
                if distance == 1:
                    raw = float(gamma[i, j])
                    if diag_abs > 0.0:
                        limit = ratio_max * diag_abs
                        if limit > 0.0:
                            raw = np.sign(raw) * min(abs(raw), limit)
                    out[i, j] = raw
                elif not self.config.zero_non_nearest_neighbor_gamma:
                    out[i, j] = float(gamma[i, j])

        return out

    def _build_qarray_model(self):
        if not _QARRAY_AVAILABLE:  # pragma: no cover - depends on optional package
            return None

        n_barriers = len(self.barrier_names)
        n_gates = 2 + n_barriers + 1  # plunger_L, plunger_R, barriers..., sensor

        cdd = [[0.12, 0.08], [0.08, 0.13]]
        cgd = np.zeros((2, n_gates), dtype=float)
        cgd[0, 0] = 0.13
        cgd[1, 1] = 0.11
        for barrier_idx in range(n_barriers):
            gate_idx = 2 + barrier_idx
            cgd[0, gate_idx] = 0.035 + 0.004 * max(n_barriers - barrier_idx - 1, 0)
            cgd[1, gate_idx] = 0.035 + 0.004 * barrier_idx

        cds = [[0.002, 0.002]]
        cgs = np.zeros((1, n_gates), dtype=float)
        cgs[0, 0] = 0.002
        cgs[0, 1] = 0.002
        cgs[0, 2 : 2 + n_barriers] = 0.003
        cgs[0, -1] = 0.10

        for implementation in ("rust", "default"):
            try:
                return ChargeSensedDotArray(
                    Cdd=cdd,
                    Cgd=cgd.tolist(),
                    Cds=cds,
                    Cgs=cgs.tolist(),
                    T=50.0,
                    coulomb_peak_width=0.8,
                    algorithm="default",
                    implementation=implementation,
                )
            except Exception:
                continue
        return None

    @property
    def uses_qarray(self) -> bool:
        return self._qarray_model is not None

    @property
    def effective_barrier_exponent_matrix(self) -> np.ndarray:
        """Effective Γ matrix used by the simulator after locality enforcement."""
        return self._effective_gamma.copy()

    def generate_campaign(
        self,
        barrier_compensation_mapping: Mapping[str, Sequence[str]],
    ) -> Tuple[Dict[str, xr.Dataset], Dict[str, Dict[str, float]]]:
        """Generate ``ds_raw_all`` style pair scans and truth metadata."""
        ds_raw_all: Dict[str, xr.Dataset] = {}
        truth: Dict[str, Dict[str, float]] = {}
        for target_barrier, drive_barriers in barrier_compensation_mapping.items():
            for drive_barrier in drive_barriers:
                pair_key = f"{target_barrier}_vs_{drive_barrier}"
                ds_pair, pair_truth = self.generate_pair_scan(target_barrier, drive_barrier)
                ds_raw_all[pair_key] = ds_pair
                truth[pair_key] = pair_truth
        return ds_raw_all, truth

    def generate_campaign_from_pair_resolution(
        self,
        pair_resolution: Sequence[Mapping[str, object]],
    ) -> Tuple[Dict[str, xr.Dataset], Dict[str, Dict[str, float]]]:
        """Generate synthetic scans from pair-driven topology metadata.

        Each entry is expected to expose:
        - ``target_barrier``
        - ``drive_barriers`` (sequence)
        """
        mapping: Dict[str, Sequence[str]] = {}
        for entry in pair_resolution:
            target = str(entry["target_barrier"])
            drives = [str(name) for name in list(entry["drive_barriers"])]
            mapping[target] = drives
        return self.generate_campaign(mapping)

    def generate_pair_scan(self, target_barrier: str, drive_barrier: str) -> Tuple[xr.Dataset, Dict[str, float]]:
        """Generate one synthetic 2D scan for a target-drive barrier pair."""
        if target_barrier not in self._index:
            raise KeyError(f"Unknown target barrier '{target_barrier}'.")
        if drive_barrier not in self._index:
            raise KeyError(f"Unknown drive barrier '{drive_barrier}'.")

        i = self._index[target_barrier]
        j = self._index[drive_barrier]
        x = self.config.drive_values
        y = self.config.detuning_values
        offsets = np.asarray(self.config.barrier_dc_offsets, dtype=float)

        gamma_row = np.asarray(self._effective_gamma[i, :], dtype=float)
        gamma_ij = float(gamma_row[j])
        t0_i = float(self.config.base_tunnel_couplings[i])
        base_exponent = float(np.dot(gamma_row, offsets))
        tunnel_at_zero = t0_i * np.exp(base_exponent)
        slope_at_zero = gamma_ij * tunnel_at_zero

        # Eq. (2): t_i = t0_i * exp(sum_j Gamma_ij * dB_j), with one swept drive
        # and all other barriers held at their configured offsets.
        dB_matrix = np.repeat(offsets[None, :], x.size, axis=0)
        dB_matrix[:, j] = offsets[j] + x
        exponent_vs_drive = dB_matrix @ gamma_row
        if self.config.linearize_exponential_locally:
            tunnel_vs_drive = tunnel_at_zero + slope_at_zero * x
        else:
            tunnel_vs_drive = t0_i * np.exp(exponent_vs_drive)
        tunnel_vs_drive = np.clip(tunnel_vs_drive, 1e-6, None)

        primary_signal = np.empty((x.size, y.size), dtype=float)
        for row, drive in enumerate(x):
            center = self.config.detuning_center + self.config.detuning_center_drive_factor * float(drive)
            transition = finite_temperature_excess_charge(
                y,
                tunnel_coupling=tunnel_vs_drive[row],
                thermal_energy=self.config.thermal_energy,
                center=center,
            )
            detuning_centered = y - center
            if self.config.use_paper_signal_model:
                v0 = float(self.config.paper_signal_v0)
                delta_v = float(self.config.paper_signal_delta_v)
                s0 = float(self.config.paper_signal_s0)
                s1 = float(self.config.paper_signal_s1)
                primary_signal[row, :] = v0 + delta_v * transition + (
                    s0 + (s1 - s0) * transition
                ) * detuning_centered
            else:
                primary_signal[row, :] = self.config.signal_offset + self.config.signal_scale * transition

        qarray_bg = self._qarray_background(x, y, drive_idx=j, target_idx=i)
        analytic_bg = self._analytic_background(x, y, drive_idx=j, target_idx=i)
        pair_rng = np.random.default_rng(self._pair_seed(target_barrier, drive_barrier))
        noise = pair_rng.normal(0.0, self.config.noise_std, size=primary_signal.shape)

        amplitude = (
            primary_signal
            + self.config.qarray_background_weight * qarray_bg
            + self.config.analytic_background_weight * analytic_bg
            + noise
        )

        x_norm = (x - np.mean(x)) / max(float(np.ptp(x)), 1e-12)
        y_norm = (y - np.mean(y)) / max(float(np.ptp(y)), 1e-12)
        phase = 0.08 * x_norm[:, None] - 0.05 * y_norm[None, :]
        i_data = amplitude * np.cos(phase)
        q_data = amplitude * np.sin(phase)

        ds = xr.Dataset(
            data_vars={
                "I": (("x_volts", "y_volts"), i_data),
                "Q": (("x_volts", "y_volts"), q_data),
                "amplitude_truth": (("x_volts", "y_volts"), amplitude),
                "tunnel_truth": (("x_volts",), tunnel_vs_drive),
            },
            coords={
                "x_volts": xr.DataArray(x, dims="x_volts", attrs={"long_name": drive_barrier, "units": "V"}),
                "y_volts": xr.DataArray(y, dims="y_volts", attrs={"long_name": target_barrier, "units": "V"}),
            },
            attrs={
                "target_barrier": target_barrier,
                "drive_barrier": drive_barrier,
                "uses_qarray_background": self.uses_qarray,
            },
        )

        truth = {
            "target_barrier": target_barrier,
            "drive_barrier": drive_barrier,
            "gamma_ij": gamma_ij,
            "t0_i": t0_i,
            "dt_dB_at_zero": slope_at_zero,
            "uses_paper_signal_model": float(self.config.use_paper_signal_model),
            "uses_qarray_background": float(self.uses_qarray),
        }
        return ds, truth

    def _qarray_background(self, drive_values, detuning_values, drive_idx: int, target_idx: int) -> np.ndarray:
        if self._qarray_model is None:
            return np.zeros((drive_values.size, detuning_values.size), dtype=float)

        n_barriers = len(self.barrier_names)
        n_gates = 2 + n_barriers + 1
        vg = np.zeros((drive_values.size, detuning_values.size, n_gates), dtype=float)

        vg[:, :, 0] = 0.55 * detuning_values[None, :]
        vg[:, :, 1] = -0.55 * detuning_values[None, :]

        barrier_offset = 2
        vg[:, :, barrier_offset + drive_idx] = drive_values[:, None]
        vg[:, :, barrier_offset + target_idx] += 0.20 * detuning_values[None, :]

        for barrier in range(n_barriers):
            vg[:, :, barrier_offset + barrier] += 0.03 * drive_values[:, None] * (1.0 / (1.0 + abs(barrier - drive_idx)))

        try:
            sensor_output = self._qarray_model.charge_sensor_open(vg)
            if isinstance(sensor_output, tuple):
                sensor_output = sensor_output[0]
            sensor_output = np.asarray(sensor_output, dtype=float)
            if sensor_output.ndim == 3:
                sensor_output = sensor_output[:, :, 0]
            return _normalize_map(sensor_output)
        except Exception:
            return np.zeros((drive_values.size, detuning_values.size), dtype=float)

    @staticmethod
    def _analytic_background(drive_values, detuning_values, drive_idx: int, target_idx: int) -> np.ndarray:
        x = np.asarray(drive_values, dtype=float)
        y = np.asarray(detuning_values, dtype=float)
        x_norm = (x - np.mean(x)) / max(float(np.ptp(x)), 1e-12)
        y_norm = (y - np.mean(y)) / max(float(np.ptp(y)), 1e-12)

        wave = np.sin(2.0 * np.pi * (y_norm[None, :] + 0.15 * target_idx))
        envelope = np.cos(np.pi * (x_norm[:, None] + 0.10 * drive_idx))
        gradient = 0.3 * x_norm[:, None] - 0.15 * y_norm[None, :]
        return _normalize_map(wave * envelope + gradient)

    def _pair_seed(self, target_barrier: str, drive_barrier: str) -> int:
        token = f"{self.config.random_seed}:{target_barrier}:{drive_barrier}".encode("utf-8")
        digest = hashlib.sha256(token).digest()[:8]
        return int.from_bytes(digest, "little", signed=False)

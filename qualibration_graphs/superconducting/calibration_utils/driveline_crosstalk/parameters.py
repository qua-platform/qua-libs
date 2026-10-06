"""Parameters for the three independent crosstalk calibrations."""

from pathlib import Path

import numpy as np
from pydantic import Field
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters

MATRIX_DIRECTORY = Path(__file__).resolve().parents[2] / "calibrations" / "QEC" / "crosstalktest"


class NodeSpecificParameters(RunnableParameters):
    """Pulse, sweep, and matrix-storage settings shared by the three nodes."""

    qubits: list[str] = Field(default_factory=lambda: ["qA2", "qA3", "qA4"])
    """Selected drive and target qubits, in matrix order."""

    operation: str = "saturation"
    """Positive square XY pulse used to probe drive-line crosstalk."""

    pulse_length_ns: int = Field(default=2000, ge=16, multiple_of=4)
    """Probe pulse duration in ns, on the 4 ns timing grid."""

    common_lo_frequency_hz: float = 5.5e9
    """Common drive LO frequency in Hz for phase-coherent compensation."""

    max_drive_if_in_mhz: float = Field(default=500, gt=0, le=500)
    """Maximum absolute drive intermediate frequency in MHz."""

    max_amp_factor: float = Field(default=0.8, gt=0, lt=1.99)
    """Upper endpoint of the inclusive source-amplitude-prefactor sweep."""

    amplitude_points: int = Field(default=101, ge=8)
    """Number of equally spaced amplitude points, including zero."""

    timeout: int = Field(default=300, ge=1, le=300)
    matrix_directory: str = str(MATRIX_DIRECTORY)
    """Three canonical H5 matrices; full records use the standard node storage."""


class AmplitudeNodeSpecificParameters(NodeSpecificParameters):
    """Power-Rabi acquisition and fit settings."""

    num_shots: int = Field(default=150, ge=1)
    """Number of averages per power-Rabi sweep point."""

    self_drive_amp_scale: float = Field(default=0.025, gt=0, le=1)
    """Additional amplitude scale applied to diagonal self-drive traces."""

    min_fit_r2: float = Field(default=0.5, ge=0, le=1)
    """Minimum R-squared required to accept an oscillation fit."""


class PhaseNodeSpecificParameters(NodeSpecificParameters):
    """Coarse and fine compensation-phase sweep settings."""

    num_shots: int = Field(default=20, ge=1)
    """Number of averages per phase/amplitude point."""

    num_phases: int = Field(default=64, ge=8)
    """Full-period coarse sweep, followed by a narrow sweep around its minimum."""


class CheckNodeSpecificParameters(NodeSpecificParameters):
    """Independent cancellation-validation settings."""

    num_shots: int = Field(default=400, ge=1)
    """Number of averages per independent validation point."""

    max_mean_residual: float = Field(default=0.03, gt=0, lt=1)
    """Maximum absolute mean residual excited-state population."""

    max_residual_fraction: float = Field(default=0.1, gt=0, lt=1)
    """Require small absolute residual and at least 90% suppression when resolvable."""


class CommonParameters(NodeParameters, CommonNodeParameters, NodeSpecificParameters, QubitsExperimentNodeParameters):
    """Combined parameters shared by the drive-line crosstalk nodes."""

    timeout: int = Field(default=300, ge=1, le=300)
    """Session timeout in seconds, bounded by the cloud execution limit."""


class AmplitudeParameters(
    NodeParameters, CommonNodeParameters, AmplitudeNodeSpecificParameters, QubitsExperimentNodeParameters
):
    """Parameter set for 11c_driveline_crosstalk_amplitude."""

    timeout: int = Field(default=300, ge=1, le=300)
    """Session timeout in seconds, bounded by the cloud execution limit."""


class PhaseParameters(
    NodeParameters, CommonNodeParameters, PhaseNodeSpecificParameters, QubitsExperimentNodeParameters
):
    """Parameter set for 11d_driveline_crosstalk_phase."""

    timeout: int = Field(default=300, ge=1, le=300)
    """Session timeout in seconds, bounded by the cloud execution limit."""


class CheckParameters(
    NodeParameters, CommonNodeParameters, CheckNodeSpecificParameters, QubitsExperimentNodeParameters
):
    """Parameter set for 11e_driveline_crosstalk_compensation_check."""

    timeout: int = Field(default=300, ge=1, le=300)
    """Session timeout in seconds, bounded by the cloud execution limit."""


def amplitude_factors(parameters: NodeSpecificParameters) -> np.ndarray:
    """Return the inclusive drive-amplitude-prefactor sweep."""
    return np.linspace(0, parameters.max_amp_factor, parameters.amplitude_points)

from typing import Optional

from pydantic import Field
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    operation: str = "saturation"
    """Square XY operation played on the drive line and on the compensation line, stretched to pulse_length_ns. QUA
    scales a pulse by less than 2, so its amplitude must be above max_drive_amplitude / 2: the default 'saturation'
    operation is a square pulse with a large amplitude. Default is 'saturation'."""
    pulse_length_ns: int = Field(default=2000, ge=16, multiple_of=4)
    """Probe pulse duration in ns. Long pulses resolve weak crosstalk. Default is 2000 ns."""
    max_drive_amplitude: float = Field(default=0.5, gt=0, lt=1)
    """Maximum amplitude played on the drive line, as a fraction of the full scale of its output. It is the same for
    every drive line, so that weak crosstalk is resolved regardless of the amplitude of the operation.
    Default is 0.5."""
    amplitude_points: int = Field(default=101, ge=16)
    """Number of points of the drive amplitude sweep, from zero to its maximum. Default is 101."""
    self_drive_amp_scale: Optional[float] = Field(default=None, gt=0, le=1)
    """Maximum self-drive (diagonal) amplitude, as a fraction of max_drive_amplitude. If None, it is chosen such that
    the sweep spans about 1.5 Rabi periods, using the pulse area of the calibrated 'x180' operation. Default is None."""
    num_shots: int = Field(default=150, ge=1)
    """Number of averages for the amplitude (power Rabi) sweep. Default is 150."""
    num_shots_phase: int = Field(default=20, ge=1)
    """Number of averages for the coarse and fine compensation-phase sweeps. Default is 20."""
    num_shots_check: int = Field(default=400, ge=1)
    """Number of averages for the compensation check. Default is 400."""
    num_phases: int = Field(default=64, ge=8)
    """Number of points of the coarse (full period) and fine compensation-phase sweeps. Default is 64."""
    max_drive_if_in_mhz: float = Field(default=400, gt=0, le=500)
    """Maximum |IF| allowed when the drive line plays at the target frequency. Pairs above it are skipped.
    Default is 400 MHz."""
    min_fit_r2: float = Field(default=0.5, ge=0, le=1)
    """Minimum R-squared of the power Rabi fit to accept an amplitude ratio. Default is 0.5."""
    max_mean_residual: float = Field(default=0.03, gt=0, lt=1)
    """Maximum mean residual excited-state population after compensation. Default is 0.03."""
    max_residual_fraction: float = Field(default=0.1, gt=0, lt=1)
    """Maximum residual excitation relative to the uncompensated excitation. Default is 0.1."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass

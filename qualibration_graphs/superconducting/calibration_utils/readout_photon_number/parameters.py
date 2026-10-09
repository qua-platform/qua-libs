"""Matched square-drive sweeps for CKP and Ramsey photon calibration."""

import numpy as np
from pydantic import Field, model_validator
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = Field(default=200, ge=10)
    operation: str = "readout_square"
    """Square resonator pulse; the amplitude factor multiplies its calibrated voltage."""
    amp_factors: list[float] = Field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6])
    drive_detuning_mhz: float = 0.0
    """Drive frequency relative to the calibrated resonator frequency."""
    ringdown_wait_ns: int = Field(default=3000, ge=16)
    min_contrast: float = Field(default=0.08, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_sweeps(self):
        amps = np.asarray(self.amp_factors)
        if len(amps) < 3 or not np.all(np.isfinite(amps)) or amps[0] != 0 or np.any(np.diff(amps) <= 0):
            raise ValueError("amp_factors must start at zero and contain at least three increasing finite values")
        if amps[-1] >= 2:
            raise ValueError("Amplitude factors must be smaller than 2 (QUA fixed-point amplitude range)")
        timing = [value for key, value in self.model_dump().items() if key.endswith("_ns")]
        if any(value % 4 for value in timing):
            raise ValueError("All timing parameters must be divisible by 4 ns")
        return self


class CKPSpecificParameters(NodeSpecificParameters):
    max_shots_per_batch: int = Field(default=10, ge=1)
    """Bound each cloud request; batches are averaged with equal shot counts."""
    resonator_span_mhz: float = Field(default=4.0, gt=0)
    resonator_step_mhz: float = Field(default=0.2, gt=0)
    min_qubit_detuning_mhz: float = -40.0
    max_qubit_detuning_mhz: float = 10.0
    qubit_step_mhz: float = Field(default=0.5, gt=0)
    ringup_ns: int = Field(default=2400, ge=16)
    probe_ns: int = Field(default=100, ge=16)
    probe_operation: str = "x180_Square"
    reference_operation: str = "x180"
    """Calibrated pi gate whose waveform area sets the square probe voltage."""
    probe_area_factor: float = Field(default=0.8, gt=0, le=1)
    min_line_contrast: float = Field(default=0.08, gt=0, lt=1)


class RamseySpecificParameters(NodeSpecificParameters):
    ckp_data_id: int | None = None
    """Successful 23f CKP run supplying measured chi, kappa and the comparison curve."""
    min_duration_ns: int = Field(default=16, ge=16)
    max_duration_ns: int = Field(default=512, ge=32)
    duration_step_ns: int = Field(default=16, ge=4)
    num_frame_rotations: int = Field(default=12, ge=5)
    post_drive_idle_ns: int = Field(default=3000, ge=16)
    max_photon_number: float = Field(default=100.0, gt=0)


class CKPParameters(NodeParameters, CommonNodeParameters, CKPSpecificParameters, QubitsExperimentNodeParameters):
    """Parameters for 23f_readout_ckp."""


class RamseyParameters(NodeParameters, CommonNodeParameters, RamseySpecificParameters, QubitsExperimentNodeParameters):
    """Parameters for 23e_readout_photon_ramsey."""


def sweep_values(start, stop, step):
    """Build an inclusive sweep with at least five points."""
    if stop <= start:
        raise ValueError("Sweep stop must exceed start")
    values = np.arange(start, stop + step / 2, step)
    if values.size < 5:
        raise ValueError("At least five sweep points are required")
    return values

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from calibration_utils.common_utils.experiment import BaseExperimentNodeParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.run_video_mode.video_mode_specific_parameters import (
    VideoModeCommonParameters,
)

from typing import List, Optional


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    elements: List[str] = None
    """The element which the fast line is connected to. Can be a QuantumDot, BarrierGate or SensorDot."""
    sensor_names: Optional[List[str]] = None
    """The list of sensor dot names to be included in the measurement."""
    square_wave_frequency_start_MHz: float = 0.01
    """The starting frequency of the square wave in MHz."""
    square_wave_frequency_stop_MHz: float = 20
    """The ending frequency of the square wave in MHz."""
    square_wave_frequency_step_MHz: float = 0.1
    """The frequency step of the square wave in MHz."""
    square_wave_amplitude: float = 0.1
    """The amplitude of the square wave applied to the fast line."""
    estimated_bias_tee_tau_ns: Optional[float] = None
    """Estimated bias tee time constant in ns. Used as the initial guess for the
    high-pass fit and as the simulated τ when generating synthetic data.
    If None, defaults to 320 ns (f_c ~ 500 kHz)."""
    use_simulated_data: bool = False
    """Whether to generate simulated data rather than measuring via the OPX."""


class Parameters(
    NodeParameters,
    BaseExperimentNodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass

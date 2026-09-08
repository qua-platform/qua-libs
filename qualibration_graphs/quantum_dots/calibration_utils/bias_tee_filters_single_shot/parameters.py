from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from calibration_utils.common_utils.experiment import BaseExperimentNodeParameters
from qualibration_libs.parameters import CommonNodeParameters
from typing import List, Optional


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    elements: List[str] = None
    """The element which the fast line is connected to. Can be a QuantumDot, BarrierGate or SensorDot."""
    sensor_names: Optional[List[str]] = None
    """The list of sensor dot names to be included in the measurement."""
    step_amplitude: float = 0.1
    """The step size on the element."""
    measurement_time: int = 100000
    """The measurement time of the sensor."""
    integration_time: int = 1000
    """How much time to integrate to a single data point. Sliced demodulation will be used."""
    wait_time_after_pulse: int = 1000
    """The wait time after the pulse is played, before the measurement is started."""
    reset_wait_time: int = 100
    """Wait time at the start of each averaging loop."""
    estimated_bias_tee_tau_ns: Optional[float] = None
    """Estimated bias tee time constant in ns. Used as the initial guess for the
    exponential fit and as the simulated τ when generating synthetic data.
    If None, defaults to 20000 ns (20 µs)."""
    use_simulated_data: bool = False
    """Whether to generate simulated data rather than measuring via the OPX."""


class Parameters(
    NodeParameters,
    BaseExperimentNodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass

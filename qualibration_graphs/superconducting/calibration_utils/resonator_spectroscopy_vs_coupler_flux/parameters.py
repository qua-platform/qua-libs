"""Parameter definitions for resonator spectroscopy versus coupler flux calibration."""

from typing import ClassVar, Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for resonator spectroscopy versus coupler flux."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    min_flux_offset_in_v: float = -0.5
    """Minimum flux bias offset in volts. Default is -0.5 V."""
    max_flux_offset_in_v: float = 0.5
    """Maximum flux bias offset in volts. Default is 0.5 V."""
    num_flux_points: int = 101
    """Number of flux points. Default is 101."""
    frequency_span_in_mhz: float = 15
    """Frequency span in MHz. Default is 15 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Frequency step in MHz. Default is 0.1 MHz."""
    input_line_impedance_in_ohm: float = 50
    """Input line impedance in ohms. Default is 50 Ohm."""
    line_attenuation_in_db: float = 0
    """Line attenuation in dB. Default is 0 dB."""
    settle_time_in_ns: int = 20000
    """Settle time in ns. Default is 20000 ns."""
    measure_qubit: Literal["control", "target"] = "target"
    """Which qubit to measure: 'control' or 'target'. Default is 'target'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for resonator spectroscopy versus coupler flux calibration."""

    targets_name: ClassVar[str] = "qubit_pairs"

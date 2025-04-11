from typing import Literal

import numpy as np
from pydantic import model_validator
from qualang_tools.units import unit
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters

from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class ReadoutOptimization3dParameters(RunnableParameters):
    num_runs: int = 100
    """Number of runs to perform. Default is 100."""
    frequency_span_in_mhz: float = 10
    """Span of frequencies to sweep in MHz. Default is 10 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""
    min_amplitude_factor: float = 0.5
    """Minimum amplitude factor. Default is 0.5."""
    max_amplitude_factor: float = 1.99
    """Maximum amplitude factor. Default is 1.99."""
    num_amplitudes: int = 10
    """Number of amplitudes to sweep. Default is 10."""
    max_duration_in_ns: int = 4000
    """Maximum duration in ns. Default is 4000."""
    num_durations: int = 8
    """Number of durations to sweep. Default is 8."""
    plotting_dimension: Literal["2D", "3D"] = "2D"
    """Plotting dimension. Default is "2D"."""
    fidelity_smoothing_intensity: float = 0.5
    """Fidelity smoothing intensity. Default is 0.5."""
    max_readout_amplitude: float = 0.125
    """Maximum readout amplitude. Default is 0.125."""

    @model_validator(mode="after")
    def check_plot_type_is_2d_or_3d(self):
        if self.plotting_dimension not in ["2D", "3D"]:
            raise ValueError(f"Expected plot dimension to be '2D' or '3D', got {self.plotting_dimension}")

        return self

    @model_validator(mode="after")
    def check_durations_are_divisible_by_4(self):
        if self.max_duration_in_ns / self.num_durations % 4 != 0:
            raise ValueError(
                f"Expected readout segment length to be disvisible by 4, got "
                f"{self.max_duration_in_ns} / {self.num_durations} = "
                f"{self.max_duration_in_ns / self.num_durations}"
            )
        return self


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    ReadoutOptimization3dParameters,
    QubitsExperimentNodeParameters,
):
    pass


def get_frequency_detunings_in_hz(node_parameters: ReadoutOptimization3dParameters):
    u = unit(coerce_to_integer=True)

    span = node_parameters.frequency_span_in_mhz * u.MHz
    step = node_parameters.frequency_step_in_mhz * u.MHz

    dfs = np.arange(-span / 2, +span / 2, step)

    return dfs


def get_amplitude_factors(node_parameters: ReadoutOptimization3dParameters):
    amps = np.linspace(
        start=node_parameters.min_amplitude_factor,
        stop=node_parameters.max_amplitude_factor,
        num=node_parameters.num_amplitudes,
    )

    return amps


def get_durations(node_parameters: ReadoutOptimization3dParameters):
    durations = np.linspace(
        start=0,
        stop=node_parameters.max_duration_in_ns,
        num=node_parameters.num_durations + 1,
    )[1:]

    return durations

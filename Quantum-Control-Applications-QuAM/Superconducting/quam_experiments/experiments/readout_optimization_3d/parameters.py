from typing import Literal

import numpy as np
from pydantic import model_validator, Field
from qualang_tools.units import unit
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters

from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class ReadoutOptimization3dParameters(RunnableParameters):
    num_runs: int = 100
    frequency_span_in_mhz: float = 10
    frequency_step_in_mhz: float = 0.1
    min_amplitude_factor: float = 0.5
    max_amplitude_factor: float = 1.99
    num_amplitudes: int = 10
    max_duration_in_ns: int = 4000
    num_durations: int = 8
    plotting_dimension: Literal["2D", "3D"] = "2D"
    fidelity_smoothing_intensity: float = 0.5
    max_readout_amplitude: float = Field(
        0.125,
        description="upper limit for readout pulse amplitude to " "avoid saturation when doing multiplexed readout",
    )

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

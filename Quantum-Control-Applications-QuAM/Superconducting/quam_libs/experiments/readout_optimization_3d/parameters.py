from typing import Literal

import numpy as np
from qualang_tools.units import unit
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters

from quam_libs.experiments.node_parameters import (
    QubitsExperimentNodeParameters,
    SimulatableNodeParameters,
    FluxControlledNodeParameters,
    MultiplexableNodeParameters,
    DataLoadableNodeParameters,
    QmSessionNodeParameters
)


class ReadoutOptimization3dParameters(RunnableParameters):
    num_averages: int = 100
    frequency_span_in_mhz: float = 10
    frequency_step_in_mhz: float = 0.1
    min_amplitude_factor: float = 0.5
    max_amplitude_factor: float = 1.99
    num_amplitudes: int = 10
    min_duration_in_ns: int = 500
    max_duration_in_ns: int = 4000
    num_durations: int = 8


class Parameters(
    NodeParameters,
    SimulatableNodeParameters,
    DataLoadableNodeParameters,
    QmSessionNodeParameters,
    ReadoutOptimization3dParameters,
    FluxControlledNodeParameters,
    MultiplexableNodeParameters,
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
        num=node_parameters.num_amplitudes
    )

    return amps


def get_durations(node_parameters: ReadoutOptimization3dParameters):
    durations = np.linspace(
        start=0,
        stop=node_parameters.max_duration_in_ns,
        num=node_parameters.num_durations + 1
    )[1:]

    return durations
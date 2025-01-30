from typing import Literal

import numpy as np
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


class RamseyParameters(RunnableParameters):
    num_averages: int = 100
    frequency_detuning_in_mhz: float = 1.0
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 3000
    num_time_points: int = 500
    log_or_linear_sweep: Literal["log", "linear"] = "log"
    use_state_discrimination: bool = False

class Parameters(
    NodeParameters,
    SimulatableNodeParameters,
    DataLoadableNodeParameters,
    QmSessionNodeParameters,
    RamseyParameters,
    FluxControlledNodeParameters,
    MultiplexableNodeParameters,
    QubitsExperimentNodeParameters,
):

    def get_idle_times_in_clock_cycles(self) -> np.ndarray:
        """
        Get the idle-times sweep axis according to the sweep type.

        The dephasing time sweep is in units of clock cycles (4ns).
        The minimum is 4 clock cycles.
        """
        if self.log_or_linear_sweep == "linear":
            idle_times = self._get_idle_times_linear_sweep_in_clock_cycles()
        elif self.log_or_linear_sweep == "log":
            idle_times = self._get_idle_times_log_sweep_in_clock_cycles()
        else:
            raise ValueError(f"Expected sweep type to be 'log' or 'linear', got {self.log_or_linear_sweep}")

        return idle_times

    def _get_idle_times_linear_sweep_in_clock_cycles(self):
        return (
            np.linspace(
                self.min_wait_time_in_ns,
                self.max_wait_time_in_ns,
                self.num_time_points,
            ) // 4
        ).astype(int)

    def _get_idle_times_log_sweep_in_clock_cycles(self):
        return np.unique(
            np.geomspace(
                self.min_wait_time_in_ns,
                self.max_wait_time_in_ns,
                self.num_time_points,
            ) // 4
        ).astype(int)

"""Parameter definitions for single-qubit gate set tomography experiments.

This module defines the parameters used for configuring GST experiments,
including circuit lengths, number of shots, and operation types.
"""

from typing import ClassVar, Literal

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for single-qubit GST experiments."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    max_length: int = 256
    """Maximum number of gates that appear in a repeated germ. Default is 256."""
    log_scale: bool = True
    """If True, use log-scale lengths: 1, 2, 4, 8, 16, ... up to max_length. Default is True."""
    delta_length: int = 20
    """Step between lengths in linear scale mode. Default is 20."""
    model: Literal["smq1Q_XY"] = "smq1Q_XY"
    """Model to use for the GST experiment. Default is "smq1Q_XY"."""
    # use_state_discrimination: bool = True
    # """Whether to use state discrimination for readout. Default is True."""
    # use_strict_timing: bool = False
    # """Use strict timing in the QUA program. Default is False."""
    # use_input_stream: bool = False
    # """Whether to use input streams for circuit execution. Default is False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameters for single-qubit GST experiments."""

    def get_lengths(self) -> list[int]:
        """Generate circuit depths based on the parameter configuration.

        Length *L* means at most L gates in the germ repetition.
        Germs consisting of N gates will be repeated at most L/N times.
        Germs that are longer than L will not feature in the GST sequences.
        This is the standard GST convention.

        - If ``log_scale`` is True, lengths follow a power-of-two progression:
          1, 2, 4, 8, 16, ... up to ``max_length``.
        - If ``log_scale`` is False, lengths are linearly spaced using
          ``delta_length`` until ``max_length``.  The first value
          is always set to 1.

        Returns
        -------
        list[int]
            Sorted circuit lengths.
        """
        if self.log_scale:
            lengths: list[int] = []
            current_length = 1
            while current_length <= self.max_length:
                lengths.append(current_length)
                current_length *= 2
            return lengths

        assert (
            self.max_length / self.delta_length
        ).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
        lengths = list(range(0, self.max_length + 1, self.delta_length))
        lengths[0] = 1
        return lengths
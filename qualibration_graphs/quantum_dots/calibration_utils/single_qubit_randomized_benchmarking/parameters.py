from typing import Optional
import numpy as np
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    use_state_discrimination: bool = True
    """Perform qubit state discrimination. Default is True."""
    use_strict_timing: bool = False
    """Use strict timing in the QUA program. Default is False."""
    num_random_sequences: int = 300
    """Number of random RB sequences. Default is 300."""
    num_shots: int = 10
    """Number of averages. Default is 10."""
    max_circuit_depth: int = 2048
    """Maximum circuit depth (number of Clifford gates). Default is 2048."""
    delta_clifford: int = 20
    """Delta clifford (number of Clifford gates between the RB sequences). Default is 20."""
    log_scale: bool = True
    """If True, use log scale depths: 1,2,4,8,16,32... up to max_circuit_depth. Default is True."""
    seed: Optional[int] = None
    """Seed for the random number generator. Default is None."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    def get_depths(self):
        """
        Generate an array of circuit depths based on the parameter configuration.

        This method produces a list of circuit depths depending on whether
        the `log_scale` flag is enabled:

        - If `log_scale` is True, depths follow a logarithmic progression:
          1, 2, 4, 8, 16, 32, ... up to `max_circuit_depth`.
        - If `log_scale` is False, depths are linearly spaced using
          `delta_clifford` until `max_circuit_depth`. The first value is
          always set to 1.

        Returns:
            numpy.ndarray: An array of circuit depths (integers).

        Raises:
            AssertionError: If `max_circuit_depth / delta_clifford` is not
            an integer when `log_scale` is False.

        Examples:
            >>> params = Parameters(log_scale=True, max_circuit_depth=32)
            >>> params.get_depths()
            array([ 1,  2,  4,  8, 16, 32])

            >>> params = Parameters(log_scale=False,
            ...                     max_circuit_depth=10,
            ...                     delta_clifford=2)
            >>> params.get_depths()
            array([ 1,  2,  4,  6,  8, 10])
        """
        # Generate depth list based on log_scale parameter
        if self.log_scale:
            # Log scale: 1, 2, 4, 8, 16, 32, ... up to max_circuit_depth
            depths = [1]  # Start with depth 1
            current_depth = 2
            while current_depth <= self.max_circuit_depth:
                depths.append(current_depth)
                current_depth *= 2
            depths = np.array(depths)
        else:
            # Linear scale using delta_clifford
            assert (
                self.max_circuit_depth / self.delta_clifford
            ).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
            depths = np.arange(0, self.max_circuit_depth + 0.1, self.delta_clifford, dtype=int)
            depths[0] = 1  # Ensure we start with depth 1
        return depths

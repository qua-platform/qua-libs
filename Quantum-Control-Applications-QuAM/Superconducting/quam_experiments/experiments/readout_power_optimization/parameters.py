from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_runs: int = 2000
    """Number of runs to perform. Default is 2000."""
    start_amp: float = 0.5
    """Start amplitude. Default is 0.5."""
    end_amp: float = 1.99
    """End amplitude. Default is 1.99."""
    num_amps: int = 10
    """Number of amplitudes to sweep. Default is 10."""
    outliers_threshold: float = 0.98
    """Outliers threshold. Default is 0.98."""
    plot_raw: bool = False
    """Plot raw data. Default is False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass

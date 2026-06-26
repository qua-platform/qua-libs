"""Parameters for the CZ gate black-box optimisation node (04_cz_optimization).

The three-parameter search space optimises the voltage-balanced CZ macro:

    barrier_voltage  —  barrier gate voltage (V), passed as
                        point={barrier_gate: voltage} to the CZ macro.
    wait_duration    —  hold time at the exchange point (ns, multiple of 4).
    ramp_duration    —  ramp time between voltage levels (ns, multiple of 4).

Scoring
-------
A pluggable scoring function converts raw circuit measurement probabilities
into a scalar objective (higher is better).  The default is a dummy sum;
replace with a physics-informed cost function for production use.
"""

from __future__ import annotations

from qualibrate.core import NodeParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitPairExperimentNodeParameters,
)

from .parameters import CMAESParameters


class CZOptOptParameters(CMAESParameters):
    """Parameters specific to the CZ gate optimisation."""

    num_shots: int = 10
    """Number of shots per circuit per candidate. Default is 100."""

    include_quadrature_circuits: bool = False
    """If True, include circuits 4 and 5 (Y/X Ramsey variants) in addition
    to the 3 core circuits. Default is False."""

    # --- Barrier voltage bounds ---
    barrier_voltage_min: float = 0.0
    """Lower bound for barrier voltage (V). Default is 0.0."""
    barrier_voltage_max: float = 0.5
    """Upper bound for barrier voltage (V). Default is 0.5."""

    # --- Wait duration bounds (ns, must be multiple of 4) ---
    wait_duration_min: int = 16
    """Lower bound for wait duration (ns). Default is 16."""
    wait_duration_max: int = 2000
    """Upper bound for wait duration (ns). Default is 2000."""

    # --- Ramp duration bounds (ns, must be multiple of 4) ---
    ramp_duration_min: int = 16
    """Lower bound for ramp duration (ns). Default is 16."""
    ramp_duration_max: int = 200
    """Upper bound for ramp duration (ns). Default is 200."""

    # --- CMA-ES overrides ---
    population_size: int = 2
    sigma0: float = 0.15
    success_threshold: float = 0.0
    """Minimum score to consider optimisation successful. Default is 0.0
    (dummy scoring function always succeeds)."""

    action_sigma: float = 0.1
    """Standard deviation of the Gaussian exploration noise added to TD3
    actions (applied uniformly to all 3 search dimensions). Default is 0.1."""

    num_steps: int = 1000
    """Total number of timesteps for the TD3 training loop. Default is 1000."""

    seed: int = 42
    """Random seed for reproducibility. Default is 42."""

    timeout: int = 5000
    """Waiting time for OPX resources (in seconds). Default is 1000 s."""

    compilation_timeout: float = 5000.0
    """Timeout in seconds for the QMM gRPC API calls. Default is 1000 s."""


class CZOptParameters(
    NodeParameters,
    CommonNodeParameters,
    CZOptOptParameters,
    QubitPairExperimentNodeParameters,
):
    """Composite parameter set for 04_cz_optimization."""

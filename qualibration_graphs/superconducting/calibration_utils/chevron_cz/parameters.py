from types import SimpleNamespace
from typing import Callable, ClassVar, Optional

from qualang_tools.bakery import baking
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters

from calibration_utils.cz_iswap_flux_bootstrap.parameters import estimate_qubit_flux_shift


class NodeSpecificParameters(RunnableParameters):
    """Parameters for the CZ chevron calibration (node 31).

    Attributes
    ----------
    num_shots : int
        Number of averages per point.
    max_time_in_ns : int
        Maximum flux pulse duration in ns.
    amp_range : float
        Fractional half-range around the base amplitude (sweep covers
        ``[1 - amp_range, 1 + amp_range]`` times the base).
    amp_step : float
        Step size for the amplitude scaling sweep.
    use_state_discrimination : bool
        If True, use threshold-based state discrimination; otherwise raw IQ.
    use_saved_detuning : bool
        If True, read the CZ flux amplitude from ``qubit_pair.detuning``
        (written by node 30) instead of computing it from qubit frequencies.
        Use this for a fast re-run or when node 30 is not in the graph.
    operation : str
        Pair macro to calibrate and update (e.g. ``"cz_unipolar"``).
        Used when looking up the saved macro amplitude and in state update.
    """

    num_shots: int = 100
    max_time_in_ns: int = 160
    amp_range: float = 0.1
    amp_step: float = 0.003
    use_state_discrimination: bool = True
    use_saved_detuning: bool = False
    operation: str = "cz_unipolar"


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    targets_name: ClassVar[str] = "qubit_pairs"


def estimate_cz_flux_amplitude(
    parameters: NodeSpecificParameters,
    qp,
    log_callable: Optional[Callable[[str], None]] = None,
) -> float:
    """Estimate the CZ (11→20) flux-pulse base amplitude for the moving qubit.

    Thin wrapper around :func:`estimate_qubit_flux_shift` pinned to the CZ interaction.
    """
    cz_params = SimpleNamespace(use_saved_detuning=parameters.use_saved_detuning, cz_or_iswap="cz")
    return estimate_qubit_flux_shift(cz_params, qp, log_callable)


def baked_waveform(qubit, baked_config, base_level: float = 0.5, max_samples: int = 16):
    """Create truncated baked waveforms for the chevron CZ calibration.

    Generates a list of baking objects, each containing an incrementally longer flux pulse
    (1..max_samples samples) at the specified base_level. Each baked pulse is registered
    as an operation named "flux_pulse{i}" on the provided qubit z line.

    The coupler (when present) is intentionally NOT baked here. Baking both qubit z and
    coupler in the same context causes an internal ``align()`` inside ``run()`` that
    conflicts with ``strict_timing_()`` and produces timing gaps on the qubit element.
    Instead the coupler is played separately in the QUA program with 4 ns granularity,
    which is sufficient since the coupler only needs to hold a DC bias level.

    Args:
        qubit: The qubit object whose z line is used.
        baked_config: A mutable QM configuration object to which baked operations are added.
        base_level (float): The constant waveform level used for the short baked segments.
        max_samples (int): The maximum number of samples (and thus baked variants) to generate.

    Returns:
        List of baking objects, index i corresponds to pulse length i+1 samples.
    """
    pulse_segments = []
    waveform = [base_level] * max_samples
    for i in range(1, max_samples + 1):
        with baking(baked_config, padding_method="right") as b:
            wf = waveform[:i]
            b.add_op(f"flux_pulse{i}", qubit.z.name, wf)
            b.play(f"flux_pulse{i}", qubit.z.name)
        pulse_segments.append(b)
    return pulse_segments

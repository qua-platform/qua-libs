from typing import Dict, List, Literal, Optional

from calibration_utils.gate_virtualization.base_parameters import (
    GateVirtualizationBaseParameters,
)


class CrossCapacitance1DParameters(GateVirtualizationBaseParameters):
    """Parameters for OPX-only 1D cross-capacitance measurement via paired plunger sweeps.

    For each (target_plunger, perturbing_gate) pair, two 1D sweeps of the
    target plunger are performed: one at baseline and one with the perturbing
    gate stepped by ``step_voltage``.  The shift in charge transition position
    divided by the step voltage gives the cross-capacitance coefficient.

    All sweeps are executed in a single QUA program using OPX outputs.
    For QDAC-driven sweeps or perturbations, use the companion
    ``CrossCapacitance1DQdacParameters``.

    See Volk et al., npj Quantum Information (2019) 5:29, Supplementary Fig. S1.
    """

    cross_capacitance_mapping: Optional[Dict[str, List[str]]] = None
    """Mapping of target plunger gate -> list of perturbing gates whose
    cross-talk to this dot will be measured.
    Example: ``{"virtual_dot_1": ["virtual_dot_2", "barrier_12"]}``.
    If None, must be provided before running (automatic generation is not
    yet implemented)."""
    step_voltage: float = 0.010
    """Perturbation voltage (V) applied to the perturbing gate between the
    reference and shifted sweeps.  The paper uses 10 mV."""
    sweep_span: float = 0.050
    """Total voltage span of the 1D plunger sweep (V)."""
    sweep_points: int = 201
    """Number of points in each 1D plunger sweep."""
    update_mode: Literal["additive", "overwrite"] = "additive"
    """How to update the compensation matrix.
    ``"additive"`` adds the measured residual to the existing entry (suitable
    for iterative refinement).
    ``"overwrite"`` replaces the entry with the measured value (suitable for
    initial matrix population)."""

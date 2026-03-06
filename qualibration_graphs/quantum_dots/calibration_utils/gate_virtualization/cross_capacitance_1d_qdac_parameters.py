from typing import Dict, List, Literal, Optional

from calibration_utils.gate_virtualization.base_parameters import (
    GateVirtualizationBaseParameters,
)


class CrossCapacitance1DQdacParameters(GateVirtualizationBaseParameters):
    """Parameters for QDAC-capable 1D cross-capacitance measurement.

    Extends the base 1D cross-capacitance approach with QDAC support for
    the plunger sweep and/or the perturbing gate step.  Uses one QUA
    program per (target, perturbing) pair since QDAC voltages cannot be
    changed from within a QUA program.

    For OPX-only operation (single program), use the companion
    ``CrossCapacitance1DParameters`` with node ``04_1d_cross_capacitance``.

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
    sweep_from_qdac: bool = False
    """Whether the target plunger sweep is driven by the QDAC."""
    perturb_from_qdac: bool = False
    """Whether the perturbing gate step is applied via the QDAC."""
    update_mode: Literal["additive", "overwrite"] = "additive"
    """How to update the compensation matrix.
    ``"additive"`` adds the measured residual to the existing entry (suitable
    for iterative refinement).
    ``"overwrite"`` replaces the entry with the measured value (suitable for
    initial matrix population)."""

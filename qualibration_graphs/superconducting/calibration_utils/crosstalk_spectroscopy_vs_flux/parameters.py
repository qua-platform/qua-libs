"""Parameters for the crosstalk spectroscopy vs flux calibration node."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from qualibrate import NodeParameters, QualibrationNode
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Parameters specific to crosstalk spectroscopy vs flux calibration."""

    target_qubits: Optional[List[str]] = ["q1"]
    """Qubit(s) whose f_01 is measured while sweeping neighbor flux bias. Default is ['q1']."""
    aggressor_qubits: Optional[List[str]] = ["q2"]
    """Neighbor qubit(s) or coupler(s) whose flux bias is swept. Default is ['q2']."""
    measure_self: bool = True
    """Serial (T, T) self-calibration before cross-talk sweeps, using target sweep grids.
    If False, normalize using freq_vs_flux_01_quad_term from state instead. Default is True."""
    self_slope_tolerance: float = 0.2
    """Log a warning when |self_slope_measured / self_slope_model - 1| exceeds this value. Default is 0.2."""
    num_shots: int = 50
    """Number of averages per (frequency, flux) point. Default is 20."""
    operation: str = "saturation"
    """Qubit XY operation played during spectroscopy. Default is 'saturation'."""
    operation_amplitude_factor: float = 0.2
    """Scale factor applied to the configured operation amplitude. Default is 0.2."""
    operation_len_in_ns: Optional[int] = 50000
    """Duration of the spectroscopy pulse in ns. Default is 50000 ns."""
    target_qubit_frequency_span: float = 100
    """Frequency sweep span for self pair (T, T), in MHz. Default is 100 MHz."""
    aggressor_qubit_frequency_span: float = 100
    """Frequency sweep span for cross pairs (T, A), in MHz. Default is 100 MHz."""
    frequency_num_points: int = 51
    """Number of frequency points in the spectroscopy sweep. Default is 51."""
    target_flux_offset_span_in_v: float = 0.02
    """Flux pulse offset span for self pair (T, T), in V. Default is 0.02 V."""
    aggressor_flux_offset_span_in_v: float = 0.1
    """Flux pulse offset span on the aggressor for cross pairs (T, A), in V. Default is 0.1 V."""
    flux_num_points: int = 21
    """Number of flux bias points in the sweep. Default is 21."""
    flux_detuning_mode: Literal["auto_for_linear_response", "auto_fill_sweep_window", "manual"] = "manual"
    """Strategy for setting the target qubit's flux detuning before the 2D sweep.
    'manual' uses manual_flux_detuning_in_v; 'auto_fill_sweep_window' places the qubit so the
    expected frequency shift fills the aggressor flux/frequency sweep window; 'auto_for_linear_response'
    places the qubit for ~1% linearity over the aggressor flux sweep. Default is 'manual'."""
    manual_flux_detuning_in_v: float = 0.03
    """Fixed target-qubit flux detuning used when flux_detuning_mode is 'manual', in V. Default is 0.03 V."""
    expected_crosstalk: float = -0.2
    """Expected change in target flux per unit aggressor flux (dPhi_target/dPhi_aggressor).
    Used by auto flux-detuning modes to estimate the linear response region. Default is -0.2."""
    flux_pulse_padding_in_ns: float = 2000
    """Padding between flux pulse edges and the XY pulse, in ns. Also added to the flux pulse duration.
    Default is 2000 ns."""
    input_line_impedance_in_ohm: Optional[int] = 50
    """Input line impedance for amplitude scaling, in ohms. Default is 50 Ohm."""
    line_attenuation_in_db: Optional[int] = 0
    """Line attenuation for amplitude scaling, in dB. Default is 0 dB."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameters for the crosstalk spectroscopy vs flux calibration node."""


def build_crosstalk_pairs(
    node: QualibrationNode,
) -> Tuple[Dict[str, List[str]], Dict[str, list]]:
    """Build ordered (target, aggressor) pairs per target, matching stream panel order."""
    aggressor_names = list(node.parameters.aggressor_qubits or [])
    pairs_by_target: Dict[str, List[str]] = {}
    pairs_by_target_objs: Dict[str, list] = {}

    for target_name in node.parameters.target_qubits or []:
        if node.parameters.measure_self:
            pair_names = [target_name] + [name for name in aggressor_names if name != target_name]
        elif node.parameters.multiplexed:
            pair_names = list(aggressor_names)
        else:
            pair_names = [name for name in aggressor_names if name != target_name]
        pairs_by_target[target_name] = pair_names
        pairs_by_target_objs[target_name] = [node.machine.qubits[name] for name in pair_names]

    return pairs_by_target, pairs_by_target_objs

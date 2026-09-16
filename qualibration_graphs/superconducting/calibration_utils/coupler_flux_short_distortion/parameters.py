"""Parameter definitions for coupler flux short distortion (cryoscope)."""

from typing import ClassVar, List, Literal

from qualang_tools.bakery import baking
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


def baked_coupler_waveform(config, waveform_amp: float, coupler, max_length: int = 16):
    """Create baked pulse segments with 1ns granularity up to ``max_length`` ns for coupler flux."""
    pulse_segments = []
    waveform = [waveform_amp] * max_length
    for i in range(1, max_length + 1):
        with baking(config, padding_method="right") as b:
            wf = waveform[:i]
            b.add_op(f"coupler_flux_pulse{i}", coupler.name, wf)
            b.play(f"coupler_flux_pulse{i}", coupler.name)
        pulse_segments.append(b)
    return pulse_segments


class NodeSpecificParameters(RunnableParameters):
    """Specific parameters for coupler flux short distortion (cryoscope) characterization."""

    num_shots: int = 200
    """Number of averages to perform. Default is 200."""
    cryoscope_len: int = 240
    """Length of the cryoscope operation in nanoseconds. Default is 240."""
    num_frames: int = 17
    """Number of frames to use in the cryoscope experiment. Default is 17."""
    exponential_fit_time_fractions: List[float] = [0.5, 0.01]
    """List of time fractions for the exponential fit. Default is [0.5, 0.01]."""
    n_exponentials: int = 2
    """Number of exponential components in the IIR fit."""
    update_state: bool = False
    """Master gate for writing fitted filters into QUAM state."""
    update_state_from_GUI: bool = False
    """When re-analysing via ``load_data_id``, enable ``update_state`` from the GUI."""
    update_iir: bool = True
    """When ``update_state`` is set, append IIR taps to ``exponential_filter``."""
    update_fir: bool = False
    """When ``update_state`` is set, write ``feedforward_filter`` from FIR analysis."""
    measure_qubit: Literal["control", "target"] = "target"
    """Which qubit in the pair to measure: 'control' or 'target'. Default is 'target'."""

    detuning_in_mhz: float = 200.0
    """Signed detuning from the qubit frequency at the coupler's decouple_offset in MHz
    (positive = above, negative = below); used to pick the cryoscope flux amplitude."""

    freq_to_flux_source: Literal["auto", "spectroscopy", "ramsey"] = "auto"
    """How to pick the cryoscope coupler flux-pulse amplitude and freq→flux inversion.

    ``auto`` (default) plays ``coupler_flux_amplitude_in_v``; analysis still tries 03c
    then 09b when inverting frequency to flux.
    ``spectroscopy`` / ``ramsey`` invert the 03c / 09b curve for the pulse amplitude
    (run IDs from qubit extras — run those nodes with ``save_load_id=True`` first)."""

    coupler_flux_amplitude_in_v: float = 0.1
    """Coupler flux pulse amplitude in V. Used when ``freq_to_flux_source`` is ``auto``."""

    use_fir: bool = False
    """Run FIR analysis after IIR."""
    fir_max_taps: int = 48
    """Upper bound for forward and inverse FIR length."""
    debug_plots: bool = False
    """If True, show diagnostic figures: cryoscope frequency, unwrapped phase,
    freq-vs-flux curve, raw I vs frame, and all FIR diagnostic plots."""
    log_time_axis: bool = False
    """If True, plot flux response (and debug cryoscope frequency) vs log time; else linear."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for coupler flux short distortion calibration node."""

    targets_name: ClassVar[str] = "qubit_pairs"

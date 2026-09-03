"""Parameter definitions for cryoscope experiment."""

from typing import Literal

from qualang_tools.bakery import baking
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


def baked_waveform(config, waveform_amp: float, qubit, max_length: int = 16):
    """Create baked pulse segments with 1ns granularity up to ``max_length`` ns.

    This mirrors the previous inline implementation inside ``17c_qubit_flux_short_distortion.py`` and is
    extracted here so it can be shared / unit tested. Each index ``i`` (1..max_length)
    produces a baking object that plays a constant waveform of ``i`` ns with amplitude
    ``waveform_amp`` on the qubit flux line.

    Parameters
    ----------
    config : dict
        Configuration dictionary (typically produced by ``machine.generate_config()``)
        that the baking context mutates.
    waveform_amp : float
        The absolute amplitude to use for the flux pulse.
    qubit : Any
        QUAM qubit object containing the ``z`` element name.
    max_length : int, optional
        Maximum pulse length in ns to bake (default 16 to keep within baking memory limits).

    Returns
    -------
    list
        A list of baking objects; element ``i-1`` corresponds to a pulse of length ``i`` ns.
    """
    pulse_segments = []
    # Create the base waveform (1ns resolution). Represent as list of samples.
    waveform = [waveform_amp] * max_length
    for i in range(1, max_length + 1):  # inclusive
        with baking(config, padding_method="right") as b:
            wf = waveform[:i]
            b.add_op(f"flux_pulse{i}", qubit.z.name, wf)
            b.play(f"flux_pulse{i}", qubit.z.name)
        pulse_segments.append(b)
    return pulse_segments


class NodeSpecificParameters(RunnableParameters):
    """Cryoscope-specific parameters for flux line characterization."""

    num_shots: int = 5000
    """Number of averages to perform. Default is 5000."""
    reset_type: str = "active"
    """Type of reset to perform: 'active' or 'thermal'."""

    detuning_target_in_mhz: int = 300
    """Target detuning from sweetspot for the cryoscope pulse in MHz. Default is 300."""
    cryoscope_len: int = 240
    """Length of the cryoscope operation in nanoseconds. Default is 240."""
    num_frames: int = 17
    """Number of frames to use in the cryoscope experiment. Default is 17."""
    n_exponentials: int = 2
    """Number of exponential components in IIR to fit in the cryoscope flux step response model ``y(t) = a_dc + Σ a_i exp(-t/tau_i)``."""
    use_fir: bool = False
    """Run FIR analysis after IIR. Default False."""
    fir_max_taps: int = 48
    """Forward / inverse FIR length (fixed; no auto length search)."""

    update_iir: bool = False
    """Push IIR exponential filter into state on this run."""
    update_fir: bool = False
    """Push FIR feedforward filter into state on this run."""

    freq_to_flux_source: Literal["auto", "ramsey", "spectroscopy", "quad_term"] = "auto"
    """Which frequency->voltage relation to use when picking the cryoscope Z amplitude and when converting the measured f(t) into a flux step response. 'auto' (default) tries Ramsey vs flux (09a), then qubit spectroscopy vs flux (03b), then the quadratic freq_vs_flux_01_quad_term; the other values force one specific source. Run IDs are never entered by hand: they are read from each qubit's extras ('ramsey_vs_flux_calibration_load_id' for 09a, 'qubit_spectroscopy_vs_flux_load_id' for 03b), so run those nodes with save_load_id=True first."""
    debug_plots: bool = False
    """If True, also generate diagnostic figures (unwrapped phase, freq-vs-flux reference curve)."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameters for cryoscope calibration node."""

"""Flux-activated f-LRU: the 2 GS/s flux element a program needs to play ``qubit.flux_lru``.

The |f0> -> |e1> transition cannot be driven by the XY line: |f0> and |e1> have the same total
parity (transmon level + photon number), and a charge drive only couples states of opposite
parity. A flux drive conserves parity. Near the sweet spot the qubit frequency depends
quadratically on flux, so a flux tone at f_m modulates the transmon frequencies at 2 * f_m, and
the |f0> -> |e1> resonance is reached for 2 * f_m = f_r - f_12. The photon then leaks out of the
resonator and leaves the qubit in |e0>.

f_r - f_12 is about 1.5 GHz, so f_m is about 750 MHz: above the 500 MHz Nyquist frequency of an
LF-FEM port at 1 GS/s. ``add_flux_lru_elements`` therefore switches the qubit's flux port to 2 GS/s
in the generated config, and the QUAM state keeps it at 1 GS/s. The pulses the state builds for that
port (CZ, for instance) hold one sample per ns, so every element on the port gets 2 GS/s copies of its
pulses, upsampled by linear interpolation. Elements on other ports keep the 1 GS/s originals, so a
pulse they share is not touched.

Two port settings are not converted. FIR ``feedforward`` taps are defined per sample, so at 2 GS/s
they act on half the time span they were fitted on; a warning is logged when the port has them.
Exponential and high-pass filters are specified in time and are left as they are. No hardware check
of either at 2 GS/s has been made.

The tone adds to the flux offset, and the sum must stay within the port's output range: +-0.5 V in
``direct`` mode, +-2.5 V in ``amplified`` mode.
"""

import copy
import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from quam_config.lru_components import FLUX_LRU_OPERATION

__all__ = [
    "FLUX_LRU_MAX_TONE_HZ",
    "FLUX_LRU_OPERATION",
    "FLUX_LRU_SAMPLING_RATE",
    "FLUX_PORT_RANGE_V",
    "UPSAMPLED_SUFFIX",
    "add_flux_lru_elements",
    "flux_lru_element",
    "flux_tone_headroom_v",
    "upsample_linear",
]

logger = logging.getLogger(__name__)

#: LF-FEM sampling rate needed to synthesize the ~750 MHz flux tone.
FLUX_LRU_SAMPLING_RATE = 2e9
#: Highest flux-tone frequency allowed, in Hz, kept clear of the 1 GHz Nyquist limit at 2 GS/s.
FLUX_LRU_MAX_TONE_HZ = 0.95e9
#: Largest voltage an LF-FEM analog output can play, per output mode.
FLUX_PORT_RANGE_V = {"direct": 0.5, "amplified": 2.5}
#: Appended to the names of the 2 GS/s copies of a flux port's pulses and waveforms.
UPSAMPLED_SUFFIX = ".2GSPS"
#: Length of the flux-LRU pulse when no duration is given; a ``duration`` argument to ``play`` overrides it.
DEFAULT_FLUX_LRU_LENGTH_NS = 100


def flux_tone_headroom_v(qubit) -> float:
    """Largest flux-tone amplitude, in V, that fits on top of the qubit's joint flux offset.

    Programs set the joint flux point (``initialize_qpu``), so the tone adds to ``qubit.z.joint_offset``.
    """
    output_mode = getattr(qubit.z.opx_output, "output_mode", "direct")
    return FLUX_PORT_RANGE_V[output_mode] - abs(qubit.z.joint_offset)


def flux_lru_element(qubit) -> str:
    """Name of the qubit's flux-LRU element in a config from ``add_flux_lru_elements``."""
    return f"{qubit.name}.flux_lru"


def upsample_linear(samples: Sequence[float]) -> List[float]:
    """Samples at 2 GS/s for a waveform given at 1 GS/s: the originals with their midpoints in between.

    The last sample is held for its second half-ns, so the waveform keeps its length.
    """
    x = np.asarray(samples, dtype=float)
    y = np.empty(2 * len(x))
    y[0::2] = x
    y[1:-1:2] = (x[:-1] + x[1:]) / 2
    y[-1] = x[-1]
    return y.tolist()


def _upsampled_waveform(config: Dict, name: str) -> str:
    """Name of the 2 GS/s version of waveform ``name``, added to ``config`` if needed."""
    waveform = config["waveforms"][name]
    if waveform.get("type") != "arbitrary":
        return name  # constant waveforms do not depend on the sampling rate
    new_name = name + UPSAMPLED_SUFFIX
    if new_name not in config["waveforms"]:
        upsampled = copy.deepcopy(waveform)
        upsampled["samples"] = upsample_linear(waveform["samples"])
        config["waveforms"][new_name] = upsampled
    return new_name


def _upsampled_pulse(config: Dict, name: str) -> str:
    """Name of the 2 GS/s version of pulse ``name``, added to ``config`` if needed."""
    new_name = name + UPSAMPLED_SUFFIX
    if new_name not in config["pulses"]:
        pulse = copy.deepcopy(config["pulses"][name])
        pulse["waveforms"] = {key: _upsampled_waveform(config, wf) for key, wf in pulse.get("waveforms", {}).items()}
        config["pulses"][new_name] = pulse
    return new_name


def _switch_port_to_2gsps(config: Dict, port_address) -> None:
    """Set the port at ``port_address`` to 2 GS/s and point every element on it to 2 GS/s pulses."""
    controller, fem, port_id = port_address
    port = config["controllers"][controller]["fems"][fem]["analog_outputs"][port_id]
    if port.get("sampling_rate") == FLUX_LRU_SAMPLING_RATE:
        return
    port["sampling_rate"] = FLUX_LRU_SAMPLING_RATE
    port.pop("upsampling_mode", None)  # only defined at 1 GS/s
    if port.get("filter", {}).get("feedforward"):
        logger.warning(
            f"Flux port {tuple(port_address)} has FIR feedforward taps fitted at 1 GS/s; at 2 GS/s they act on "
            "half the time span. They are left as they are."
        )
    for element in config["elements"].values():
        if tuple(element.get("singleInput", {}).get("port", ())) != tuple(port_address):
            continue
        element["operations"] = {
            operation: _upsampled_pulse(config, pulse) for operation, pulse in element.get("operations", {}).items()
        }


def add_flux_lru_elements(
    config: Dict,
    qubits,
    amplitude_v: Optional[float] = None,
    tone_frequency: Optional[float] = None,
    length_ns: Optional[int] = None,
    check_headroom: bool = True,
) -> Dict:
    """Add, for each qubit, a flux-LRU element playing a constant tone on its flux port, switched to 2 GS/s.

    Values left as None come from ``qubit.flux_lru`` (node 26b's calibration); node 26b itself passes
    them all, since it sweeps them.

    Args:
        config: Config generated from the QUAM state; modified in place.
        qubits: Qubits whose flux port carries the tone.
        amplitude_v: Tone amplitude in volts at the port; ``FluxLru.play`` or ``amp()`` scales it.
        tone_frequency: Tone frequency (the element's IF) in Hz, half the |f0> -> |e1> frequency.
        length_ns: Pulse length in ns; a ``duration`` argument to ``play`` overrides it.
        check_headroom: Raise when the amplitude does not fit on top of the joint flux offset. Node 26b
            turns it off and checks the largest swept amplitude instead.

    Returns:
        The modified config.

    Raises:
        ValueError: when a value is neither given nor calibrated, the amplitude exceeds the port's
            headroom, or the tone is too close to the 1 GHz Nyquist limit.
    """
    for qubit in qubits:
        stored = getattr(qubit, "flux_lru", None)
        if stored is None and None in (amplitude_v, tone_frequency, length_ns):
            raise ValueError(f"{qubit.name} has no calibrated flux_lru: run node 26b first.")
        amplitude = stored.amplitude_v if amplitude_v is None else amplitude_v
        frequency = stored.tone_frequency if tone_frequency is None else tone_frequency
        length = stored.duration_ns if length_ns is None else length_ns
        if check_headroom and abs(amplitude) > flux_tone_headroom_v(qubit):
            raise ValueError(
                f"{qubit.name}: the {amplitude:.3f} V flux tone does not fit on top of the "
                f"{qubit.z.joint_offset:.3f} V flux offset (headroom {flux_tone_headroom_v(qubit):.3f} V)."
            )
        if abs(frequency) >= FLUX_LRU_MAX_TONE_HZ:
            raise ValueError(
                f"{qubit.name}: the flux tone at {frequency / 1e6:.0f} MHz is too close to the 1 GHz Nyquist limit."
            )

        z = config["elements"][qubit.z.name]
        _switch_port_to_2gsps(config, z["singleInput"]["port"])

        element = flux_lru_element(qubit)
        config["waveforms"][f"{element}.wf"] = {"type": "constant", "sample": float(amplitude)}
        config["pulses"][f"{element}.pulse"] = {
            "operation": "control",
            "length": int(length),
            "waveforms": {"single": f"{element}.wf"},
        }
        config["elements"][element] = {
            "singleInput": copy.deepcopy(z["singleInput"]),
            "intermediate_frequency": float(frequency),
            "operations": {FLUX_LRU_OPERATION: f"{element}.pulse"},
        }
    return config

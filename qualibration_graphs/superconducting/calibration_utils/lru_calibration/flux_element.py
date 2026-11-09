"""Flux-activated f-LRU: a modulated flux element on the qubit's flux port, added to the generated config only.

The |f0> -> |e1> transition cannot be driven by the XY line: |f0> and |e1> have the same total
parity (transmon level + photon number), and a charge drive only couples states of opposite
parity. A flux drive conserves parity. Near the sweet spot the qubit frequency depends
quadratically on flux, so a flux tone at f_m modulates the transmon frequencies at 2 * f_m, and
the |f0> -> |e1> resonance is reached for 2 * f_m = f_r - f_12. The photon then leaks out of the
resonator and leaves the qubit in |e0>.

f_r - f_12 is about 1.5 GHz, so f_m is about 750 MHz: above the 500 MHz Nyquist frequency of an
LF-FEM port at 1 GS/s. The flux port is therefore switched to 2 GS/s for this run. Pulses the
QUAM state builds at 1 GS/s on that port (CZ waveforms, for instance) do not match the new rate,
so the qubit's flux operations are left out of the run's config; only the static flux offset is
used. Pulses that other elements also play (CZ spectator pulses, for instance) stay in the config.
Nothing here changes the QUAM state.

The tone adds to the flux offset, and the sum must stay within the port's output range: +-0.5 V in
``direct`` mode, +-2.5 V in ``amplified`` mode.
"""

import copy
from typing import Dict

#: LF-FEM sampling rate needed to synthesize the ~750 MHz flux tone.
FLUX_LRU_SAMPLING_RATE = 2e9
#: Operation name of the flux-LRU tone.
FLUX_LRU_OPERATION = "flux_lru"
#: Largest voltage an LF-FEM analog output can play, per output mode.
FLUX_PORT_RANGE_V = {"direct": 0.5, "amplified": 2.5}


def flux_tone_headroom_v(qubit) -> float:
    """Largest flux-tone amplitude, in V, that fits on top of the qubit's joint flux offset.

    The node sets the joint flux point (``initialize_qpu``), so the tone adds to ``qubit.z.joint_offset``.
    """
    output_mode = getattr(qubit.z.opx_output, "output_mode", "direct")
    return FLUX_PORT_RANGE_V[output_mode] - abs(qubit.z.joint_offset)


def flux_lru_element(qubit) -> str:
    """Name of the qubit's flux-LRU element in the run's config."""
    return f"{qubit.name}.flux_lru"


def add_flux_lru_elements(config: Dict, qubits, amplitude_v: float, intermediate_frequency: float = 750e6) -> Dict:
    """Add, for each qubit, a flux-LRU element playing a constant ``amplitude_v`` tone on its flux port.

    Args:
        config: Config generated from the QUAM state; modified in place.
        qubits: Qubits whose flux port carries the tone.
        amplitude_v: Tone amplitude in volts at the port; the node scales it with ``amp()``.
        intermediate_frequency: Initial tone frequency; the node sets it with ``update_frequency``.

    Returns:
        The modified config.
    """
    for qubit in qubits:
        z_name = qubit.z.name
        z = config["elements"][z_name]
        controller, fem, port_id = z["singleInput"]["port"]
        port = config["controllers"][controller]["fems"][fem]["analog_outputs"][port_id]
        port["sampling_rate"] = FLUX_LRU_SAMPLING_RATE
        port.pop("upsampling_mode", None)  # only defined at 1 GS/s

        # The flux pulses were sampled at 1 GS/s; leave them out of this run. Other elements can play the
        # same pulses (CZ spectator pulses), so only the pulses no other element uses are removed.
        removed = set(z.get("operations", {}).values())
        z["operations"] = {}
        removed -= {p for e in config["elements"].values() for p in e.get("operations", {}).values()}
        waveforms = {w for pulse in removed for w in config["pulses"].pop(pulse, {}).get("waveforms", {}).values()}
        waveforms -= {w for pulse in config["pulses"].values() for w in pulse.get("waveforms", {}).values()}
        for waveform in waveforms:
            config["waveforms"].pop(waveform, None)

        element = flux_lru_element(qubit)
        config["waveforms"][f"{element}.wf"] = {"type": "constant", "sample": amplitude_v}
        config["pulses"][f"{element}.pulse"] = {
            "operation": "control",
            "length": 100,
            "waveforms": {"single": f"{element}.wf"},
        }
        config["elements"][element] = {
            "singleInput": copy.deepcopy(z["singleInput"]),
            "intermediate_frequency": intermediate_frequency,
            "operations": {FLUX_LRU_OPERATION: f"{element}.pulse"},
        }
    return config

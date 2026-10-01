"""
QUA macros for Use Case 3: Interleaved CZ 2Q RB.

Provides:
    cz_gate          — flux-based CZ gate via q1_z DC offset
    multiplexed_readout — simultaneous readout of rr1/rr2
    active_reset     — active qubit reset using conditional pi pulses
    qua_declaration  — declare standard I/Q variables and streams
"""

from qm.qua import *
from qualang_tools.addons.variables import assign_variables_to_element


def cz_gate(dc0):
    """Execute a CZ gate via the tc12 coupler flux line.

    Applies a calibrated DC offset to bring the coupler into the interaction
    regime for one CZ period, then returns to the parking flux.

    Args:
        dc0: Parking flux value to return to after the gate (coupling_off_flux).
    """
    set_dc_offset("tc12", "single", -0.10557)
    wait(189 // 4, "tc12")
    align()
    set_dc_offset("tc12", "single", dc0)
    wait(10)


def multiplexed_readout(I, I_st, Q, Q_st, resonators, sequential=False, amplitude=1.0, weights=""):
    """Perform multiplexed readout on one or more resonators.

    Args:
        I: List of QUA fixed variables for I quadrature results.
        I_st: List of QUA streams for I (or None to skip saving).
        Q: List of QUA fixed variables for Q quadrature results.
        Q_st: List of QUA streams for Q (or None to skip saving).
        resonators: List of resonator indices [1, 2] or a single int.
        sequential: If True, align between successive resonator measurements.
        amplitude: Amplitude scaling for the readout pulse.
        weights: Prefix for integration weight names (e.g. "rotated_" or "opt_").
    """
    if type(resonators) is not list:
        resonators = [resonators]

    for ind, res in enumerate(resonators):
        measure(
            "readout" * amp(amplitude),
            f"rr{res}",
            None,
            dual_demod.full(weights + "cos", weights + "sin", I[ind]),
            dual_demod.full(weights + "minus_sin", weights + "cos", Q[ind]),
        )

        if I_st is not None:
            save(I[ind], I_st[ind])
        if Q_st is not None:
            save(Q[ind], Q_st[ind])

        if sequential and ind < len(resonators) - 1:
            align(f"rr{res}", f"rr{res + 1}")


def qua_declaration(nb_of_qubits):
    """Declare standard I/Q variables and streams for multi-qubit readout.

    Args:
        nb_of_qubits: Number of qubits (and resonators) in the experiment.

    Returns:
        Tuple of (I, I_st, Q, Q_st, n, n_st).
    """
    n = declare(int)
    n_st = declare_stream()
    I = [declare(fixed) for _ in range(nb_of_qubits)]
    Q = [declare(fixed) for _ in range(nb_of_qubits)]
    I_st = [declare_stream() for _ in range(nb_of_qubits)]
    Q_st = [declare_stream() for _ in range(nb_of_qubits)]
    for i in range(nb_of_qubits):
        assign_variables_to_element(f"rr{i + 1}", I[i], Q[i])
    return I, I_st, Q, Q_st, n, n_st


def active_reset(threshold, qubit, resonator, max_tries=1, Ig=None):
    """Active qubit reset using measurement-conditioned pi pulses.

    Measures the resonator and conditionally plays x180 until the qubit is
    detected in the ground state (I < threshold) or max_tries is exceeded.

    Args:
        threshold: I-quadrature threshold separating |g> and |e>.
        qubit: Qubit element name (e.g. "q1_xy").
        resonator: Resonator element name (e.g. "rr1").
        max_tries: Maximum reset attempts (default 1).
        Ig: Optional pre-declared QUA fixed variable for the I result.

    Returns:
        Tuple of (Ig, counter) — final I measurement and number of tries used.
    """
    if Ig is None:
        Ig = declare(fixed)
    if (max_tries < 1) or (not float(max_tries).is_integer()):
        raise Exception("max_tries must be an integer >= 1.")

    assign(Ig, threshold + 2**-28)
    counter = declare(int)
    assign(counter, 0)

    align(qubit, resonator)
    with while_((Ig > threshold) & (counter < max_tries)):
        measure(
            "readout",
            resonator,
            None,
            dual_demod.full("rotated_cos", "rotated_sin", Ig),
        )
        play("x180", qubit, condition=(Ig > threshold))
        assign(counter, counter + 1)
    return Ig, counter

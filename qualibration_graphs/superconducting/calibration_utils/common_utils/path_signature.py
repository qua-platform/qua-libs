"""Real-time path and depth-2 signature of the readout trajectory (arXiv:2402.09532).

What is computed
----------------
``demod.accumulated`` returns the running integral of the demodulated readout signal, i.e.
the *path* ``X(t) = (I(t), Q(t))`` sampled at the end of every chunk. The depth-1 signature
of that path is its endpoint ``(I(T), Q(T))``, which is exactly what a conventional full
demodulation returns -- so the baseline for any comparison comes free from the same shots.

The depth-2 signature adds the four iterated integrals. Two of them are symmetric and fixed
by the endpoints (``\\int I dI = I(T)^2 / 2``), so the only new information is the
antisymmetric part, the Levy area::

    A = \\int dI dQ - \\int dQ dI

which is what separates a trajectory that drifted straight to a blob from one that turned
around mid-measurement -- the state-transition signal the paper exploits.

Midpoint (Stratonovich) increments
----------------------------------
The cross terms are accumulated with midpoint increments::

    cross_IQ = 1/2 * sum_k (I_k + I_{k-1}) (Q_k - Q_{k-1})
    cross_QI = 1/2 * sum_k (Q_k + Q_{k-1}) (I_k - I_{k-1})

with ``I_{-1} = Q_{-1} = 0``. This is not cosmetic. Summing the two midpoint terms
telescopes exactly::

    cross_IQ + cross_QI == I(T) * Q(T)

so every single shot carries its own arithmetic self-check, which also catches fixed-point
saturation in the accumulation. Left-point (Ito) increments would instead leave the residual
``-sum_k dI_k dQ_k``, the quadratic covariation, which is noise-dominated and nonzero -- the
check would not be exact and could not be used as a guard. The antisymmetric combination
``cross_IQ - cross_QI`` is identical for both conventions, so the Levy area is unaffected by
the choice.

Units
-----
Everything here stays in raw demodulation units, the same convention as
``qubit.resonator.gef_centers`` (see :mod:`~calibration_utils.common_utils.gef_discrimination`,
which documents units as the #1 bug source). The endpoints are in raw units and the cross
terms are in raw units *squared*; nothing is converted to Volts, precisely so that the
identity above holds verbatim on the stored numbers.

Resource budget
---------------
Accumulated I/Q demodulation costs 4 PPU processing blocks per measured qubit -- one per
single-output ``demod.accumulated``, see :func:`declare_path_arrays` for why there are four
rather than two -- against a limit of 16 per MW-FEM (20 per OPX+); a resonator that only plays the pulse without
demodulating still costs 1. :func:`accumulated_demod_batches` solves that budget per FEM, so
that a batch measures as many qubits as fit while the remaining selected qubits still play
their readout and the crosstalk environment is unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from qm.qua import assign, declare, declare_stream, demod, fixed, for_, measure, save

# PPU processing-block budget, per FEM.
BLOCKS_PER_ACCUMULATED_READOUT = 4  # four demod.accumulated, one block each
BLOCKS_PER_IDLE_READOUT = 1  # resonator plays the pulse but does not demodulate
MW_FEM_BLOCK_LIMIT = 16
OPX_PLUS_BLOCK_LIMIT = 20

# demod chunking is expressed in units of 4 ADC samples.
SAMPLES_PER_CHUNK_UNIT_NS = 4


# --------------------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------------------
def flat_weights_reason(pulse) -> Optional[str]:
    """Describe why ``pulse`` does not carry flat integration weights, or None if it does.

    Flat means constant magnitude over the pulse: quam stores that as ``[(value, length)]``
    (or a single scalar). A nonzero ``integration_weights_angle`` is *not* a problem here --
    it only rotates the IQ plane, and the Levy area is rotation invariant.

    Non-flat weights matter because ``demod.accumulated`` requires ``samples_per_chunk >= 7``
    (28 ns) once a pulse carries arbitrary weights, which would silently forbid the fine
    chunking this node exists to use.
    """
    weights = pulse.integration_weights
    if weights is None:
        return None
    if isinstance(weights, str):
        # A quam reference such as '#./default_integration_weights' resolves to flat weights.
        return None if "default_integration_weights" in weights else f"weights reference {weights!r} is not the default"
    if isinstance(weights, (int, float)):
        return None
    try:
        segments = list(weights)
    except TypeError:
        return f"unrecognised integration_weights {weights!r}"
    values = []
    for segment in segments:
        if isinstance(segment, (tuple, list)) and len(segment) == 2:
            values.append(segment[0])
        else:
            values.append(segment)
    if len(set(values)) <= 1:
        return None
    return f"integration weights vary over the pulse ({len(set(values))} distinct values)"


def n_chunks_for(pulse, samples_per_chunk: int) -> int:
    """Number of accumulated-demod chunks covering ``pulse``.

    Raises:
        ValueError: If the pulse length is not an exact multiple of the chunk duration --
            ``demod.accumulated`` requires the chunks to tile the pulse.
    """
    chunk_ns = SAMPLES_PER_CHUNK_UNIT_NS * samples_per_chunk
    if pulse.length % chunk_ns:
        raise ValueError(
            f"readout length {pulse.length} ns is not a multiple of the chunk duration "
            f"{chunk_ns} ns (samples_per_chunk={samples_per_chunk}); the chunks must tile "
            f"the pulse exactly."
        )
    return pulse.length // chunk_ns


def preflight_path_signature(
    qubits: Sequence[Any], operation: str, samples_per_chunk: int, reset_type: str
) -> Dict[str, int]:
    """Validate the acquisition and return the per-qubit chunk count.

    Everything that would make the acquired data quietly wrong is a hard failure here rather
    than a warning, because none of it is visible in the resulting dataset.

    Args:
        qubits: The qubits the node operates on.
        operation: Name of the resonator operation to acquire with.
        samples_per_chunk: Chunk size in units of 4 ns.
        reset_type: The node's reset type; only thermal reset is supported.

    Returns:
        Mapping of qubit name to its number of chunks.

    Raises:
        ValueError: On any unsupported configuration.
    """
    if samples_per_chunk < 1:
        raise ValueError(f"samples_per_chunk must be >= 1 (got {samples_per_chunk}).")
    if reset_type != "thermal":
        raise ValueError(f"Only 'thermal' reset is supported, got {reset_type!r}.")

    chunks: Dict[str, int] = {}
    for qubit in qubits:
        pulse = qubit.resonator.operations.get(operation)
        if pulse is None:
            raise ValueError(f"{qubit.name}: resonator has no operation {operation!r}.")
        reason = flat_weights_reason(pulse)
        if reason is not None:
            raise ValueError(
                f"{qubit.name}: {reason}. Accumulated demodulation at "
                f"{SAMPLES_PER_CHUNK_UNIT_NS * samples_per_chunk} ns chunks requires flat "
                f"integration weights (arbitrary weights force chunks of >= 28 ns)."
            )
        chunks[qubit.name] = n_chunks_for(pulse, samples_per_chunk)
    return chunks


# --------------------------------------------------------------------------------------
# Batching against the PPU block budget
# --------------------------------------------------------------------------------------
def _fem_key(qubit) -> Tuple[Any, Any]:
    """Identify the FEM (or controller) whose block budget this resonator draws on."""
    port = getattr(qubit.resonator, "opx_output", None)
    return (getattr(port, "controller_id", None), getattr(port, "fem_id", None))


def _block_limit(qubit) -> int:
    """PPU block limit of the FEM this resonator lives on."""
    from quam.components import MWChannel

    return MW_FEM_BLOCK_LIMIT if isinstance(qubit.resonator, MWChannel) else OPX_PLUS_BLOCK_LIMIT


def max_measured_per_batch(n_in_group: int, block_limit: int) -> int:
    """Largest number of simultaneously demodulated qubits within one FEM.

    Every selected qubit in the group either demodulates (4 blocks) or just plays its readout
    (1 block), so ``4 m + (n - m) <= limit``.
    """
    return max(
        0, min(n_in_group, (block_limit - n_in_group) // (BLOCKS_PER_ACCUMULATED_READOUT - BLOCKS_PER_IDLE_READOUT))
    )


def accumulated_demod_batches(qubits: Sequence[Any], multiplexed: bool) -> List[Dict[int, Any]]:
    """Split the qubits into batches that fit the accumulated-demod block budget.

    Each batch is a ``{index_in_qubits: qubit}`` mapping, matching the shape of
    ``BatchableList.batch()`` so node code reads the same as elsewhere. Qubits on different
    FEMs are packed into the same batch, since their budgets are independent.

    Args:
        qubits: The qubits the node operates on.
        multiplexed: If False every qubit is measured on its own and the limit never binds.

    Raises:
        ValueError: If a FEM carries so many selected qubits that not even one of them can be
            demodulated while the others play their readout.
    """
    if not multiplexed:
        return [{i: qubit} for i, qubit in enumerate(qubits)]

    groups: Dict[Tuple[Any, Any], List[Tuple[int, Any]]] = defaultdict(list)
    for i, qubit in enumerate(qubits):
        groups[_fem_key(qubit)].append((i, qubit))

    per_group_batches: List[List[List[Tuple[int, Any]]]] = []
    for key, items in groups.items():
        limit = _block_limit(items[0][1])
        max_measured = max_measured_per_batch(len(items), limit)
        if max_measured == 0:
            raise ValueError(
                f"{len(items)} selected qubits share FEM {key}, which leaves no room for even "
                f"one accumulated demodulation within its {limit} processing blocks "
                f"(4 per measured qubit, 1 per idle readout). Select fewer qubits on that FEM "
                f"or set multiplexed=False."
            )
        per_group_batches.append([items[j : j + max_measured] for j in range(0, len(items), max_measured)])

    n_batches = max(len(batches) for batches in per_group_batches)
    merged: List[Dict[int, Any]] = []
    for b in range(n_batches):
        batch: Dict[int, Any] = {}
        for group_batches in per_group_batches:
            if b < len(group_batches):
                batch.update(dict(group_batches[b]))
        merged.append(batch)
    return merged


# --------------------------------------------------------------------------------------
# QUA macros
# --------------------------------------------------------------------------------------
def declare_path_arrays(n_chunks: int):
    """Declare the four QUA arrays that the accumulated demodulation fills.

    A complex input channel (an IQ pair, or an MW FEM input) rejects
    ``dual_demod.accumulated``: the QOP supports complex demodulation only for *full*
    demodulation. The four single-output demodulations that a dual demod would have paired
    internally are therefore acquired separately, as ``(II, IQ, QI, QQ)``, and recombined on
    the PPU by :func:`reduce_to_signature`::

        I(t) = II(t) + IQ(t)
        Q(t) = QI(t) + QQ(t)

    That is exactly the pairing ``_InComplexChannel.measure`` uses for full demodulation, so
    the endpoint of the recombined path is still the quantity a conventional ``measure``
    would have returned.
    """
    return tuple(declare(fixed, size=n_chunks) for _ in range(4))


def measure_path(qubit, operation: str, path_arrays, samples_per_chunk: int) -> None:
    """Acquire the readout path into ``path_arrays`` with accumulated demodulation.

    Args:
        qubit: The qubit whose resonator is measured.
        operation: Name of the resonator operation to acquire with.
        path_arrays: The four arrays from :func:`declare_path_arrays`, filled with the
            accumulated ``(II, IQ, QI, QQ)`` components.
        samples_per_chunk: Chunk size in units of 4 ns.
    """
    II, IQ, QI, QQ = path_arrays
    pulse = qubit.resonator.operations[operation]
    labels = list(pulse.integration_weights_mapping)
    measure(
        operation,
        qubit.resonator.name,
        demod.accumulated(labels[0], II, samples_per_chunk, "out1"),
        demod.accumulated(labels[1], IQ, samples_per_chunk, "out2"),
        demod.accumulated(labels[2], QI, samples_per_chunk, "out1"),
        demod.accumulated(labels[0], QQ, samples_per_chunk, "out2"),
    )


def path_quadratures(path_arrays):
    """The two arrays holding the recombined ``I`` / ``Q`` path.

    Valid only after :func:`reduce_to_signature`, which recombines the four demodulated
    components in place into the first and third arrays.
    """
    return path_arrays[0], path_arrays[2]


def reduce_to_signature(path_arrays, n_chunks: int, k, I_end, Q_end, cross_IQ, cross_QI) -> None:
    """Recombine an acquired path and reduce it to its endpoint and depth-2 cross terms.

    The ``I = II + IQ`` / ``Q = QI + QQ`` recombination happens in place inside the same loop
    that accumulates the cross terms, so it costs no extra memory and no extra pass: chunk
    ``k`` is recombined just before it is used, and chunk ``k - 1`` was recombined on the
    previous iteration.

    Midpoint increments are accumulated (see the module docstring) and halved once at the end,
    which is exact in binary and keeps one multiplication out of the inner loop. The ``k = 0``
    boundary term is added separately since it references ``I_{-1} = Q_{-1} = 0``.

    Args:
        path_arrays: The four QUA arrays filled by :func:`measure_path`. On return, the
            first and third hold the recombined ``I`` and ``Q`` paths.
        n_chunks: Length of those arrays.
        k: A QUA int used as the loop variable.
        I_end, Q_end, cross_IQ, cross_QI: QUA fixed variables receiving the results.
    """
    II, IQ, QI, QQ = path_arrays
    assign(II[0], II[0] + IQ[0])
    assign(QI[0], QI[0] + QQ[0])
    assign(cross_IQ, II[0] * QI[0])
    assign(cross_QI, QI[0] * II[0])
    with for_(k, 1, k < n_chunks, k + 1):
        assign(II[k], II[k] + IQ[k])
        assign(QI[k], QI[k] + QQ[k])
        assign(cross_IQ, cross_IQ + (II[k] + II[k - 1]) * (QI[k] - QI[k - 1]))
        assign(cross_QI, cross_QI + (QI[k] + QI[k - 1]) * (II[k] - II[k - 1]))
    assign(cross_IQ, cross_IQ * 0.5)
    assign(cross_QI, cross_QI * 0.5)
    assign(I_end, II[n_chunks - 1])
    assign(Q_end, QI[n_chunks - 1])


def save_path(path_arrays, n_chunks: int, k, I_path_st, Q_path_st) -> None:
    """Stream a full acquired path, for offline validation of the real-time reduction.

    Must be called after :func:`reduce_to_signature`, which is what recombines the four
    demodulated components into the I/Q path.
    """
    I_path, Q_path = path_quadratures(path_arrays)
    with for_(k, 0, k < n_chunks, k + 1):
        save(I_path[k], I_path_st)
        save(Q_path[k], Q_path_st)


def declare_feature_streams(num_qubits: int, states: Sequence[str]):
    """Declare ``{(state, feature): [stream per qubit]}`` for the four streamed features."""
    return {
        (state, feature): [declare_stream() for _ in range(num_qubits)]
        for state in states
        for feature in ("I_end", "Q_end", "cross_IQ", "cross_QI")
    }


def feature_handle_name(state: str, feature: str, qubit_index: int) -> str:
    """Result-handle name of a streamed feature, e.g. ``cross_IQ_g1``."""
    return f"{feature}_{state}{qubit_index + 1}"


def path_handle_name(state: str, quadrature: str, qubit_index: int) -> str:
    """Result-handle name of a streamed path, e.g. ``I_path_g1``."""
    return f"{quadrature}_path_{state}{qubit_index + 1}"

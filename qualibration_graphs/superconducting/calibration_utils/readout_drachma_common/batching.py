from collections.abc import Sequence
from typing import Protocol

from quam.components import IQChannel, MWChannel
from quam_builder.architecture.superconducting.components.twpa import TWPA
from quam_builder.architecture.superconducting.qubit import AnyTransmon


class HasMultiplexed(Protocol):
    """Any node parameters object exposing the `multiplexed` flag."""

    multiplexed: bool


def max_cores_per_fem(qubits: Sequence[AnyTransmon]) -> int:
    """
    Pure physical per-FEM core-sharing cap, independent of whether a run multiplexes its
    readouts: in order to perform simultaneous, I/Q based accumulated demodulation during
    qubit readout, each qubit requires four `demod.accumulated` processing blocks. Each of
    these consumes a resource on the PPU, up to a maximum of 16 for the MW-FEM and 20 for
    the OPX+, leading to a maximum of 3 qubits on the MW-FEM and 4 on the OPX+ that can
    share a FEM's cores.
    """
    res_per_demod = 4

    if isinstance(qubits[0].resonator, MWChannel):
        resource_limit = 16
    elif isinstance(qubits[0].resonator, IQChannel):
        resource_limit = 20
    else:
        raise TypeError(f"Unrecognized resonator type {type(qubits[0].resonator)}")

    # Since we have to play the readout pulse on every non-measured resonator,
    # this will also occupy a thread. So, we have to make sure that
    # `max_accumulated_readouts` * 4 + `leftover_qubits` < limit
    max_reads = 0
    while max_reads < len(qubits):
        if ((max_reads + 1) * res_per_demod + len(qubits) - (max_reads + 1)) > resource_limit:
            break
        max_reads += 1

    return max_reads


def get_max_accumulated_readouts(qubits: Sequence[AnyTransmon], node_parameters: HasMultiplexed) -> int:
    """
    Per-batch simultaneous-measurement cap: `max_cores_per_fem`'s hardware limit if the
    measurement is multiplexed, otherwise unbounded (each qubit measured alone never
    contends for PPU resources with another).
    """
    if not node_parameters.multiplexed:
        return len(qubits)
    return max_cores_per_fem(qubits)


def build_batch_groups(
    qubits: Sequence[AnyTransmon], node_parameters: HasMultiplexed, max_per_fem: int | None = None
) -> list[list[int]]:
    """Split `qubits` into simultaneous-measurement batches (as lists of indices into
    `qubits`), respecting the per-MW-FEM PPU resource limit.

    Non-multiplexed runs are unaffected -- each qubit already gets its own batch, so
    no FEM ever has more than one qubit measured at a time. For multiplexed runs,
    qubits are first grouped by physical FEM (`opx_output.controller_id`, `.fem_id`),
    each FEM's group is capped via `get_max_accumulated_readouts` (the same PPU
    accumulated-demod resource limit as readout_optimization_3d, ~3 qubits/MW-FEM),
    and one chunk from each FEM is combined into the same batch per round -- so
    qubits on different FEMs still get multiplexed together, and only same-FEM
    groups larger than the cap are split across sequential batches.

    Caveat: `get_max_accumulated_readouts`'s resource formula was derived for
    readout_optimization_3d's measurement pattern, where some qubits in a shot only
    idle-play their readout pulse while others get full accumulated demod. Here every
    qubit in a batch gets both `measure` and `measure_sliced` (none are idle-only), so
    this is an approximation -- it lands on 3 for a 3-qubit FEM group (matching the
    known hardware limit), but should be re-verified on hardware if a FEM group size
    differs.
    """
    if not node_parameters.multiplexed:
        return [[i] for i in range(len(qubits))]

    fem_to_indices = {}
    for i, qubit in enumerate(qubits):
        opx_output = qubit.resonator.opx_output
        fem_key = (opx_output.controller_id, opx_output.fem_id)
        fem_to_indices.setdefault(fem_key, []).append(i)

    fem_chunks = {}
    for fem_key, indices in fem_to_indices.items():
        fem_qubits = [qubits[i] for i in indices]
        max_per_fem = (
            max_per_fem if max_per_fem is not None else get_max_accumulated_readouts(fem_qubits, node_parameters)
        )
        fem_chunks[fem_key] = [indices[j : j + max_per_fem] for j in range(0, len(indices), max_per_fem)]

    num_rounds = max(len(chunks) for chunks in fem_chunks.values())
    batch_groups = []
    for round_idx in range(num_rounds):
        batch = []
        for chunks in fem_chunks.values():
            if round_idx < len(chunks):
                batch.extend(chunks[round_idx])
        batch_groups.append(batch)
    return batch_groups


def assign_core_labels(
    qubits: Sequence[AnyTransmon],
    twpas: Sequence[TWPA] = (),
) -> tuple[list[str | None], dict[str, str]]:
    """Assign each qubit index a core label, so `qubit.xy.core`/`qubit.resonator.core` can be
    set to share cores wherever it is safe to do so -- config generation otherwise allocates
    a dedicated core for every qubit's xy/resonator elements, which can exceed a FEM's
    physical core count once enough qubits are assigned to it.

    A qubit's own xy and resonator never play simultaneously, so they always get the same
    label. Labels cycle `0, 1, ..., max_cores_per_fem - 1, 0, 1, ...` across all of a FEM's
    qubits (ranked by order of appearance in `qubits`), independent of how the node batches
    its measurements: within any single batch, at most `max_cores_per_fem` same-FEM qubits
    are ever active at once (the caller is responsible for that -- see `build_batch_groups`),
    so same-label qubits never actually overlap in time, and it is always safe to reuse a
    label once every `max_cores_per_fem` qubits. This must hold regardless of `multiplexed`:
    even when every batch has only one qubit (non-multiplexed), squeezing an entire FEM's
    qubits onto a single label still exceeds the FEM's physical per-core capacity and fails
    at config-allocation time, since config generation only cares about how many distinct
    qubit configs share a label, not whether they're ever simultaneous.

    Labels embed fem_id directly (e.g. "fem_{fem_id}_core_{pos}") -- client-side config
    generation (quam.components.channels.Channel.apply_to_config) treats `core` as a single
    flat namespace with no controller/FEM scoping of its own, so two qubits on different FEMs
    must never receive the same literal label string, even if their per-FEM position happens
    to match -- reusing a bare "c{pos}" label across FEMs previously caused compiler crashes
    for exactly that reason. controller_id is deliberately left out of the label
    (single-controller setups only) -- if qubits ever span multiple controllers, two different
    controllers' same-numbered FEM would collide on this label and controller_id would need to
    be added back in.

    Each TWPA's pump (sticky, kept alive for the whole program via `initialize_qpu`/
    `twpa_keepalive`) and pump_ (non-sticky calibration variant, never played at the same
    time as pump) share one label. That label is globally unique across all TWPAs -- not just
    distinct per-FEM like qubit labels -- since multiple TWPAs may pump concurrently, and it is
    also distinct from every qubit label on the same FEM: unlike a qubit, the pump runs
    continuously for the entire program, so it can never share a core with anything else on
    its FEM -- qubit label positions on that FEM start counting after the TWPA's reserved slot.
    Returns `(qubit_core_labels, twpa_core_labels)`, the latter keyed by `twpa.name`.
    """
    twpa_labels: dict[str, str] = {}
    fem_reserved: dict[tuple, int] = {}
    for i, twpa in enumerate(twpas):
        opx_output = twpa.pump.opx_output
        fem_key = (opx_output.controller_id, opx_output.fem_id)
        twpa_labels[twpa.name] = f"twpa{i}"
        fem_reserved[fem_key] = fem_reserved.get(fem_key, 0) + 1

    fem_to_indices: dict[tuple, list[int]] = {}
    for i, qubit in enumerate(qubits):
        opx_output = qubit.resonator.opx_output
        fem_key = (opx_output.controller_id, opx_output.fem_id)
        fem_to_indices.setdefault(fem_key, []).append(i)

    core_labels: list[str | None] = [None] * len(qubits)
    for fem_key, indices in fem_to_indices.items():
        _, fem_id = fem_key
        fem_qubits = [qubits[i] for i in indices]
        max_per_fem = max_cores_per_fem(fem_qubits)
        base = fem_reserved.get(fem_key, 0)
        for local_rank, idx in enumerate(indices):
            core_labels[idx] = f"fem_{fem_id}_core_{base + local_rank % max_per_fem}"
    return core_labels, twpa_labels

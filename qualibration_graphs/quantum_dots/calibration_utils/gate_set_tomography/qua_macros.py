"""QUA gate-playback helpers for gate set tomography.

:func:`play_gst_sequence` dispatches ``prep_id``, ``meas_id``, and ``germ_id`` through
``switch_`` / ``case_`` blocks. Case bodies are lists of qubit gate callables built
offline by :func:`~.gst_sequences.build_gst_playback_macro_lists` from the same
``gate_map`` dictionaries and ``basic_gates_map`` as used for sequence indexing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from qm.qua import case_, declare, for_, switch_


def play_gst_sequence(
    qubit,
    prep_id,
    meas_id,
    germ_id,
    repetition,
    prep_case_macros: Sequence[Sequence[Callable[..., None]]],
    meas_case_macros: Sequence[Sequence[Callable[..., None]]],
    germ_case_macros: Sequence[Sequence[Callable[..., None]]],
) -> None:
    """Play a GST segment on *qubit* for the given case indices and germ repetition count.

    Parameters
    ----------
    qubit
        Qubit with XY gate macros (e.g. ``x90``, ``y90``) matching ``basic_gates_map``.
    prep_id, meas_id, germ_id
        Integer case labels from :func:`~.gst_sequences.build_gate_map`.
    repetition
        Number of times to play the germ body (0 skips the germ loop).
    prep_case_macros, meas_case_macros, germ_case_macros
        For each case index ``i``, ``*_case_macros[i]`` is an iterable of bound callables
        to invoke in order (from :func:`~.gst_sequences.build_gst_playback_macro_lists`).
    """
    germ_rep = declare(int)

    with switch_(prep_id, unsafe=True):
        for i, macros in enumerate(prep_case_macros):
            with case_(i):
                for m in macros:
                    m()

    with switch_(germ_id, unsafe=True):
        for i, macros in enumerate(germ_case_macros):
            with case_(i):
                with for_(germ_rep, 0, germ_rep < repetition, germ_rep + 1):
                    for m in macros:
                        m()

    with switch_(meas_id, unsafe=True):
        for i, macros in enumerate(meas_case_macros):
            with case_(i):
                for m in macros:
                    m()

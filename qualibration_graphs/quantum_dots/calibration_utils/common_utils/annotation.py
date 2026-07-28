from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode


def annotate_node_figures(_node: QualibrationNode) -> None:
    """Optionally annotate node figures.

    This repository's dot calibration nodes call this helper, but figure annotation
    is not required for the analysis/plotting correctness tests. Keep it as a safe
    no-op so nodes can run standalone in minimal environments.
    """


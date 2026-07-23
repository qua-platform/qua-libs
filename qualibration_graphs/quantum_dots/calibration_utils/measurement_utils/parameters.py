from typing import Literal

from qualibrate.core.parameters import RunnableParameters


class ParityDiffAnalysisParameters(RunnableParameters):
    analysis_signal: Literal["E_p1_given_p0_0", "E_p1_given_p0_1"] = "E_p1_given_p0_0"
    """Which conditional expectation to use for fitting.
    E_p1_given_p0_0: P(second=1 | first=0) — post-select on empty dot.
    E_p1_given_p0_1: P(second=1 | first=1) — post-select on loaded dot."""
    parity_measurement: bool = False
    """Whether to use parity pre measurement. Default is False."""

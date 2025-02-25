from typing import Literal
from pydantic import Field
from qualibrate.parameters import RunnableParameters


class FluxControlledNodeParameters(RunnableParameters):
    flux_point_joint_or_independent: Literal["joint", "independent"] = Field(
        default="joint",
        description="Whether to use the joint-flux point for all qubits, or to set"
        "each to qubit's flux point independently for maximum separation.",
    )

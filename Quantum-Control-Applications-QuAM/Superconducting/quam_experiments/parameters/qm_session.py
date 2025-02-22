from pydantic import Field
from qualibrate.parameters import RunnableParameters

class QmSessionNodeParameters(RunnableParameters):
    timeout: int = Field(
        default=120,
        description="Waiting time for the OPX resources to become available before giving up (in seconds).",
    )

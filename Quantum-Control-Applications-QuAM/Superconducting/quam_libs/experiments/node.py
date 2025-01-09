from typing import Optional, List

from qualibrate import NodeParameters


class QubitExperimentNodeParameters(NodeParameters):
    qubits: Optional[List[str]] = None

class SimulatableNodeParameters(QubitExperimentNodeParameters):
    simulate: bool = False
    simulation_duration_ns: int = 2500

from typing import List, Optional, Union, Literal
from pydantic import Field
from qualibrate import QualibrationNode
from qualibrate.parameters import RunnableParameters
from qualibration_libs.batchable_list import BatchableList
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_config import QuAM


class QubitsExperimentNodeParameters(RunnableParameters):
    qubits: Optional[List[str]] = Field(
        default=None,
        description="A list of qubit names which should participate in the execution of the node.",
    )
    multiplexed: bool = Field(
        default=False,
        description="Whether to play control pulses, readout pulses and active/thermal reset "
        "at the same time for all qubits (True) or to play the experiment sequentially"
        "for each qubit (False)",
    )
    use_state_discrimination: bool = Field(
        default=False,
        description="Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated quadratures 'I' and 'Q'.",
    )
    reset_type: Literal["thermal", "active", "active_gef"] = Field(
        default="thermal",
        description="The qubit reset method to use. Must be implemented as a method of QuAM.qubit.",
    )


def make_batchable_list_from_multiplexed(items: List, multiplexed: bool) -> BatchableList:
    if multiplexed:
        batched_groups = [[i for i in range(len(items))]]
    else:
        batched_groups = [[i] for i in range(len(items))]

    return BatchableList(items, batched_groups)


def get_qubits(node: QualibrationNode) -> BatchableList[QuAM.qubit_type]:
    # todo: need a docstring!
    # todo: make a method once https://github.com/qua-platform/qualibrate-core/pull/89 is merged
    qubits = _get_qubits(node.machine, node.parameters)

    if isinstance(node.parameters, QubitsExperimentNodeParameters):
        multiplexed = node.parameters.multiplexed
    else:
        multiplexed = False

    qubits_batchable_list = make_batchable_list_from_multiplexed(qubits, multiplexed)

    return qubits_batchable_list


def _get_qubits(machine: QuAM, node_parameters: QubitsExperimentNodeParameters) -> List[AnyTransmon]:
    # todo: make a method once https://github.com/qua-platform/qualibrate-core/pull/89 is merged
    if node_parameters.qubits is None or node_parameters.qubits == "":
        qubits = machine.active_qubits
    else:
        qubits = [machine.qubits[q] for q in node_parameters.qubits]

    return qubits

from typing import List, Optional, Union

from pydantic import Field
from qualibrate.parameters import RunnableParameters

from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_config.my_quam import BaseQuAM

from quam_experiments.parameters import MultiplexableNodeParameters
from quam_libs.batchable_list import BatchableList
from quam_experiments.parameters.multiplexable import make_batchable_list_from_multiplexed


class QubitsExperimentNodeParameters(RunnableParameters):
    qubits: Optional[Union[List[str], str]] = Field(
        default=None,
        description="A list of qubit names, or comma-separated list of qubit names"
        " which should participate in the execution of the node.",
    )


def get_qubits_used_in_node(
    machine: BaseQuAM, node_parameters: QubitsExperimentNodeParameters
) -> BatchableList[AnyTransmon]:
    # todo: make a method once https://github.com/qua-platform/qualibrate-core/pull/89 is merged
    qubits = _get_qubits(machine, node_parameters)

    if isinstance(node_parameters, MultiplexableNodeParameters):
        multiplexed = node_parameters.multiplexed
    else:
        multiplexed = False

    qubits_batchable_list = make_batchable_list_from_multiplexed(qubits, multiplexed)

    return qubits_batchable_list


def _get_qubits(machine: BaseQuAM, node_parameters: QubitsExperimentNodeParameters) -> List[AnyTransmon]:
    # todo: make a method once https://github.com/qua-platform/qualibrate-core/pull/89 is merged
    if node_parameters.qubits is None or node_parameters.qubits == "":
        qubits = machine.active_qubits
    else:
        qubits = [machine.qubits[q] for q in node_parameters.qubits]

    return qubits

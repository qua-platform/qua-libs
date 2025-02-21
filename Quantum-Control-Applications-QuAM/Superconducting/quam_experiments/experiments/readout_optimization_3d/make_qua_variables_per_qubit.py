from typing import Sequence, Union, Tuple, List

from qm.qua import declare, fixed, QuaVariableType

from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_experiments.experiments.readout_optimization_3d.parameters import ReadoutOptimization3dParameters


def make_qua_variables_per_qubit(qubits: Sequence[AnyTransmon], measurement_batch: Sequence[AnyTransmon],
                                 node_parameters: ReadoutOptimization3dParameters) -> Tuple[List[Union[QuaVariableType, None]]]:
    """
    Create lists of QUA readout variables for accumulated demodulation for each
    qubit only if the qubit is in the measurement batch, otherwise, no variable
    is created and the corresponding list element is None.
    """
    II_g, IQ_g, QI_g, QQ_g, II_e, IQ_e, QI_e, QQ_e, = [[] for _ in range(8)]

    qua_variables_per_qubit = [II_g, IQ_g, QI_g, QQ_g, II_e, IQ_e, QI_e, QQ_e]

    for i in range(len(qubits)):
        for j in range(len(qua_variables_per_qubit)):
            if qubits[i] not in measurement_batch:
                qua_variables_per_qubit[j].append(None)
            else:
                qua_variables_per_qubit[j].append(declare(fixed, size=node_parameters.num_durations))

    return II_g, IQ_g, QI_g, QQ_g, II_e, IQ_e, QI_e, QQ_e

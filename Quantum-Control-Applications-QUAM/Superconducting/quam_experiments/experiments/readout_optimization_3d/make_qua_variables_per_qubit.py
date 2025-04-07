from typing import Sequence, Union, Tuple, List

from qm.qua import declare, fixed, QuaVariableType

from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_experiments.experiments.readout_optimization_3d.parameters import (
    ReadoutOptimization3dParameters,
)


try:
    from qm.qua.type_hints import QuaVariable

    QuaVariableFloat = QuaVariable[float]
except ImportError:
    from qm.qua._dsl import QuaVariableType as QuaVariableFloat


def make_qua_variables_per_qubit(
    measurement_batch: Sequence[AnyTransmon],
    node_parameters: ReadoutOptimization3dParameters,
) -> Tuple[List[Union[QuaVariableFloat, None]]]:
    """
    Create lists of QUA readout variables for accumulated demodulation for each
    qubit only if the qubit is in the measurement batch, otherwise, no variable
    is created and the corresponding list element is None.
    """
    II_g, IQ_g, QI_g, QQ_g, I_g, Q_g, II_e, IQ_e, QI_e, QQ_e, I_e, Q_e = [[] for _ in range(12)]

    qua_variables_per_qubit = [
        II_g,
        IQ_g,
        QI_g,
        QQ_g,
        I_g,
        Q_g,
        II_e,
        IQ_e,
        QI_e,
        QQ_e,
        I_e,
        Q_e,
    ]

    for i in range(len(measurement_batch)):
        for j in range(len(qua_variables_per_qubit)):
            qua_variables_per_qubit[j].append(declare(fixed, size=node_parameters.num_durations))

    return II_g, IQ_g, QI_g, QQ_g, I_g, Q_g, II_e, IQ_e, QI_e, QQ_e, I_e, Q_e

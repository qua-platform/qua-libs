from typing import Sequence, Tuple, List, Union

from qm.qua import declare_stream
from qm.qua._dsl import _ResultSource

from quam_builder.architecture.superconducting.qubit import AnyTransmon


def make_qua_streams_per_qubit(
    measurement_batch: Sequence[AnyTransmon],
) -> Tuple[List[Union[_ResultSource, None]]]:
    """
    Create lists of QUA output streams for measurement data for each qubit
    only if the qubit is in the measurement batch, otherwise, no variable
    is created and the corresponding list element is None.
    """
    qua_streams_per_qubit = I_g_st, Q_g_st, I_e_st, Q_e_st = [[] for _ in range(4)]

    for i in range(len(measurement_batch)):
        for j in range(len(qua_streams_per_qubit)):
            qua_streams_per_qubit[j].append(declare_stream())

    return I_g_st, Q_g_st, I_e_st, Q_e_st

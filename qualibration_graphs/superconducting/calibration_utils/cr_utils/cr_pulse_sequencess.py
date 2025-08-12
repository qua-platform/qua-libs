from qm.qua import *

from quam_builder.architecture.superconducting.qubit_pair import AnyTransmonPair


def get_cr_elements(qp: AnyTransmonPair):
    qc = qp.qubit_control
    qt = qp.qubit_target
    cr = qp.cross_resonance
    cr_elems = [qc.xy.name, qt.xy.name, cr.name]
    return qc, qt, cr, cr_elems

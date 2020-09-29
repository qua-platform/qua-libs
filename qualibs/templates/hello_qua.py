from qm.qua import *


def hello_qua():
    with program() as prog:
        play('playOp', 'qe1')

    return prog

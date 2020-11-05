from qm.qua import program, play


def hello_qua():
    with program() as prog:
        play('playOp', 'qe1')

    return prog

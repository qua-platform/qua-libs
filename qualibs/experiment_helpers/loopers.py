from qualibs.templates.measurement_config import config
# from qualibs.templates.hello_qua import hello_qua
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from typing import Tuple
from dataclasses import dataclass

QMm = QuantumMachinesManager()
QM = QMm.open_qm(config)


# job=QM.simulate(hello_qua(),SimulationConfig(5000))

def looper(str, dependent_vars={}, independent_vars={}, method=integration.full):
    I = declare(fixed, value=0)

    measure('readoutOp', 'measElem', None, method('x', I))
    save(I, str)


@dataclass
class Saver:
    indep_var: Tuple = None
    dep_var: Tuple = None
    avg_var: Tuple = None

    def save(self):
        stream1 = declare_stream()
        with stream_processing():
            stream1.buffer(i,j).map(FUNCTIONS.average()).save_all('stream1')


saver = Saver()

with program() as prog:
    I=declare(fixed)
    ind = declare(int)
    with for_(ind, 0, ind < 3, ind + 1):
        measure('readoutOp', 'measElem', None, integration.full('x', I))
        saver.save()

# saver = Saver()
#
# with program() as prog:
#     with for_(avg):
#         with for_(i):
#             with for_(k):
#                 saver.save(indep_var=(i, j), dep_var=(I, Q), avg_var=avg)
#
job = QM.execute(prog)
saver.plot(job)
#
# with program() as prog:
#     str = declare_stream()
#     out_loop = declare(int)
#     in_loop = declare(int)
#     # I = declare(fixed, value=0)
#     with for_(out_loop, 0, out_loop < 3, out_loop + 1):
#         with for_(in_loop, 0, in_loop < 3, in_loop + 1):
#             looper(str, method=integration.full)
#             # measure('readoutOp', 'measElem', None, integration.full('x', I))
#             # save(I, str)
#     with stream_processing():
#         str.save_all('str')
#
# job = QM.simulate(prog, SimulationConfig(11000))
# res = job.result_handles
# print(len(res.str.fetch_all()))
# # samples=job.get_simulated_samples()
# # samples.con1.plot()

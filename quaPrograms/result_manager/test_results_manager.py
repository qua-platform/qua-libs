from result_manager.results import *

from base_configuration.vanilla_config import *
from base_configuration.hello_qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
import shutil


def test_results_manager():
    qmm = QuantumMachinesManager(store=ResultStore())
    qm1 = qmm.open_qm(config)
    job = qm1.simulate(hello_qua(), SimulationConfig(1000))
    res = job.result_handles
    res.save_to_store()
    assert(1)
    shutil.rmtree('res')

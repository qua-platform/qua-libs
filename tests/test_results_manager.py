import pytest

from qualibs.results.results import *
from qualibs.templates.vanilla_config import *
from qualibs.templates.hello_qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
import shutil

curr_path='res'

def test_results_manager_path_generate():
    myResults = ResultStore()
    qmm = QuantumMachinesManager(store=myResults)
    qm1 = qmm.open_qm(config)
    job = qm1.simulate(hello_qua(), SimulationConfig(1000))
    res = job.result_handles
    res.save_to_store()
    assert(os.path.exists(myResults.get_save_path(res._job_id)))
    shutil.rmtree(curr_path)

def test_results_manager_add_result():
    myResults=ResultStore()
    qmm = QuantumMachinesManager(store=myResults)
    myResults.add_result('a', 1)
    qm1 = qmm.open_qm(config)
    job = qm1.simulate(hello_qua(), SimulationConfig(1000))
    res = job.result_handles
    res.save_to_store()
    res_path = myResults.get_save_path(res._job_id)
    with open(os.path.join(res_path,'results.json'),'r') as f:
        res_dictionary=json.loads(f.read())
        assert(res_dictionary['a']==1)
    shutil.rmtree(curr_path)

def test_results_manager_drop_result():
    myResults=ResultStore()
    qmm = QuantumMachinesManager(store=myResults)
    myResults.add_result('a', 1)
    myResults.drop_result('a')
    qm1 = qmm.open_qm(config)
    job = qm1.simulate(hello_qua(), SimulationConfig(1000))
    res = job.result_handles
    res.save_to_store()
    res_path = myResults.get_save_path(res._job_id)
    with open(os.path.join(res_path,'results.json'),'r') as f:
        res_dictionary=json.loads(f.read())
        with pytest.raises(KeyError):
            res_dictionary['a']==1
    shutil.rmtree(curr_path)

def test_results_manager_save_calling_script():
    myResults=ResultStore(script_path=__file__)
    qmm = QuantumMachinesManager(store=myResults)
    myResults.add_result('a', 1)
    qm1 = qmm.open_qm(config)
    job = qm1.simulate(hello_qua(), SimulationConfig(1000))
    res = job.result_handles
    res.save_to_store()
    res_path=myResults.get_save_path(res._job_id)
    assert(os.path.exists(os.path.join(res_path,__file__)))
    shutil.rmtree(curr_path)

def test_results_manager_save_arb_root():
    myResults = ResultStore()
    qmm = QuantumMachinesManager(file_store_root=r'c:\\',store=myResults)
    qm1 = qmm.open_qm(config)
    job = qm1.simulate(hello_qua(), SimulationConfig(1000))
    res = job.result_handles
    res.save_to_store()
    assert (os.path.exists(myResults.get_save_path(res._job_id)))
    shutil.rmtree(curr_path)

def test_results_manager_list_results_in_folder():
    myResults = ResultStore()
    qmm = QuantumMachinesManager(store=myResults)
    qm1 = qmm.open_qm(config)
    job = qm1.simulate(hello_qua(), SimulationConfig(1000))
    res = job.result_handles
    res.save_to_store()
    # shutil.rmtree('res')


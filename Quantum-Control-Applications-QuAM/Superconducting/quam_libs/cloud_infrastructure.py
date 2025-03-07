import warnings
import importlib

if importlib.util.find_spec("iqcc_cloud_client"):
    from iqcc_cloud_client import IQCC_Cloud


class CloudQuantumMachinesManager:
    def __init__(self, backend):
        self.backend = backend

    def open_qm(self, config: dict, close_other_machines: bool,keep_dc_offsets_when_closing=True):
        self._qm = CloudQuantumMachine(self.backend, config)
        return self._qm


class CloudQuantumMachine:
    def __init__(self, backend,config: dict):
        self._qc = IQCC_Cloud(
            quantum_computer_backend=backend)
        self._config = config

    def execute(self, program):
        run_data = self._qc.execute(program, self._config,options = {"timeout":600})
        self.job = CloudJob(run_data)
        return self.job

    def get_running_job(self):
        if self.job.result_handles.is_processing():
            return self.job
        else:
            return None

    def close(self):
        pass


class CloudJob:
    def __init__(self, run_data: dict):
        self._run_data = run_data
        self.result_handles = CloudResultHandles(self._run_data['result'])


class CloudResultHandles:
    def __init__(self, results_dict: dict):
        self._results_dict = results_dict
        self._is_processing = True
        for result in results_dict:
            setattr(self, result, results_dict[result])

    def is_processing(self):
        is_processing = self._is_processing
        if is_processing:
            self._is_processing = False
        return is_processing

    def wait_for_all_values(self):
        pass

    def keys(self):
        return self._results_dict.keys()

    def get(self, handle: str):
        return CloudResult(self._results_dict[handle])
                   
class CloudResult:
    def __init__(self, data):
        self._data = data

    def fetch_all(self):
        return self._data

    def wait_for_values(self, *args):
        pass
from qm.persistence import *
from pathlib import Path
from datetime import date
from shutil import copyfile
import os
from typing import Callable
import json
import pandas as pd


class ResultStore(BaseStore):

    def __init__(self, store_path_root: str = '.', exp_name: str = '', script_path: str = '') -> None:
        """
            Defines the data saved on a node run
            :param store_path_root: Results will be arranged in file system relative this path
            :type store_path_root: str
            :param exp_name: short descriptive name for the experiment
            :type exp_name: str
            :param script_path: path to the script generating the run
            :type script_path: str
        """
        super().__init__()
        self.store_path_root = Path(store_path_root).absolute()
        self.script_path = script_path
        self.exp_name = exp_name
        self.results = {}

    def _job_path(self, job_id: str):

        d1 = date.today().strftime("%d%m%Y")
        if self.exp_name:
            path = Path(f"{self.store_path_root}/res/{d1}/{job_id}_{self.exp_name}")
        else:
            path = Path(f"{self.store_path_root}/res/{d1}/{job_id}")

        path.mkdir(parents=True, exist_ok=True)
        return path

    def job_named_result(self, job_id: str, name: str) -> BinaryAsset:
        return FileBinaryAsset(self._job_path(job_id).joinpath(f"result_{name}.npy"))

    def all_job_results(self, job_id: str) -> BinaryAsset:

        """
        This function defines what is saved upon calling job.save_results()
        """
        with open(self._results_path(job_id), "w") as dictator_file:
            json.dump(self.results, dictator_file)
        if self.script_path != '':
            copyfile(self.script_path, self._script_path(job_id))

        return FileBinaryAsset(self._job_path(job_id).joinpath(f"results.npz"))

    def _script_path(self, job_id):
        head, tail = os.path.split(self.script_path)
        return self._job_path(job_id).joinpath(tail)

    def _results_path(self, job_id):
        return self._job_path(job_id).joinpath(f"results.json")

    def add_result(self, name: str, res):
        """
            Adds a result to the saved results dictionary
            :param name: name of the saved variable
            :type name: str
            :param res: the stored result

        """
        self.results[name] = res
        print(f"Result {name} added to store")

    def drop_result(self, name: str):
        """
            Removes a result from saved results dictionary
            :param name: name of the saved variable
            :type name: str
        """
        self.results.pop(name)
        print(f"Result {name} removed from store")

    def list_results(self):
        print('------------------')
        print('Results dictionary')
        print('------------------')
        [print(f"{key} : {self.results[key]}") for key in self.results]

if __name__ =='__main__':
    a=ResultStore()
    a.add_result('this',1)
    a.add_result('that', 2)
    a.list_results()
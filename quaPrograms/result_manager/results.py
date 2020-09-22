import datetime

from qm.persistence import *
from pathlib import Path
from datetime import date
from shutil import copyfile
import os
import yattag
import json


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
        self.results = {'exp_name': self.exp_name, 'user_name': os.getlogin()}

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

        if self.script_path != '':
            copyfile(self.script_path, self._script_path(job_id))

        self.add_result('run_end_time', datetime.datetime.now().strftime('%c'))
        self.add_result('job_id', job_id)

        with open(self._results_path(job_id), "w") as results_file:
            json.dump(self.results, results_file)

        self._make_log_file()

        return FileBinaryAsset(self._job_path(job_id).joinpath(f"results.npz"))

    def _script_path(self, job_id):
        head, tail = os.path.split(self.script_path)
        return self._job_path(job_id).joinpath(tail)

    def _results_path(self, job_id):
        return self._job_path(job_id).joinpath(f"results.json")

    def _make_log_file(self):
        try:
            log = f"Log for QM run {self.results['job_id']}\n"
            if self.exp_name:
                log += f"Experiment name: {self.exp_name}\n"

            log += f"Run ended on {self.results['run_end_time']}\n"

            with open(self._job_path(self.results['job_id']).joinpath(f"result.log"), 'w') as log_file:
                log_file.write(log)

        except KeyError:
            print('No job id. cannot generate log file')
            # raise KeyError



    def get_save_path(self, job_id):
        return self._job_path(job_id)

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


def result_report():
    pass


def get_results_in_path(path):
    folder_list = [x[0] for x in os.walk(path)]
    res_folder_list=[]
    for folder in folder_list:
        if os.path.exists(os.path.join(folder,'result.log')):
            res_folder_list.append(folder)
            print('results folder found!')
    print(res_folder_list)

if __name__ == '__main__':
    a = ResultStore()
    a.add_result('this', 1)
    a.add_result('that', 2)
    a.list_results()
    get_results_in_path('res')

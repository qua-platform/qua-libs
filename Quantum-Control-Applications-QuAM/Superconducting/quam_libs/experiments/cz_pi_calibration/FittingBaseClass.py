from sklearn import preprocessing
from typing import List, Union
from scipy import optimize
import numpy as np
import itertools
import json


def find_nearest(array, value):
    """
    finds the index which corresponds to the value in the array closest to the value given
    :param array:
    :param value:
    :return:
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class FittingBaseClass:
    def __init__(
        self,
        x_data: Union[np.ndarray, List[float]],
        y_data: Union[np.ndarray, List[float]],
        guess=None,
        verbose=None,
        plot=False,
        save_file=False,
    ):

        self.x_data = x_data
        self.y_data = y_data
        self.xn = preprocessing.normalize([x_data], return_norm=True)
        self.yn = preprocessing.normalize([y_data], return_norm=True)
        self.x = self.xn[0][0]
        self.y = self.yn[0][0]
        self.x_normal = self.xn[1][0]
        self.y_normal = self.yn[1][0]

        self.guess = guess
        self.verbose = verbose
        self.plot = plot
        self.save_file = save_file

        self.out = None
        self.popt = None
        self.pcov = None
        self.perr = None

    def generate_initial_params(self):
        raise NotImplementedError

    def load_guesses(self, guess_dict):
        raise NotImplementedError

    def func(self):
        raise NotImplementedError

    def generate_out_dictionary(self):
        raise NotImplementedError

    def plot_fn(self):
        raise NotImplementedError

    def print_initial_guesses(self):
        raise NotImplementedError

    def print_fit_results(self):
        raise NotImplementedError

    def save(self):
        fit_params = dict(itertools.islice(self.out.items(), 1, len(self.out)))
        fit_params["x_data"] = self.x_data.tolist()
        fit_params["y_data"] = self.y_data.tolist()
        fit_params["y_fit"] = (self.fit_type(self.x, self.popt) * self.y_normal).tolist()
        json_object = json.dumps(fit_params)
        if self.save_file[-5:] == ".json":
            self.save_file = self.save_file[:-5]
        with open(f"{self.save_file}.json", "w") as outfile:
            outfile.write(json_object)

    def fit_type(self, x_var, a):
        return self.func(x_var, *a)

    def fit_data(self, p0):

        self.popt, self.pcov = optimize.curve_fit(self.func, self.x, self.y, p0=p0)

        self.perr = np.sqrt(np.diag(self.pcov))

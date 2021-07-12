import numpy as np
from qm.qua import *
from pygsti.construction import make_lsgst_experiment_list
from pygsti.modelpacks import GSTModelPack
import pygsti


class QuaGST:
    def __init__(self, model: GSTModelPack, *gate_macros):

        self.pygsti_model = model
        self.gates = gate_macros

        assert len(self.pygsti_model.gates) == len(self.gates)


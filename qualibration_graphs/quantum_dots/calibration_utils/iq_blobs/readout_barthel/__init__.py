"""
readout_barthel
A small library to simulate Barthel-style single-shot readout traces for a two-state system
with decay during measurement, and to fit flexible probabilistic models with NumPyro.
"""
from .simulate import simulate_readout, SimulationParams
from readout_barthel_project.readout_barthel.models.gmm_model import build_gmm_model
from .fit import fit_model, MCMCConfig

from qualibrate import QualibrationNode

from quam_libs.components import QuAM
from quam_libs.lib.instrument_limits import instrument_limits
from quam_libs.macros import qua_declaration
from quam_libs.experiments.simulation import simulate_and_plot
from quam_libs.experiments.execution import print_progress_bar
from quam_libs.experiments.qubit_spectroscopy.program import define_program
from quam_libs.experiments.qubit_spectroscopy.parameters import Parameters
from quam_libs.experiments.qubit_spectroscopy.node import get_optional_pulse_duration
from quam_libs.experiments.qubit_spectroscopy.analysis import fetch_dataset, fit_qubits
from quam_libs.experiments.qubit_spectroscopy.plotting import plot_qubit_response

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit

import matplotlib.pyplot as plt
import numpy as np
from qm.qua import *

class NodeBase(QualibrationNode):
    
    # Class containing tools to help handling units and conversions.
    u = unit(coerce_to_integer=True)
    
    def __init__(self, 
                 name, 
                 parameters = None, 
                 description = None, 
                 *, 
                 parameters_class = None, 
                 modes = None
                 
                 
                 ):
        
        super().__init__(name, parameters, description, parameters_class=parameters_class, modes=modes)

        self._initialize_quam_and_qop()
    
    def _initialize_quam_and_qop(self):
        
        # Instantiate the QuAM class from the state file
        self.machine = QuAM.load()
        # Generate the OPX and Octave configurations
        self.config = self.machine.generate_config()
        # Get the relevant QuAM components
        self.qubits = self.machine.get_qubits_used_in_node(self.parameters)
        # self.resonators = self.machine.get_resonators_used_in_node(self.parameters)
        self.num_qubits = len(self.qubits)
        # Open Communication with the QOP
    
    def connect(self):
        self.qmm = self.machine.connect()
        
    def export(self):
        return self.machine.export()
        

class QubitSpectroscopy(NodeBase):
    
    def __init__(self, 
                parameters = None,
                description = None,
                *,  
                parameters_class = None,
                modes = None
            ):
        
        name = "03a_Qubit_Spectroscopy"
        super().__init__(name, parameters, description, parameters_class=parameters_class, modes=modes)
        
        self.program = self.define_program()
    
    def define_program(self, qubits, machine):
        return define_program(self, qubits, machine)
        
    def simulate_pulse_sheet(self):
        fig, samples = simulate_and_plot(self.qmm, self.config, self.program, self.parameters)
        results = {"figure": fig, "samples": samples}
        
        return results
    
    def execute(self):
        with qm_session(self.qmm, self.config, timeout=self.parameters.timeout) as qm:
            job = qm.execute(self.program)
            print_progress_bar(job, "n", self.parameters.num_averages)
        
        return job

    def fetch_data_from_job(self, job):
        # {Data_fetching_and_dataset_creation}
        if self.parameters.load_data_id is None:
            ds = fetch_dataset(job, self.qubits, frequencies=dfs)
            node.results = {"ds": ds}
        else:
            node = self.load_from_id(node.parameters.load_data_id)
            ds = node.results["ds"]

            # %% {Data_analysis}
            ds, fit_results = fit_qubits(ds, qubits, node.parameters)
            node.results["fit_results"] = fit_results

            # %% {Plotting}
            fig = plot_qubit_response(ds, qubits, fit_results["fit_ds"])
            node.results["figure"] = fig

            # %% {Update_state}
            if node.parameters.load_data_id is None:
                with node.record_state_updates():
                    for q in qubits:
                        if fit_results[q.name]["fit_successful"]:
                            # Update the qubit IF
                            q.xy.intermediate_frequency += fit_results[q.name]["drive_freq"] - q.xy.RF_frequency
                            # Update the IW angle
                            q.resonator.operations["readout"].integration_weights_angle = fit_results[q.name]["angle"]
                            # Update the saturation amplitude
                            limits = instrument_limits(q.xy)
                            if fit_results[q.name]["saturation_amplitude"] < limits.max_wf_amplitude:
                                q.xy.operations["saturation"].amplitude = fit_results[q.name]["saturation_amplitude"]
                            else:
                                q.xy.operations["saturation"].amplitude = limits.max_wf_amplitude
                            # Update the expected x180 amplitude
                            if fit_results[q.name]["x180_amplitude"] < limits.max_x180_wf_amplitude:
                                q.xy.operations["x180"].amplitude = fit_results[q.name]["x180_amplitude"]
                            else:
                                q.xy.operations["x180"].amplitude = limits.max_x180_wf_amplitude
                node.results["ds"] = ds

                # {Save_results}
                node.outcomes = {q.name: "successful" for q in qubits}
                node.results["initial_parameters"] = node.parameters.model_dump()
                node.machine = machine
                node.save()
                
    def analyze_data(self, ds):
        return fit_qubits(ds, self.qubits, self.parameters)
    
    def plot_data(self, ds, fit_results):
    
    def run():
    
    @classmethod
    def load(cls, name):
        pass
    
    


node = QubitSpectroscopy()
node.run()

class QubitSpectroscopyVsFlux(QubitSpectroscopy):
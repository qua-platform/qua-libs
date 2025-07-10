# %%
from pathlib import Path
from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    qubits: List[str] = ["qubitC1", "qubitC2", "qubitC3"]



multiplexed = True
flux_point = "joint"
reset_type_thermal_or_active = "active"



g = QualibrationGraph(
    name="char_graph",
    parameters=Parameters(),
    nodes={
        "resonator_spectroscopy": library.nodes["02a_Resonator_Spectroscopy"].copy(multiplexed=multiplexed, 
                                                                                   name="resonator_spectroscopy"),
        "resonator_spec_vs_flux": library.nodes["02b_Resonator_Spectroscopy_vs_Flux"].copy(flux_point_joint_or_independent=flux_point, 
                                                                                           name="resonator_spec_vs_flux"),
        "qubit_spectroscopy": library.nodes["03a_Qubit_Spectroscopy"].copy(flux_point_joint_or_independent=flux_point,
                                                                           multiplexed=multiplexed,
                                                                           name="qubit_spectroscopy"),
        "qubit_spec_vs_flux": library.nodes["03b_Qubit_Spectroscopy_vs_Flux"].copy(flux_point_joint_or_independent=flux_point, 
                                                                                   multiplexed=multiplexed, 
                                                                                   name="qubit_spec_vs_flux"),
        "power_rabi": library.nodes["04_Power_Rabi"].copy(flux_point_joint_or_independent=flux_point, 
                                                          multiplexed=multiplexed, 
                                                          name="power_rabi"),
        "ramsey": library.nodes["06_Ramsey"].copy(flux_point_joint_or_independent=flux_point, 
                                                  multiplexed=multiplexed, 
                                                  name="ramsey"),
        "readout_frequency_optimization": library.nodes["07a_Readout_Frequency_Optimization"].copy(flux_point_joint_or_independent=flux_point, 
                                                                                                   multiplexed=multiplexed, 
                                                                                                   name="readout_frequency_optimization"),
        "IQ_blobs": library.nodes["07b_IQ_Blobs"].copy(flux_point_joint_or_independent=flux_point, 
                                                       multiplexed=multiplexed, 
                                                       name="IQ_blobs"),
        "EF_spectroscopy": library.nodes["11a_Qubit_Spectroscopy_E_to_F"].copy(flux_point_joint_or_independent=flux_point, 
                                                                     multiplexed=multiplexed, 
                                                                     name="EF_spectroscopy"),
        "T1": library.nodes["05_T1"].copy(name="T1"),
        "T2": library.nodes["05b_T2e"].copy(name="T2"),
    },
    connectivity=[
                  ("resonator_spectroscopy", "resonator_spec_vs_flux"),
                  ("resonator_spec_vs_flux", "qubit_spectroscopy"),
                  ("qubit_spectroscopy", "qubit_spec_vs_flux"),
                  ("qubit_spec_vs_flux", "power_rabi"),
                  ("power_rabi",  "ramsey"),
                  ("ramsey", "readout_frequency_optimization"),
                  ("readout_frequency_optimization", "IQ_blobs"),
                  ("IQ_blobs", "EF_spectroscopy"),
                  ("EF_spectroscopy", "T1"),
                  ("T1", "T2"),
                  ],

    orchestrator=BasicOrchestrator(skip_failed=True),
)
# %%

g.run()
# %%

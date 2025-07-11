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
    name="readout_graph",
    parameters=Parameters(),
    nodes={
        "readout_frequency_optimization": library.nodes["07a_Readout_Frequency_Optimization"].copy(flux_point_joint_or_independent=flux_point, 
                                                                                                   multiplexed=multiplexed, 
                                                                                                   name="readout_frequency_optimization"),
        "readout_power_optimization": library.nodes["07c_Readout_Power_Optimization"].copy(flux_point_joint_or_independent=flux_point, multiplexed=multiplexed, name="readout_power_optimization"),
        # GEF freq
        # GEF discirminator
        },
    connectivity=[
                  ("readout_frequency_optimization", "readout_power_optimization"),
                  ],

    orchestrator=BasicOrchestrator(skip_failed=True),
)
# %%

g.run()
# %%

# %%
from typing import List, Optional

from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    virtual_gate_set_id: str = "main_qpu"
    """Name of the VirtualGateSet to calibrate."""
    sensor_names: Optional[List[str]] = None
    """List of sensor dot names to use for readout."""


g = QualibrationGraph(
    name="GateVirtualization",
    parameters=Parameters(),
    nodes={
        # 1) Sensor Coulomb peak analysis + positioning at optimal operating point
        #    while keeping plunger sweeps centered for downstream 2D calibrations
        "sensor_coulomb_peak_alignment": library.nodes[
            "05_sensor_gate_sweep"
        ].copy(name="sensor_coulomb_peak_alignment"),
        # 2) Relative capacitive shift calibration at the sensor dot
        "sensor_gate_compensation": library.nodes[
            "01_sensor_gate_compensation"
        ].copy(name="sensor_gate_compensation"),
        # 3) Virtualized plunger calibration (nearest-neighbour 2D scans)
        "virtual_plunger_calibration": library.nodes[
            "02_virtual_plunger_calibration"
        ].copy(name="virtual_plunger_calibration"),
        # 4) Barrier compensation (kept as final stage)
        "barrier_compensation": library.nodes[
            "03_barrier_compensation"
        ].copy(name="barrier_compensation"),
    },
    connectivity=[
        ("sensor_coulomb_peak_alignment", "sensor_gate_compensation"),
        ("sensor_gate_compensation", "virtual_plunger_calibration"),
        ("virtual_plunger_calibration", "barrier_compensation"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run()

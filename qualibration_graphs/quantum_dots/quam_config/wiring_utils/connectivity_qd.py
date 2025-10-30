from typing import List
from qualang_tools.wirer.wirer.channel_specs import ChannelSpec

from qualang_tools.wirer.connectivity.connectivity_base import ConnectivityBase
from qualang_tools.wirer.connectivity.wiring_spec import WiringFrequency, WiringIOType

from .wiring_lines import WiringLineType

__all__ = ["QuantumDotConnectivity"]

class QuantumDotConnectivity(ConnectivityBase): 
    """
    High Level wiring for QD based architectures. 
    """

    def add_plunger_lines(self, dot_ids: List[int], constraints: ChannelSpec = None): 
        """Add plunger gate lines for QDs"""

        elements = [f"dot_{dot_id}" for dot_id in dot_ids]

        return self.add_wiring_spec(
            WiringFrequency.DC, 
            WiringIOType.OUTPUT, 
            WiringLineType.PLUNGER, 
            False, 
            constraints, 
            elements
        )
    
    def add_barrier_lines(self, barrier_ids: List[int], constraints: ChannelSpec = None): 
        """Add barrier gate lines"""

        elements = [f"barrier_{b_id}" for b_id in barrier_ids]

        return self.add_wiring_spec(
            WiringFrequency.DC, 
            WiringIOType.OUTPUT, 
            WiringLineType.BARRIER, 
            False, 
            constraints, 
            elements
        )
    
    def add_sensor_lines(self, sensor_ids: List[int], triggered: bool = False, constraints: ChannelSpec = None):
        """Add sensor lines"""
                    
        sensor_elements = [f"sensor_{sensor_id}" for sensor_id in sensor_ids]
        sensor_spec = self.add_wiring_spec(
            WiringFrequency.DC, 
            WiringIOType.OUTPUT, 
            WiringLineType.SENSOR,
            False, 
            constraints, 
            sensor_elements
        )
        

        resonator_elements = [f"resonator_{sensor_id}" for sensor_id in sensor_ids]
        resonator_spec = self.add_wiring_spec(
            WiringFrequency.RF, 
            WiringIOType.INPUT_AND_OUTPUT, 
            WiringLineType.RESONATOR, 
            triggered, 
            constraints, 
            resonator_elements,
            shared_line=True
        )
        
        return sensor_spec, resonator_spec








    
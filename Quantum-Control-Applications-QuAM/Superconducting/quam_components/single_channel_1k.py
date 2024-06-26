from typing import Union, Tuple, Literal, Dict, Any

from quam import quam_dataclass
from quam.components import SingleChannel

__all__ = ["SingleChannel1k"]


@quam_dataclass
class SingleChannel1k(SingleChannel):
    """QuAM component for a single (not IQ) output channel.

    Args:
        output_mode (str): The output voltage/frequency mode of the channel.
            "direct": 1 Vpp output range, 750 MHz bandwidth
            "amplified": 5 Vpp output range, 330 MHz bandwidth.
        upsampling_mode (str): The pulse optimization mode of the channel.
            "mw": optimized for microwave pulses
            "pulse": optimized for a clean step response

    """
    output_mode: str = None
    upsampling_mode: str = None

    def _add_analog_port_to_config(
        self,
        address: Union[Tuple[str, int], Tuple[str, int, int]],
        config,
        offset: float,
        port_type: Literal["input", "output"],
    ) -> Dict[str, Any]:
        """
        Add the analog port to the configuration, with some additional
        parameters specialized to the OPX1000 for a DC channel.
        """
        # start with the default port configuration for a single channel
        port = super()._add_analog_port_to_config(address, config, offset, port_type)

        if self.output_mode is not None:
            port["output_mode"] = self.output_mode

        if self.upsampling_mode is not None:
            port["upsampling_mode"] = self.upsampling_mode

        return port

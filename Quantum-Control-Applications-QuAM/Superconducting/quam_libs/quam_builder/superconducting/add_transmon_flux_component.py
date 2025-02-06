from typing import Dict, Union

from quam_libs.components_2.superconducting.architectural_elements.flux_line import FluxLine
from quam_libs.components_2.superconducting.qpu import FixedFrequencyTransmon, FluxTunableTransmon, BaseTransmon


def add_transmon_flux_component(transmon: Union[FixedFrequencyTransmon, FluxTunableTransmon, BaseTransmon], wiring_path: str, ports: Dict[str, str]):
    if "opx_output" in ports:
        transmon.z = FluxLine(opx_output=f"{wiring_path}/opx_output")
    else:
        raise ValueError(f"Unimplemented mapping of port keys to channel for ports: {ports}")

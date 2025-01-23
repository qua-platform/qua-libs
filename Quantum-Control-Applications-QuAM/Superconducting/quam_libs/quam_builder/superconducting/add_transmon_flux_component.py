from typing import Dict

from quam_libs.components import FluxLine, Transmon


def add_transmon_flux_component(transmon: Transmon, wiring_path: str, ports: Dict[str, str]):
    if "opx_output" in ports:
        transmon.z = FluxLine(opx_output=f"{wiring_path}/opx_output")
    else:
        raise ValueError(f"Unimplemented mapping of port keys to channel for ports: {ports}")

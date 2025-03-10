from typing import Dict, Union
from quam_builder.architecture.superconducting.components.flux_line import FluxLine
from quam_builder.architecture.superconducting.qubit import (
    FixedFrequencyTransmon,
    FluxTunableTransmon,
)


def add_transmon_flux_component(
    transmon: Union[FixedFrequencyTransmon, FluxTunableTransmon],
    wiring_path: str,
    ports: Dict[str, str],
):
    """
    Adds a flux component to a transmon qubit based on the provided wiring path and ports.

    Parameters:
    transmon (Union[FixedFrequencyTransmon, FluxTunableTransmon]): The transmon qubit to which the flux component will be added.
    wiring_path (str): The path to the wiring configuration.
    ports (Dict[str, str]): A dictionary mapping port names to their respective configurations.

    Raises:
    ValueError: If the port keys do not match any implemented mapping.
    """
    if "opx_output" in ports:
        transmon.z = FluxLine(opx_output=f"{wiring_path}/opx_output")
    else:
        raise ValueError(f"Unimplemented mapping of port keys to channel for ports: {ports}")

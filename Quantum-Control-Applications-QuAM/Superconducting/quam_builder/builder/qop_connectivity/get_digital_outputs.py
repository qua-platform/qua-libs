from quam.components import DigitalOutputChannel


def get_digital_outputs(wiring_path: str, ports: dict[str, str]) -> dict[str, DigitalOutputChannel]:
    """
    Generates a dictionary of digital output channels based on the provided wiring path and ports.

    Parameters:
    wiring_path (str): The path to the wiring configuration.
    ports (dict[str, str]): A dictionary mapping port names to their respective configurations.

    Returns:
    dict[str, DigitalOutputChannel]: A dictionary of digital output channels.
    """
    digital_outputs = dict()
    for i, item in enumerate([port for port in ports if "digital_output" in port]):
        digital_outputs[f"octave_switch_{i}"] = DigitalOutputChannel(
            opx_output=f"{wiring_path}/{item}",
            delay=57,  # 57ns for QOP222 and above
            buffer=18,  # 18ns for QOP222 and above
        )

    return digital_outputs

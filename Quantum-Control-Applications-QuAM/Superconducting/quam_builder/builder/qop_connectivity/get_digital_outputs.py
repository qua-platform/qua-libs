from quam.components import DigitalOutputChannel


def get_digital_outputs(wiring_path: str, ports: dict[str, str]) -> dict[str, DigitalOutputChannel]:
    digital_outputs = dict()
    for i, item in enumerate([port for port in ports if "digital_output" in port]):
        digital_outputs[f"octave_switch_{i}"] = DigitalOutputChannel(
            opx_output=f"{wiring_path}/{item}",
            delay=57,  # 57ns for QOP222 and above
            buffer=18,  # 18ns for QOP222 and above
        )

    return digital_outputs

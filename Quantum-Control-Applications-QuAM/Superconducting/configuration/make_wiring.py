import json


def default_port_allocation(num_qubits: int, using_opx_1000: bool, starting_fem: int = 1):
    """
    An example port allocation is generated in the following physical order on
    the numbered channels of the OPX and Octave:

    1. Assigns a channel for a single feed-line for the resonator.
    2. Assigns XY channels (I and Q) consecutively for each qubit.
    3. Assigns Z channel consecutively for each qubit.
    4. Assigns ZZ channels consecutively for each qubit.

    Notes:
    - Requires multiple OPX+ after 2 qubits.
    - Requires multiple octaves after 4 qubits.

    Returns:
    - res_ports: List of tuples (module, i_ch, octave, octave_ch) per qubit.
    - xy_ports: List of tuples (module, i_ch, octave, octave_ch) per qubit.
    - flux_ports: List of tuples (module, ch) per qubit.
    - coupler_ports: List of tuples (module, ch) per qubit.
    """
    xy_ports, flux_ports, coupler_ports, res_ports = [], [], [], []

    num_feedlines = 1
    num_channels_per_feedline = 2
    num_xy_channels_per_qubit = 2
    num_z_channels_per_qubit = 1
    num_coupler_channels_per_qubit = 1

    def allocate_module_port(idx):
        num_chs_per_module = 8 if using_opx_1000 else 10
        module = idx // num_chs_per_module + 1
        if using_opx_1000 and starting_fem != 1:
            module += starting_fem - 1
        ch = idx % num_chs_per_module + 1
        return module, ch

    def allocate_octave_port(idx):
        num_chs_per_octave = 10
        module = idx // num_chs_per_octave + 1
        ch = (idx % num_chs_per_octave) // 2 + 1
        return module, ch

    for q_idx in range(num_qubits):
        # Assign an absolute index for every channel for every qubit
        res_idx = 0  # only one feedline, so the resonator is always at the first two channels
        xy_idx = num_channels_per_feedline * num_feedlines + num_xy_channels_per_qubit * q_idx
        z_idx = xy_idx + num_xy_channels_per_qubit * (num_qubits - q_idx) + num_z_channels_per_qubit * q_idx
        coupler_idx = z_idx + num_z_channels_per_qubit * (num_qubits - q_idx) + num_coupler_channels_per_qubit * q_idx

        # Allocate a port for each index according to the OPX+ or OPX1000 channel layouts
        res_module, res_ch = allocate_module_port(res_idx)
        xy_module, xy_ch = allocate_module_port(xy_idx)
        z_module, z_ch = allocate_module_port(z_idx)
        coupler_module, coupler_ch = allocate_module_port(coupler_idx)
        # Note: For I/Q channels, only the I-quadrature channel is returned, but both I/Q are accounted for.

        # Assign the octave ports for the XY channels
        res_octave, res_octave_ch = allocate_octave_port(res_idx)
        xy_octave, xy_octave_ch = allocate_octave_port(xy_idx)

        res_ports.append((res_module, res_ch, res_octave, res_octave_ch))
        xy_ports.append((xy_module, xy_ch, xy_octave, xy_octave_ch))
        flux_ports.append((z_module, z_ch))
        coupler_ports.append((coupler_module, coupler_ch))

    return res_ports, xy_ports, flux_ports, coupler_ports


def custom_port_allocation(wiring: dict):
    """
    Convert the override dictionary to a tuple of lists for port allocation.

    Then, use it to create a proper wiring dictionary.

    Args:
    - wiring_dict: Dictionary with custom port allocation per qubit.

    Returns:
    - res_ports: List of tuples (module, i_ch, octave, octave_ch) per qubit.
    - xy_ports: List of tuples (module, i_ch, octave, octave_ch) per qubit.
    - flux_ports: List of tuples (module, ch) per qubit.
    - coupler_ports: List of tuples (module, ch) per qubit.
    """
    res_ports, xy_ports, flux_ports, coupler_ports = [], [], [], []

    num_qubits = len(wiring["qubits"])
    for q_idx in range(1, num_qubits + 1):
        q_key = f"q{q_idx}"
        if q_key in wiring["qubits"]:
            res_ports.append(wiring["qubits"][q_key]["res"])
            xy_ports.append(wiring["qubits"][q_key]["xy"])
            flux_ports.append(wiring["qubits"][q_key]["flux"])
        else:
            raise ValueError(f"Override dictionary does not contain ports for qubit {q_idx}")

    for q_idx in range(1, num_qubits):
        q_pair_key = f"q{q_idx}{q_idx + 1}"
        if q_pair_key in wiring["qubit_pairs"]:
            coupler_ports.append(wiring["qubit_pairs"][q_pair_key]["coupler"])
        else:
            raise ValueError(f"Override dictionary does not contain coupler ports for qubit pair {q_idx}{q_idx + 1}")

    return res_ports, xy_ports, flux_ports, coupler_ports


def create_wiring(port_allocation, using_opx_1000: bool) -> dict:
    """
    Create a wiring config tailored to the number of qubits.
    """
    wiring = {"qubits": {}, "qubit_pairs": []}

    # Generate example wiring by default
    res_ports, xy_ports, flux_ports, coupler_ports = port_allocation

    num_qubits = len(port_allocation[0])
    for q_idx in range(0, num_qubits):
        if using_opx_1000:
            wiring["qubits"][f"q{q_idx}"] = create_qubit_wiring_opx1000(
                xy_ports=xy_ports[q_idx], res_ports=res_ports[q_idx], flux_ports=flux_ports[q_idx], con="con1"
            )
        else:
            wiring["qubits"][f"q{q_idx}"] = create_qubit_wiring_opx_plus(
                xy_ports=xy_ports[q_idx], res_ports=res_ports[q_idx], flux_ports=flux_ports[q_idx]
            )

    for q_idx in range(num_qubits - 1):
        if using_opx_1000:
            qubit_pair_wiring = create_qubit_pair_wiring_opx1000(
                coupler_ports=coupler_ports[q_idx],
                qubit_control=q_idx,
                qubit_target=q_idx + 1,
            )
        else:
            qubit_pair_wiring = create_qubit_pair_wiring_opx_plus(
                coupler_ports=coupler_ports[q_idx],
                qubit_control=q_idx,
                qubit_target=q_idx + 1,
            )
        wiring["qubit_pairs"].append(qubit_pair_wiring)

    return wiring


def create_qubit_wiring_opx1000(xy_ports, res_ports, flux_ports, con="con1"):
    res_module, res_i_ch_out, res_octave, res_octave_ch = res_ports
    xy_module, xy_i_ch, xy_octave, xy_octave_ch = xy_ports
    z_module, z_ch = flux_ports

    # Note: The Q channel is set to the I channel plus one.
    return {
        "xy": {
            "opx_output_I": f"#/ports/analog_outputs/{con}/{xy_module}/{xy_i_ch}",
            "opx_output_Q": f"#/ports/analog_outputs/{con}/{xy_module}/{xy_i_ch+1}",
            "digital_port": f"#/ports/digital_outputs/{con}/{xy_module}/{xy_i_ch}",
            "frequency_converter_up": f"#/octaves/octave{xy_octave}/RF_outputs/{xy_octave_ch}",
        },
        "z": {"opx_output": f"#/ports/analog_outputs/{con}/{z_module}/{z_ch}"},
        "resonator": {
            "opx_output_I": f"#/ports/analog_outputs/{con}/{res_module}/{res_i_ch_out}",
            "opx_output_Q": f"#/ports/analog_outputs/{con}/{res_module}/{res_i_ch_out+1}",
            "opx_input_I": f"#/ports/analog_inputs/{con}/{res_module}/1",
            "opx_input_Q": f"#/ports/analog_inputs/{con}/{res_module}/2",
            "digital_port": f"#/ports/digital_outputs/{con}/{res_module}/{res_i_ch_out}",
            "frequency_converter_up": "#/octaves/octave1/RF_outputs/1",
            "frequency_converter_down": "#/octaves/octave1/RF_inputs/1",
        },
    }


def create_qubit_pair_wiring_opx1000(coupler_ports, qubit_control, qubit_target, con="con1"):
    c_module, c_ch = coupler_ports

    return {
        "qubit_control": f"#/qubits/q{qubit_control}",  # reference to f"q{q_idx}"
        "qubit_target": f"#/qubits/q{qubit_target}",  # reference to f"q{q_idx + 1}"
        "coupler": {"opx_output": f"#/ports/analog_outputs/{con}/{c_module}/{c_ch}"},
    }


def create_qubit_wiring_opx_plus(xy_ports, res_ports, flux_ports):
    res_module, res_i_ch_out, res_octave, res_octave_ch = res_ports
    xy_module, xy_i_ch, xy_octave, xy_octave_ch = xy_ports
    z_module, z_ch = flux_ports

    # Note: The Q channel is set to the I channel plus one.
    return {
        "xy": {
            "opx_output_I": f"#/ports/analog_outputs/{xy_module}/{xy_i_ch}",
            "opx_output_Q": f"#/ports/analog_outputs/{xy_module}/{xy_i_ch+1}",
            "digital_port": f"#/ports/digital_outputs/{xy_module}/{xy_i_ch}",
            "frequency_converter_up": f"#/octaves/octave{xy_octave}/RF_outputs/{xy_octave_ch}",
        },
        "z": {"opx_output": f"#/ports/analog_outputs/{z_module}/{z_ch}"},
        "resonator": {
            "opx_output_I": f"#/ports/analog_outputs/{res_module}/{res_i_ch_out}",
            "opx_output_Q": f"#/ports/analog_outputs/{res_module}/{res_i_ch_out+1}",
            "opx_input_I": f"#/analog_inputs/{res_module}/1",
            "opx_input_Q": f"#/analog_inputs/{res_module}/2",
            "digital_port": f"#/ports/digital_outputs/{res_module}/{res_i_ch_out}",
            "frequency_converter_up": "#/octaves/octave1/RF_outputs/1",
            "frequency_converter_down": "#/octaves/octave1/RF_inputs/1",
        },
    }


def create_qubit_pair_wiring_opx_plus(coupler_ports, qubit_control, qubit_target, con="con1"):
    c_module, c_ch = coupler_ports

    return {
        "qubit_control": f"#/qubits/q{qubit_control}",  # reference to f"q{q_idx}"
        "qubit_target": f"#/qubits/q{qubit_target}",  # reference to f"q{q_idx + 1}"
        "coupler": {"opx_output": f"#/ports/analog_outputs/{c_module}/{c_ch}"},
    }

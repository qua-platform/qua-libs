def create_default_port_allocation(num_qubits: int, using_opx_1000: bool):
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


def create_default_wiring(num_qubits: int, using_opx_1000: bool = True) -> dict:
    """
    Create a wiring config tailored to the number of qubits.
    """
    wiring = {"qubits": {}, "qubit_pairs": []}

    def port(module: int, ch: int):
        """
        Generate the port tuple for the OPX+/OPX1000 according to the templates
        - OPX+     (con, ch)
        - OPX1000  (con, fem, ch)
        The argument `module` refers to either the:
         "con" index if using the OPX+, or the
         "fem" index if using the OPX1000 (for a single OPX1000 only).
        """
        return (f"con1", module, ch) if using_opx_1000 else (f"con{module}", ch)

    # Generate example wiring by default
    res_ports, xy_ports, flux_ports, coupler_ports = create_default_port_allocation(num_qubits, using_opx_1000)

    for q_idx in range(0, num_qubits):
        res_module, res_i_ch_out, res_octave, res_octave_ch = res_ports[q_idx]
        xy_module, xy_i_ch, xy_octave, xy_octave_ch = xy_ports[q_idx]
        z_module, z_ch = flux_ports[q_idx]
        coupler_module, coupler_ch = flux_ports[q_idx]

        # Note: The Q channel is set to the I channel plus one.
        wiring["qubits"][f"q{q_idx}"] = {
            "xy": {
                "opx_output_I": port(xy_module, xy_i_ch),
                "opx_output_Q": port(xy_module, xy_i_ch + 1),
                "frequency_converter_up": f"#/octaves/octave{xy_octave}/RF_outputs/{xy_octave_ch}",
            },
            "z": {"opx_output": port(z_module, z_ch)},
            "coupler": {"opx_output": port(coupler_module, coupler_ch)},
            "opx_output_digital": port(xy_module, xy_i_ch),
            "resonator": {
                "opx_output_I": port(res_module, res_i_ch_out),
                "opx_output_Q": port(res_module, res_i_ch_out + 1),
                "opx_input_I": port(res_module, 1),
                "opx_input_Q": port(res_module, 2),
                "digital_port": port(res_module, res_i_ch_out),
                "frequency_converter_up": "#/octaves/octave1/RF_outputs/1",
                "frequency_converter_down": "#/octaves/octave1/RF_inputs/1",
            },
        }

    for q_idx in range(num_qubits - 1):
        c_opx, c_ch = coupler_ports[q_idx]

        wiring["qubit_pairs"].append({
            "qubit_control": f"#/qubits/q{q_idx}", # reference to f"q{q_idx}"
            "qubit_target": f"#/qubits/q{q_idx + 1}", # reference to f"q{q_idx + 1}"
            "coupler": {"opx_output": (c_opx, c_ch)},
        })

    return wiring

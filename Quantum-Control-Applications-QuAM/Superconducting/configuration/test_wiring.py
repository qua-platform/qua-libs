from enum import Enum


custom_port_wiring = {
    # TODO: this should be enough to build the wiring automatically and manually
    "cluster": {"opx+": 0, "octaves": 2, "opx1000": 1, "lf-fems": 3, "mw-fems": 0, "octave_connectivity": "default"},
    # opx+/octave with default connectivity --> res:  (controller, octave, octave_ch)
    #                                           xy:   (controller, octave, octave_ch)
    #                                           flux: (controller, opx_ch)
    # opx+/octave with custom connectivity  --> res:  (controller, opx_I_ch, opx_Q_ch, octave, octave_ch, opx_trig_ch)
    #                                           xy:   (controller, opx_I_ch, opx_Q_ch, octave, octave_ch, opx_trig_ch)
    #                                           flux: (controller, opx_ch)
    # opx1000/octave with default connectivity --> res:  (controller, fem, octave, octave_ch)
    #                                              xy:   (controller, fem, octave, octave_ch)
    #                                              flux: (controller, fem, lf_ch)
    # opx1000/octave with custom connectivity  --> res:  (controller, fem, lf_I_ch, lf_Q_ch, octave, octave_ch, opx_trig_ch)
    #                                              xy:   (controller, fem, lf_I_ch, lf_Q_ch, octave, octave_ch, opx_trig_ch)
    #                                              flux: (controller, fem, lf_ch)
    # opx1000/mw-fem --> res:  (controller, fem, mw_ch)
    #                    xy:   (controller, fem, mw_ch)
    #                    flux: (controller, fem, lf_ch)
    "qubits": {
        "q1": {
            "res": (1, 1, 1, 1),
            "xy": (1, 3, 1, 2),
            "flux": (1, 2, 5),
        },
        "q2": {
            "res": (1, 1, 1, 1),
            "xy": (1, 5, 1, 3),
            "flux": (1, 3, 1),
        },
        "q3": {
            "res": (1, 1, 1, 1),
            "xy": (1, 7, 1, 4),
            "flux": (1, 3, 2),
        },
        "q4": {
            "res": (1, 1, 1, 1),
            "xy": (2, 1, 2, 1),
            "flux": (1, 3, 3),
        },
        "q5": {
            "res": (1, 1, 1, 1),
            "xy": (2, 3, 2, 2),
            "flux": (1, 3, 4),
        },
    },
    "qubit_pairs": {
        # (module, ch)
        "q12": {"coupler": (3, 5)},
        "q23": {"coupler": (3, 6)},
        "q34": {"coupler": (3, 7)},
        "q45": {"coupler": (3, 8)},
    },
}

custom_port_wiring = {
    "cluster": {"opx+": 1, "octaves": 1, "opx1000": 0, "lf-fems": 0, "mw-fems": 0, "octave_connectivity": "default"},
    "qubits": {
        'q1': {
            'res': (1, 1, 1),
            'xy': (1, 1, 2),
            'flux': (1, 4),
        },
        'q2': {
            'res': (1, 1, 1),
            'xy': (1, 1, 4),
            'flux': (1, 5),
        },
    },
    # TODO: not sure how to do this for with and without couplers
    "qubit_pairs": {
        "q12": {"coupler": (1, 4)},
    },
}

class CLUSTER_CONFIG(Enum):
    """A dictionary with the filter limitations for the QOP versions"""

    OPX_OCTAVES_DEFAULT = {
        "res": 3,
        "xy": 3,
        "flux": 2,
        "mapping": "opx+ & octave with default connectivity: "
                   "\n\t'q1': {"
                   "\n\t\t'res': (controller, octave, octave_ch),"
                   "\n\t\t'xy': (controller, octave, octave_ch),"
                   "\n\t\t'flux': (controller, opx_ch),"
                   "\n\t},"
    }
    OPX_OCTAVES_CUSTOM = {
        "res": 6,
        "xy": 6,
        "flux": 2,
        "mapping": "opx+ & octave with custom connectivity: "
                   "\n\t'q1': {"
                   "\n\t\t'res': (controller, opx_I_ch, opx_Q_ch, octave, octave_ch, opx_trig_ch)"
                   "\n\t\t'xy': (controller, opx_I_ch, opx_Q_ch, octave, octave_ch, opx_trig_ch)"
                   "\n\t\t'flux': (controller, opx_ch),"
                   "\n\t},"
    }
    OPX1000_OCTAVES_DEFAULT = {
        "res": 4,
        "xy": 4,
        "flux": 3,
        "mapping": "opx1000 & octaves with default connectivity: "
                   "\n\t'q1': {"
                   "\n\t\t'res': (controller, fem, octave, octave_ch)"
                   "\n\t\t'xy': (controller, fem, octave, octave_ch)"
                   "\n\t\t'flux': (controller, fem, opx_ch)"
                   "\n\t},"

    }
    OPX1000_OCTAVES_CUSTOM = {
        "res": 7,
        "xy": 7,
        "flux": 3,
        "mapping": "opx1000 & octaves with custom connectivity: "
                   "\n\t'q1': {"
                   "\n\t\t'res': (controller, fem, opx_I_ch, opx_Q_ch, octave, octave_ch, opx_trig_ch)"
                   "\n\t\t'xy': (controller, fem, opx_I_ch, opx_Q_ch, octave, octave_ch, opx_trig_ch)"
                   "\n\t\t'flux': (controller, fem, opx_ch)"
                   "\n\t},"
    }
    OPX1000_MW = {
        "res": 3,
        "xy": 3,
        "flux": 3,
        "mapping": "opx1000 & MW-FEMs: "
                   "\n\t'q1': {"
                   "\n\t\t'res': (controller, fem, mw_ch)"
                   "\n\t\t'xy': (controller, fem, mw_ch)"
                   "\n\t\t'flux': (controller, fem, lf_ch)"
                   "\n\t},"
    }
    @classmethod
    def get_options(cls):
        """Return the list of implemented QOP versions"""
        print(x.value["mapping"] for x in cls.value)

def check_wiring_format(wiring: dict, cluster_config: CLUSTER_CONFIG):
    cluster = wiring["cluster"]
    # Only opx+ or opx1000, but not both
    assert cluster["opx+"] * cluster["opx1000"] == 0, "A cluster can't contain both OPX+ and OPX1000."
    assert cluster["opx+"] * cluster["lf-fems"] == 0, "A cluster can't contain both OPX+ and OPX1000 LF-FEMs."
    assert cluster["opx+"] * cluster["mw-fems"] == 0, "A cluster can't contain both OPX+ and OPX1000 MW-FEMs."
    # Only opx+/octave or lf-fem/octave, but not mw-fem/octave
    assert cluster["octaves"] * cluster["mw-fems"] == 0, "A cluster can't contain both Octaves and OPX1000 MW-FEMs."
    # Get the cluster configuration and the expected wiring format
    if cluster["opx+"] > 0 and cluster["octaves"] > 0 and cluster["octave_connectivity"] == "default":
        cfg = cluster_config.OPX_OCTAVES_DEFAULT
    elif cluster["opx+"] > 0 and cluster["octaves"] > 0 and cluster["octave_connectivity"] == "custom":
        cfg = cluster_config.OPX_OCTAVES_CUSTOM
    elif cluster["opx1000"] > 0 and cluster["lf-fems"] > 0 and cluster["octaves"] > 0 and cluster["octave_connectivity"] == "default":
        cfg = cluster_config.OPX1000_OCTAVES_DEFAULT
    elif cluster["opx1000"] > 0 and cluster["lf-fems"] > 0 and cluster["octaves"] > 0 and cluster["octave_connectivity"] == "custom":
        cfg = cluster_config.OPX1000_OCTAVES_CUSTOM
    elif cluster["opx1000"] > 0 and cluster["mw-fems"] > 0:
        cfg = cluster_config.OPX1000_MW
    else:
        raise ValueError("The cluster configuration is invalid!")
    # Check the keys and wiring format:
    for q in wiring["qubits"]:
        assert "res" in wiring["qubits"][
            q], "The wiring of each qubit must contain the resonator connectivity via the key 'res'."
        assert "xy" in wiring["qubits"][
            q], "The wiring of each qubit must contain the xy-drive connectivity via the key 'xy'."
        assert "flux" in wiring["qubits"][
            q], "The wiring of each qubit must contain the flux-line connectivity via the key 'flux'."
        assert len(wiring["qubits"][q]["res"]) == cfg.value["res"], f"The dimension of 'res' must be {cfg.value['res']} for {cfg.name}, which is not the case for {q}.\nEx: {cfg.value['mapping']}"
        assert len(wiring["qubits"][q]["xy"]) == cfg.value["xy"], f"The dimension of 'xy' must be {cfg.value['xy']} for {cfg.name}, which is not the case for {q}.\nEx: {cfg.value['mapping']}"
        assert len(wiring["qubits"][q]["flux"]) == cfg.value["flux"], f"The dimension of 'flux' must be {cfg.value['flux']} for {cfg.name}, which is not the case for {q}.\nEx: {cfg.value['mapping']}"
    # TODO: same for qubit pairs
    return cfg


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

    for q_key in wiring["qubits"]:
        res_ports.append(wiring["qubits"][q_key]["res"])
        xy_ports.append(wiring["qubits"][q_key]["xy"])
        flux_ports.append(wiring["qubits"][q_key]["flux"])

    for q_pair_key in wiring["qubit_pairs"]:
        coupler_ports.append(wiring["qubit_pairs"][q_pair_key]["coupler"])

    return res_ports, xy_ports, flux_ports, coupler_ports

def create_qubit_wiring_opx_octave_default(xy_ports, res_ports, flux_ports):
    res_module, res_octave, res_octave_ch = res_ports
    xy_module, xy_octave, xy_octave_ch = xy_ports
    z_module, z_ch = flux_ports
    if res_octave_ch > 1:
        res_octave_ch_in = 2
        print("WARNING: the resonator is driven through an Octave port that is not 1, make sure that the down-conversion is done properly.")
    else:
        res_octave_ch_in = 1

    return {
        "xy": {
            "opx_output_I": f"#/ports/analog_outputs/con{xy_module}/{2 * (xy_octave_ch - 1) + 1}",
            "opx_output_Q": f"#/ports/analog_outputs/con{xy_module}/{2 * xy_octave_ch}",
            "digital_port": f"#/ports/digital_outputs/con{xy_module}/{2 * (xy_octave_ch - 1) + 1}",
            "frequency_converter_up": f"#/octaves/octave{xy_octave}/RF_outputs/{xy_octave_ch}",
        },
        "z": {"opx_output": f"#/ports/analog_outputs/con{z_module}/{z_ch}"},
        "resonator": {
            "opx_output_I": f"#/ports/analog_outputs/con{res_module}/{2 * (res_octave_ch - 1) + 1}",
            "opx_output_Q": f"#/ports/analog_outputs/con{res_module}/{2 * res_octave_ch}",
            "opx_input_I": f"#/analog_inputs/{res_module}/1",
            "opx_input_Q": f"#/analog_inputs/{res_module}/2",
            "digital_port": f"#/ports/digital_outputs/con{res_module}/{2 * (res_octave_ch - 1) + 1}",
            "frequency_converter_up": f"#/octaves/octave{xy_octave}/RF_outputs/{xy_octave_ch}",
            "frequency_converter_down": f"#/octaves/octave{xy_octave}/RF_inputs/{res_octave_ch_in}",
        },
    }

def create_qubit_wiring_opx_octave_custom(xy_ports, res_ports, flux_ports):
    res_module, res_I, res_Q, res_octave, res_octave_ch, res_opx_trig_ch = res_ports
    xy_module, xy_I, xy_Q, xy_octave, xy_octave_ch, xy_opx_trig_ch = xy_ports
    z_module, z_ch = flux_ports
    if res_octave_ch > 1:
        res_octave_ch_in = 2
        print("WARNING: the resonator is driven through an Octave port that is not 1, make sure that the down-conversion is done properly.")
    else:
        res_octave_ch_in = 1

    return {
        "xy": {
            "opx_output_I": f"#/ports/analog_outputs/con{xy_module}/{xy_I}",
            "opx_output_Q": f"#/ports/analog_outputs/con{xy_module}/{xy_Q}",
            "digital_port": f"#/ports/digital_outputs/con{xy_module}/{xy_opx_trig_ch}",
            "frequency_converter_up": f"#/octaves/octave{xy_octave}/RF_outputs/{xy_octave_ch}",
        },
        "z": {"opx_output": f"#/ports/analog_outputs/con{z_module}/{z_ch}"},
        "resonator": {
            "opx_output_I": f"#/ports/analog_outputs/con{res_module}/{res_I}",
            "opx_output_Q": f"#/ports/analog_outputs/con{res_module}/{res_Q}",
            "opx_input_I": f"#/analog_inputs/{res_module}/1",
            "opx_input_Q": f"#/analog_inputs/{res_module}/2",
            "digital_port": f"#/ports/digital_outputs/con{res_module}/{res_opx_trig_ch}",
            "frequency_converter_up": f"#/octaves/octave{xy_octave}/RF_outputs/{xy_octave_ch}",
            "frequency_converter_down": f"#/octaves/octave{xy_octave}/RF_inputs/{res_octave_ch_in}",
        },
    }

def create_qubit_wiring_opx1000_octave_default(xy_ports, res_ports, flux_ports):
    res_chassis, res_module, res_octave, res_octave_ch = res_ports
    xy_chassis, xy_module, xy_octave, xy_octave_ch = xy_ports
    z_chassis, z_module, z_ch = flux_ports

    # Note: The Q channel is set to the I channel plus one.
    return {
        "xy": {
            "opx_output_I": f"#/ports/analog_outputs/con{xy_chassis}/{xy_module}/{2 * (xy_octave_ch - 1) + 1}",
            "opx_output_Q": f"#/ports/analog_outputs/con{xy_chassis}/{xy_module}/{2 * xy_octave_ch}",
            "digital_port": f"#/ports/digital_outputs/con{xy_chassis}/{xy_module}/{2 * (xy_octave_ch - 1) + 1}",
            "frequency_converter_up": f"#/octaves/octave{xy_octave}/RF_outputs/{xy_octave_ch}",
        },
        "z": {"opx_output": f"#/ports/analog_outputs/{z_chassis}/{z_module}/{z_ch}"},
        "resonator": {
            "opx_output_I": f"#/ports/analog_outputs/con{res_chassis}/{res_module}/{2 * (res_octave_ch - 1) + 1}",
            "opx_output_Q": f"#/ports/analog_outputs/con{res_chassis}/{res_module}/{2 * xy_octave_ch}",
            "opx_input_I": f"#/ports/analog_inputs/con{res_chassis}/{res_module}/1",
            "opx_input_Q": f"#/ports/analog_inputs/con{res_chassis}/{res_module}/2",
            "digital_port": f"#/ports/digital_outputs/con{res_chassis}/{res_module}/{2 * (res_octave_ch - 1) + 1}",
            "frequency_converter_up": "#/octaves/octave1/RF_outputs/1",
            "frequency_converter_down": "#/octaves/octave1/RF_inputs/1",
        },
    }

def create_qubit_wiring_opx1000_octave_custom(xy_ports, res_ports, flux_ports):
    res_chassis, res_module, res_I, res_Q, res_octave, res_octave_ch, res_opx_trig_ch = res_ports
    xy_chassis, xy_module, xy_I, xy_Q, xy_octave, xy_octave_ch, xy_opx_trig_ch = xy_ports
    z_chassis, z_module, z_ch = flux_ports

    # Note: The Q channel is set to the I channel plus one.
    return {
        "xy": {
            "opx_output_I": f"#/ports/analog_outputs/con{xy_chassis}/{xy_module}/{xy_I}",
            "opx_output_Q": f"#/ports/analog_outputs/con{xy_chassis}/{xy_module}/{xy_Q}",
            "digital_port": f"#/ports/digital_outputs/con{xy_chassis}/{xy_module}/{xy_opx_trig_ch}",
            "frequency_converter_up": f"#/octaves/octave{xy_octave}/RF_outputs/{xy_octave_ch}",
        },
        "z": {"opx_output": f"#/ports/analog_outputs/{z_chassis}/{z_module}/{z_ch}"},
        "resonator": {
            "opx_output_I": f"#/ports/analog_outputs/con{res_chassis}/{res_module}/{res_I}",
            "opx_output_Q": f"#/ports/analog_outputs/con{res_chassis}/{res_module}/{res_Q}",
            "opx_input_I": f"#/ports/analog_inputs/con{res_chassis}/{res_module}/1",
            "opx_input_Q": f"#/ports/analog_inputs/con{res_chassis}/{res_module}/2",
            "digital_port": f"#/ports/digital_outputs/con{res_chassis}/{res_module}/{res_opx_trig_ch}",
            "frequency_converter_up": "#/octaves/octave1/RF_outputs/1",
            "frequency_converter_down": "#/octaves/octave1/RF_inputs/1",
        },
    }


def create_wiring(port_wiring, cluster_config_class: CLUSTER_CONFIG) -> dict:
    """
    Create a wiring config tailored to the number of qubits.
    """
    wiring = {"qubits": {}, "qubit_pairs": []}

    # Validate the wiring and return the cluster configuration
    cluster_config = check_wiring_format(port_wiring, cluster_config_class)
    # Generate the allocated ports
    res_ports, xy_ports, flux_ports, coupler_ports = custom_port_allocation(port_wiring)
    # Build the quam connectivity
    for q_idx, q_name in enumerate(port_wiring["qubits"]):
        if cluster_config.name == 'OPX_OCTAVES_DEFAULT':
            wiring["qubits"][q_name] = create_qubit_wiring_opx_octave_default(
                xy_ports=xy_ports[q_idx], res_ports=res_ports[q_idx], flux_ports=flux_ports[q_idx])
        elif cluster_config.name == 'OPX_OCTAVES_CUSTOM':
            wiring["qubits"][q_name] = create_qubit_wiring_opx_octave_custom(
                xy_ports=xy_ports[q_idx], res_ports=res_ports[q_idx], flux_ports=flux_ports[q_idx])
        elif cluster_config.name == 'OPX1000_OCTAVES_DEFAULT':
            wiring["qubits"][q_name] = create_qubit_wiring_opx1000_octave_default(
                xy_ports=xy_ports[q_idx], res_ports=res_ports[q_idx], flux_ports=flux_ports[q_idx])
        elif cluster_config.name == 'OPX1000_OCTAVES_CUSTOM':
            wiring["qubits"][q_name] = create_qubit_wiring_opx1000_octave_custom(
                xy_ports=xy_ports[q_idx], res_ports=res_ports[q_idx], flux_ports=flux_ports[q_idx])
        elif cluster_config.name == 'OPX1000_MW':
            raise RuntimeError("Not implemented yet")

    # TODO: this seems to assume that you always have (N-1) qubit pairs for N qubits --> not always true
    # TODO: what about when there is no couplers like always coupled flux tunable transmons?
    num_qubit_pairs = len(coupler_ports)
    # for q_idx in range(num_qubit_pairs):
    #     if using_opx_1000:
    #         qubit_pair_wiring = create_qubit_pair_wiring_opx1000(
    #             coupler_ports=coupler_ports[q_idx],
    #             qubit_control=q_idx,
    #             qubit_target=q_idx + 1,
    #         )
    #     else:
    #         qubit_pair_wiring = create_qubit_pair_wiring_opx_plus(
    #             coupler_ports=coupler_ports[q_idx],
    #             qubit_control=q_idx,
    #             qubit_target=q_idx + 1,
    #         )
    #     wiring["qubit_pairs"].append(qubit_pair_wiring)
    return wiring

# wiring = create_wiring(custom_port_wiring, CLUSTER_CONFIG)

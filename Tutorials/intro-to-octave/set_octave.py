"""
set_octave.py: script for setting all the octave parameters
"""
from qm.octave import *
import os
from qm.octave import QmOctaveConfig
from qm.octave.octave_manager import ClockMode
from dataclasses import dataclass
from typing import Union


class OctavesSettings:
    """Class for keeping track of OctavesSettings in inventory."""

    def __init__(
        self,
        name: str,
        ip: str,
        port: int = 50,
        clock: str = "Internal",
        con: str = "con1",
        port_mapping: Union[str, list] = "default",
    ):
        """Class for keeping track of OctavesSettings in inventory.

        :param name: Name of the Octave.
        :param ip: IP address of the router to which the Octave is connected.
        :param port: Port of the Octave.
        :param clock: Clock setting of the Octave. Can be "Internal", "External_10MHz", "External_100MHz" or "External_1000MHz"
        :param con: Controller to which the Octave is connected. Only used when port mapping set to default.
        :param port_mapping: Port mapping of the Octave. Default mapping is set with mapping="default", otherwise the custom mapping must be specified as a list of dictionary where each key as the following format: ('con1',  1) : ('octave1', 'I1').
        """
        self.name = name
        self.ip = ip
        self.port = port
        self.con = con
        self.clock = clock
        self.port_mapping = port_mapping


@dataclass
class ElementsSettings:
    """Class for keeping track of ElementsSettings in inventory."""

    def __init__(
        self,
        name: str,
        lo_source: str = "Internal",
        gain: int = 0,
        switch_mode: str = "on",
        rf_in_port: Union[None, list] = None,
        down_convert_LO_source: Union[None, str] = None,
        if_mode: str = "direct",
    ):
        """Class for keeping track of ElementsSettings in inventory.

        :param name: Name of the element.
        :param lo_source: LO source. Can be "Internal" (then the LO frequency is the one defined in the config) or "LO1", "LO2"... if the external LO is provided in the Octave back panel.
        :param gain: Octave output gain. Can be within [-10 : 0.5: 20] dB.
        :param switch_mode: RF switch mode of the Octave. Can be "on", "off", "trig_normal" or "trig_inverse".
        :param rf_in_port: RF input port of the Octave if the element is used to measure signals from the fridge. The syntax is [octave_name, RF input port] as in ["octave1", 1].
        :param down_convert_LO_source: LO source for the down-conversion mixer if the element is used to measure signals from the fridge. Can be "Internal", "Dmd1LO" or "Dmd2LO".
        :param if_mode: Specify the IF down-conversion mode. Can be "direct" for standard down-conversion, "envelope" for reading the signals from the envelope detector inside the Octave, or "mixer" TODO:??.
        """
        self.name = name
        self.lo_source = lo_source
        self.gain = gain
        self.switch_mode = switch_mode
        self.rf_in_port = rf_in_port
        self.down_convert_LO_source = down_convert_LO_source
        self.if_mode = if_mode


def octave_configuration(octaves_settings: list = ()):
    """
    Initiate octave_config class, set the calibration file, add octaves info and set the port mapping between the OPX and the octaves.

    :param octaves_settings: objects that holds the information about octave's name, the controller that is connected to this octave, octave's ip octave's port and octave's clock settings
    """
    octave_config = QmOctaveConfig()
    octave_config.set_calibration_db(os.getcwd())
    for i in range(len(octaves_settings)):
        if octaves_settings[i].name is None:
            raise TypeError(f"Please insert the octave name for the {i}'s octave")
        if octaves_settings[i].con is None:
            raise TypeError(f"Please insert the controller that is connected to the {i}'s octave")
        if octaves_settings[i].ip is None:
            raise TypeError(f"Please insert the octave ip for the {i}'s octave")
        if octaves_settings[i].port is None:
            raise TypeError(f"Please insert the octave port for the {i}'s octave")
        octave_config.add_device_info(octaves_settings[i].name, octaves_settings[i].ip, octaves_settings[i].port)
        if octaves_settings[i].port_mapping == "default":
            octave_config.set_opx_octave_mapping([(octaves_settings[i].con, octaves_settings[i].name)])
        else:
            octave_config.add_opx_octave_port_mapping(octaves_settings[i].port_mapping)

    return octave_config


def octave_settings(qmm, config, octaves_settings, elements_settings=None, calibration=True):
    """
    Set all the octave settings including: clock, up-converters modules, down-converters modules and calibration according to the elements_settings.

    :param qmm: Quantum Machines Manager object.
    :param config: The QM configuration.
    :param octaves_settings: objects that holds the information about octave's name, the controller that is connected to this octave, octave's ip octave's port and octave's clock settings.
    :param elements_settings: objects that holds the information about the up-converter and down-converter parameters.
    :param calibration: When True (default) calibrates all the octave elements.
    """

    # Open a quantum machine
    qm = qmm.open_qm(config)

    # setting the clock
    for i in range(len(octaves_settings)):
        if octaves_settings[i].clock == "External_10MHz":
            qm.octave.set_clock(octaves_settings[i].name, clock_mode=ClockMode.External_10MHz)
        elif octaves_settings[i].clock == "External_100MHz":
            qm.octave.set_clock(octaves_settings[i].name, clock_mode=ClockMode.External_100MHz)
        elif octaves_settings[i].clock == "External_1000MHz":
            qm.octave.set_clock(octaves_settings[i].name, clock_mode=ClockMode.External_1000MHz)
        else:
            qm.octave.set_clock(octaves_settings[i].name, clock_mode=ClockMode.Internal)

    if elements_settings is None:
        elements_settings = []

    # User defined values
    for i in range(len(elements_settings)):
        # Set LO source
        if elements_settings[i].lo_source == "LO1":
            qm.octave.set_lo_source(elements_settings[i].name, OctaveLOSource.LO1)
        elif elements_settings[i].lo_source == "LO2":
            qm.octave.set_lo_source(elements_settings[i].name, OctaveLOSource.LO2)
        elif elements_settings[i].lo_source == "LO3":
            qm.octave.set_lo_source(elements_settings[i].name, OctaveLOSource.LO3)
        elif elements_settings[i].lo_source == "LO4":
            qm.octave.set_lo_source(elements_settings[i].name, OctaveLOSource.LO4)
        elif elements_settings[i].lo_source == "LO5":
            qm.octave.set_lo_source(elements_settings[i].name, OctaveLOSource.LO5)
        elif elements_settings[i].lo_source == "Internal":
            qm.octave.set_lo_source(elements_settings[i].name, OctaveLOSource.Internal)
            qm.octave.set_lo_frequency(
                elements_settings[i].name, config["elements"][elements_settings[i].name]["mixInputs"]["lo_frequency"]
            )
        else:
            raise ValueError(
                f"Please insert LO source for element {elements_settings[i].name}. Valid values are 'Internal', 'LO1', 'LO2', 'LO3', 'LO4' and 'LO5'."
            )

        # Set gain
        qm.octave.set_rf_output_gain(elements_settings[i].name, elements_settings[i].gain)

        # Set switch mode
        if elements_settings[i].switch_mode == "trig_normal":
            qm.octave.set_rf_output_mode(elements_settings[i].name, RFOutputMode.trig_normal)
        elif elements_settings[i].switch_mode == "trig_inverse":
            qm.octave.set_rf_output_mode(elements_settings[i].name, RFOutputMode.trig_inverse)
        elif elements_settings[i].switch_mode == "off":
            qm.octave.set_rf_output_mode(elements_settings[i].name, RFOutputMode.off)
        elif elements_settings[i].switch_mode == "on":
            qm.octave.set_rf_output_mode(elements_settings[i].name, RFOutputMode.on)
        else:
            raise ValueError(
                f"Please insert switch_mode for element {elements_settings[i].name}. Valid values are 'on', 'off', 'trig_normal' and 'trig_inverse'."
            )

        # Set down-conversion modules
        if elements_settings[i].rf_in_port is not None:
            if elements_settings[i].rf_in_port[1] == 1:
                qm.octave.set_qua_element_octave_rf_in_port(
                    elements_settings[i].name, elements_settings[i].rf_in_port[0], 1
                )
                if elements_settings[i].down_convert_LO_source == "Internal":
                    if elements_settings[i].if_mode == "direct":
                        qm.octave.set_downconversion(
                            elements_settings[i].name,
                            lo_source=RFInputLOSource.Internal,
                            if_mode_i=IFMode.direct,
                            if_mode_q=IFMode.direct,
                        )

                    elif elements_settings[i].if_mode == "envelope":
                        qm.octave.set_downconversion(
                            elements_settings[i].name,
                            lo_source=RFInputLOSource.Internal,
                            if_mode_i=IFMode.envelope,
                            if_mode_q=IFMode.envelope,
                        )
                    elif elements_settings[i].if_mode == "mixer":
                        qm.octave.set_downconversion(
                            elements_settings[i].name,
                            lo_source=RFInputLOSource.Internal,
                            if_mode_i=IFMode.mixer,
                            if_mode_q=IFMode.mixer,
                        )
                    else:
                        raise ValueError(
                            f"Please insert if_mode for element {elements_settings[i].name}. Valid values are 'direct', 'envelope' and 'mixer'."
                        )

                elif elements_settings[i].down_convert_LO_source == "Dmd1LO":
                    if elements_settings[i].if_mode == "direct":
                        qm.octave.set_downconversion(
                            elements_settings[i].name,
                            lo_source=RFInputLOSource.Dmd1LO,
                            if_mode_i=IFMode.direct,
                            if_mode_q=IFMode.direct,
                        )
                    elif elements_settings[i].if_mode == "envelope":
                        qm.octave.set_downconversion(
                            elements_settings[i].name,
                            lo_source=RFInputLOSource.Dmd1LO,
                            if_mode_i=IFMode.envelope,
                            if_mode_q=IFMode.envelope,
                        )
                    elif elements_settings[i].if_mode == "mixer":
                        qm.octave.set_downconversion(
                            elements_settings[i].name,
                            lo_source=RFInputLOSource.Dmd1LO,
                            if_mode_i=IFMode.mixer,
                            if_mode_q=IFMode.mixer,
                        )
                    else:
                        raise ValueError(
                            f"Please insert if_mode for element {elements_settings[i].name}. Valid values are 'direct', 'envelope' and 'mixer'."
                        )
                else:
                    raise TypeError(
                        f"Please insert Down converter LO for element {elements_settings[i].name}. Valid values are 'Internal' and 'Dmd1LO'."
                    )

            elif elements_settings[i].rf_in_port[1] == 2:
                qm.octave.set_qua_element_octave_rf_in_port(
                    elements_settings[i].name, elements_settings[i].rf_in_port[0], 2
                )
                if elements_settings[i].if_mode == "direct":
                    qm.octave.set_downconversion(
                        elements_settings[i].name,
                        lo_source=RFInputLOSource.Dmd2LO,
                        if_mode_i=IFMode.direct,
                        if_mode_q=IFMode.direct,
                    )
                elif elements_settings[i].if_mode == "envelope":
                    qm.octave.set_downconversion(
                        elements_settings[i].name,
                        lo_source=RFInputLOSource.Dmd2LO,
                        if_mode_i=IFMode.envelope,
                        if_mode_q=IFMode.envelope,
                    )
                elif elements_settings[i].if_mode == "mixer":
                    qm.octave.set_downconversion(
                        elements_settings[i].name,
                        lo_source=RFInputLOSource.Dmd2LO,
                        if_mode_i=IFMode.mixer,
                        if_mode_q=IFMode.mixer,
                    )
                else:
                    raise ValueError(
                        f"Please insert if_mode for element {elements_settings[i].name}. Valid values are 'direct', 'envelope' and 'mixer'."
                    )
            # Check down_convert_LO_source compatibility with
            if elements_settings[i].rf_in_port[1] == 2 and (
                elements_settings[i].down_convert_LO_source == "Internal"
                or elements_settings[i].down_convert_LO_source == "Dmd1LO"
            ):
                raise ValueError(f"Only down_convert_LO_source='Dmd2LO' is valid for rf_in_port=2.")
            if elements_settings[i].rf_in_port[1] == 1 and elements_settings[i].down_convert_LO_source == "Dmd2LO":
                raise ValueError(f"down_convert_LO_source='Dmd2LO' is not valid for rf_in_port=1.")

    # calibrate all the elements
    if calibration:
        for i in range(len(elements_settings)):
            LO = float(config["elements"][elements_settings[i].name]["mixInputs"]["lo_frequency"])
            IF = float(config["elements"][elements_settings[i].name]["intermediate_frequency"])
            print(
                "-" * 37
                + f" Calibrates {elements_settings[i].name} for (LO, IF) = ({LO*1e-9:.3f} GHz, {IF*1e-6:.3f} MHz)"
            )
            qm.octave.calibrate_element(elements_settings[i].name, [(LO, IF)])

def disable_octave_ports(qm, element_list: list=()):
    """
    Switch off the Octave synthesizers corresponding to the element list.

    :param qm: The quantum machine object.
    :param element_list: List of the elements to switch off.
    """
    for el in element_list:
        qm.octave.set_lo_source(el, OctaveLOSource.Off)
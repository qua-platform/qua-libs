"""
set_octave.py: script for setting all the octave parameters
"""
from qm.octave import *
import os
import numpy as np
from qm import generate_qua_script
from qm.octave import QmOctaveConfig
import re
from qm.elements.element_with_octave import ElementWithOctave
from qm.octave.octave_manager import ClockMode
from dataclasses import dataclass, field
from typing import Optional, Dict


class RegularClass:
    pass


@dataclass
class OctavesSettings:
    """Class for keeping track of OctavesSettings in inventory."""
    name: Optional[str] = None
    con: Optional[str] = None
    ip: Optional[str] = None
    port: Optional[int] = None
    clock: str = "Internal"

@dataclass
class ElementsSettings:
    """Class for keeping track of ElementsSettings in inventory."""
    name: Optional[str] = None
    LO_source: str = "Internal"
    gain: int = 0
    switch_mode: str = "on"
    RF_in_port: Optional[int] = None
    Down_convert_LO_source: Optional[list] = None
    IF_mode: Optional[str] = "direct"

def get_elements_used_in_octave(qm=None, config=None, prog=None):
    """
    Extract the elements used in program that are connected to the octave
    :param qm: Quantum Machine object
    :param prog: QUA program object
    :return: a list of elements which are used in the program and are connected to the octave
    """

    if qm is None:
        raise "Can not find qm object"
    if config is None:
        raise "Can not find configuration"
    if prog is None:
        raise "Can not find QUA program object"

    # make a list of all the elements in the program
    elements_in_prog = []
    for element in list(config["elements"].keys()):
        if (
            re.search(f'(?<="){element}', generate_qua_script(prog)) is not None
            and re.search(f'{element}(?=")', generate_qua_script(prog)) is not None
        ):
            elements_in_prog.append(element)

    # get the elements that are connected to the octave
    elements_used_in_octave = []
    for element in elements_in_prog:
        element_i = qm.elements[element]
        if isinstance(element_i, ElementWithOctave):
            elements_used_in_octave.append(element)

    return np.array(elements_used_in_octave)


def octave_configuration(octaves_settings=None, port_mapping=True):
    """
    Initiate octave_config class, set the calibration file, add octaves info and set the port mapping between the OPX and the octaves.

    :param port_mapping: When set to True (default) the default port mapping will be set
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
        if port_mapping == True:
            octave_config.set_opx_octave_mapping([(octaves_settings[i].con, octaves_settings[i].name)])
        else:
            port_mapping = port_mapping[i]
            octave_config.add_opx_octave_port_mapping(port_mapping)

    return octave_config

def octave_settings(qmm, qm, prog, config,  octaves_settings, elements_settings=None, calibration=True):
    """
    Set all the octave settings including: clock, up-converters modules, down-converters modules and calibration.
    The default parameters are internal LO for the up-conversion modules with a 0dB gain and RFOutputMode.on.
    The LO of the down-conversion for RFin1 is set to internal and RFin2 is set to port Dmd2LO.

    :param qmm: Quantum Machines Manager object
    :param qm: Quantum Machine object
    :param prog: The QUA program
    :param config: The QM configuration
    :param octaves_settings: ojects that holds the information about octave's name, the controller that is connected to this octave, octave's ip octave's port and octave's clock settings
    :param elements_settings: objects that holds the information about the up-converter and down-converter parameters
    :param calibration: When True (default) calibrates all the elements in the program
    """
    #####################
    # setting the clock #
    #####################
    for i in range(len(octaves_settings)):
        if octaves_settings[i].clock == "External_10MHz":
            qm.octave.set_clock(octaves_settings[i].name, clock_mode=ClockMode.External_10MHz)
        elif octaves_settings[i].clock == "External_100MHz":
            qm.octave.set_clock(octaves_settings[i].name, clock_mode=ClockMode.External_100MHz)
        elif octaves_settings[i].clock == "External_1000MHz":
            qm.octave.set_clock(octaves_settings[i].name, clock_mode=ClockMode.External_1000MHz)
        else:
            qm.octave.set_clock(octaves_settings[i].name, clock_mode=ClockMode.Internal)

    ##############################################################
    # extracting octave elements and their LO and IF frequencies #
    ##############################################################
    octave_elements = get_elements_used_in_octave(qm=qm, config=config, prog=prog)
    lo_freq = [config["elements"][octave_elements[i]]["mixInputs"]["lo_frequency"] for i in range(len(octave_elements))]

    if elements_settings is None:
        elements_settings = []
    #################################
    # setting up-converters modules #
    #################################
    for i in range(len(octave_elements)):
        if octave_elements[i] not in [elements_settings[j].name for j in range(len(elements_settings))]:
            qm.octave.set_lo_source(octave_elements[i], OctaveLOSource.Internal)  # Can change to external
            qm.octave.set_lo_frequency(octave_elements[i], lo_freq[i])
            qm.octave.set_rf_output_gain(octave_elements[i], 0)
            qm.octave.set_rf_output_mode(octave_elements[i], RFOutputMode.on)
        else:
            pass
    for i in range(len(elements_settings)):
        if elements_settings[i].LO_source=="LO1":
            qm.octave.set_lo_source(elements_settings[i].name, OctaveLOSource.LO1)
        elif elements_settings[i].LO_source=="LO2":
            qm.octave.set_lo_source(elements_settings[i].name, OctaveLOSource.LO2)
        elif elements_settings[i].LO_source=="LO3":
            qm.octave.set_lo_source(elements_settings[i].name, OctaveLOSource.LO3)
        elif elements_settings[i].LO_source=="LO4":
            qm.octave.set_lo_source(elements_settings[i].name, OctaveLOSource.LO4)
        elif elements_settings[i].LO_source=="LO5":
            qm.octave.set_lo_source(elements_settings[i].name, OctaveLOSource.LO5)
        elif elements_settings[i].LO_source=="Internal":
            qm.octave.set_lo_source(elements_settings[i].name, OctaveLOSource.Internal)
            qm.octave.set_lo_frequency(elements_settings[i].name,
                                       config["elements"][elements_settings[i].name]["mixInputs"]["lo_frequency"])
        else:
            raise TypeError(f"Please insert LO source for element {elements_settings[i].name}")

        qm.octave.set_rf_output_gain(elements_settings[i].name, elements_settings[i].gain)

        if elements_settings[i].switch_mode=="trig_normal":
            qm.octave.set_rf_output_mode(elements_settings[i].name, RFOutputMode.trig_normal)
        elif elements_settings[i].switch_mode=="trig_inverse":
            qm.octave.set_rf_output_mode(elements_settings[i].name, RFOutputMode.trig_inverse)
        elif elements_settings[i].switch_mode=="off":
            qm.octave.set_rf_output_mode(elements_settings[i].name, RFOutputMode.off)
        else:
            qm.octave.set_rf_output_mode(elements_settings[i].name, RFOutputMode.on)
    ###################################
    # setting down-converters modules #
    ###################################
    for i in range(len(octave_elements)):
        if octave_elements[i] not in [elements_settings[j].name for j in range(len(elements_settings))]:
            for element_name in octave_elements:
                element = qm.elements[element_name]
                if isinstance(element, ElementWithOctave):
                    # This assumes that: FR1in measures RF1's output (which is connected to Analog output 1 and 2), FR2in measures RF2's output (which is connected to Analog output 3 and 4)
                    if (element.q_port.number == 1 or element.q_port.number == 2) and "outputs" in config["elements"][
                        element_name
                    ].keys():
                        qm.octave.set_qua_element_octave_rf_in_port(element_name, "octave1", 1)
                        qm.octave.set_downconversion(
                            element_name, lo_source=RFInputLOSource.Internal, if_mode_i=IFMode.direct, if_mode_q=IFMode.direct
                        )
                    if (element.q_port.number == 3 or element.q_port.number == 4) and "outputs" in config["elements"][
                        element_name
                    ].keys():
                        qm.octave.set_qua_element_octave_rf_in_port(element_name, "octave1", 2)
                        qm.octave.set_downconversion(
                            element_name, lo_source=RFInputLOSource.Dmd2LO, if_mode_i=IFMode.direct, if_mode_q=IFMode.direct
                        )  # Don't forget to connect external LO to Dmd2LO or Synth2 from back panel
        else:
            pass
    for i in range(len(elements_settings)):
        if elements_settings[i].RF_in_port[1] == 1:
            qm.octave.set_qua_element_octave_rf_in_port(elements_settings[i].name,
                                                        elements_settings[i].RF_in_port[0], 1)
            if elements_settings[i].Down_convert_LO_source == "Internal":
                if elements_settings[i].IF_mode == "direct":
                    qm.octave.set_downconversion(elements_settings[i].name,
                                                 lo_source=RFInputLOSource.Internal, if_mode_i=IFMode.direct,
                                                 if_mode_q=IFMode.direct)

                elif elements_settings[i].IF_mode == "envelope":
                    qm.octave.set_downconversion(elements_settings[i].name,
                                                 lo_source=RFInputLOSource.Internal, if_mode_i=IFMode.envelope,
                                                 if_mode_q=IFMode.envelope)
                else:
                    qm.octave.set_downconversion(elements_settings[i].name,
                                                 lo_source=RFInputLOSource.Internal, if_mode_i=IFMode.mixer,
                                                 if_mode_q=IFMode.mixer)
            elif elements_settings[i].Down_convert_LO_source == "Dmd1LO":
                if elements_settings[i].IF_mode == "direct":
                    qm.octave.set_downconversion(elements_settings[i].name,
                                                 lo_source=RFInputLOSource.Dmd1LO, if_mode_i=IFMode.direct,
                                                 if_mode_q=IFMode.direct)
                elif elements_settings[i].IF_mode == "envelope":
                    qm.octave.set_downconversion(elements_settings[i].name,
                                                 lo_source=RFInputLOSource.Dmd1LO, if_mode_i=IFMode.envelope,
                                                 if_mode_q=IFMode.envelope)
                else:
                    qm.octave.set_downconversion(elements_settings[i].name,
                                                 lo_source=RFInputLOSource.Dmd1LO, if_mode_i=IFMode.mixer,
                                                 if_mode_q=IFMode.mixer)
            else:
                raise TypeError(f"Please insert Down converter LO for element {elements_settings[i].name}")
        elif elements_settings[i].RF_in_port[1] == 2:
            qm.octave.set_qua_element_octave_rf_in_port(elements_settings[i].name, elements_settings[i].RF_in_port[0], 2)
            if elements_settings[i].IF_mode == "direct":
                qm.octave.set_downconversion(elements_settings[i].name,
                                             lo_source=RFInputLOSource.Dmd2LO, if_mode_i=IFMode.direct,
                                             if_mode_q=IFMode.direct)
            elif elements_settings[i].IF_mode == "envelope":
                qm.octave.set_downconversion(elements_settings[i].name,
                                             lo_source=RFInputLOSource.Dmd2LO, if_mode_i=IFMode.envelope,
                                             if_mode_q=IFMode.envelope)
            else:
                qm.octave.set_downconversion(elements_settings[i].name,
                                             lo_source=RFInputLOSource.Dmd2LO, if_mode_i=IFMode.mixer,
                                             if_mode_q=IFMode.mixer)

    #########################################################################
    # calibrate all the elements in the program that are used by the octave #
    #########################################################################
    if calibration:
        for i in range(len(octave_elements)):
            if octave_elements[i] not in [elements_settings[j].name for j in range(len(elements_settings))]:
                print("-" * 37 + f" Calibrates {octave_elements[i]}")
                qm.octave.calibrate_element(octave_elements[i], [(float(config["elements"][octave_elements[i]]["mixInputs"]["lo_frequency"]), float(config["elements"][octave_elements[i]]["intermediate_frequency"]))])
                qm = qmm.open_qm(config)
            else:
                pass
        for i in range(len(elements_settings)):
            print("-" * 37 + f" Calibrates {elements_settings[i].name}")
            qm.octave.calibrate_element(elements_settings[i].name, [(float(config["elements"][elements_settings[i].name]["mixInputs"]["lo_frequency"]), float(config["elements"][elements_settings[i].name]["intermediate_frequency"]))])
            qm = qmm.open_qm(config)
    return qmm, qm

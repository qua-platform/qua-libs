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


def get_elements_used_in_octave(qm=None, config=None, octave_config=None, prog=None):
    """
    Extract the elements used in program that are connected to the octave
    :param qm: Quantum Machine object
    :param octave_config: octave configuration
    :param prog: QUA program object
    :return: a list of elements which are used in the program and are connected to the octave
    """

    if qm is None:
        raise "Can not find qm object"
    if config is None:
        raise "Can not find configuration"
    if octave_config is None:
        raise "Can not find octave configuration"
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


def octave_configuration(default_port_mapping=True, more_than_one_octave=False, octave_ip=None, octave_port=None):
    """
    Initiate octave_config class, set the calibration file, add octaves info and set the port mapping between the OPX and the octaves.

    :param default_port_mapping: When set to True (default) the default port mapping will be set
    :param more_than_one_octave: When set to False (default) octave_config object will get information for a single octave
    :param octave_ip: IP address of the Octave (usually it is the IP address of the router to which the Octave is connected)
    :param octave_port: Port of the Octave (usually 50 for octave1, 51 for octave2...)
    :return: octave_config object
    """
    if octave_ip is None:
        raise "Please insert the octave ip"
    if octave_port is None:
        raise "Please insert the octave port"

    octave_config = QmOctaveConfig()
    octave_config.set_calibration_db(
        os.getcwd()
    )  # Saves the `calibration_db` file in the current working directory. Can change this by giving a specific path
    octave_config.add_device_info("octave1", octave_ip, octave_port)

    ##########################################################
    # define the port mapping between the OPX and the octave #
    ##########################################################
    if default_port_mapping:
        octave_config.set_opx_octave_mapping([("con1", "octave1")])
    else:
        portmap = {
            ("con1", 1): ("octave1", "I1"),
            ("con1", 2): ("octave1", "Q1"),
            ("con1", 3): ("octave1", "I2"),
            ("con1", 4): ("octave1", "Q2"),
            ("con1", 5): ("octave1", "I3"),
            ("con1", 6): ("octave1", "Q3"),
            ("con1", 7): ("octave1", "I4"),
            ("con1", 8): ("octave1", "Q4"),
            ("con1", 9): ("octave1", "I5"),
            ("con1", 10): ("octave1", "Q5"),
        }
        octave_config.add_opx_octave_port_mapping(portmap)
    #################################################
    # add the other octaves to octave_config object #
    #################################################
    if more_than_one_octave:
        octave_2_port = 51  # Insert the relevant port
        octave_config.add_device_info("octave2", octave_ip, octave_2_port)  # if having more than one octave
        default_port_mapping_octave_2 = True
        if default_port_mapping_octave_2:
            octave_config.set_opx_octave_mapping([("con2", "octave2")])
        else:
            portmap = {
                ("con1", 1): ("octave2", "I1"),
                ("con1", 2): ("octave2", "Q1"),
                ("con1", 3): ("octave2", "I2"),
                ("con1", 4): ("octave2", "Q2"),
                ("con1", 5): ("octave2", "I3"),
                ("con1", 6): ("octave2", "Q3"),
                ("con1", 7): ("octave2", "I4"),
                ("con1", 8): ("octave2", "Q4"),
                ("con1", 9): ("octave2", "I5"),
                ("con1", 10): ("octave2", "Q5"),
            }
            octave_config.add_opx_octave_port_mapping(portmap)

    return octave_config


def octave_settings(qmm, qm, prog, config, octave_config, external_clock=False, calibration=True):
    """
    Set all the octave settings including: clock, up-converters modules, down-converters modules and calibration.
    The default parameters are internal LO for the up-conversion modules with a 0dB gain and RFOutputMode.on.
    The LO of the down-conversion for RFin1 is set to internal and RFin2 is set to port Dmd2LO.

    :param qmm: Quantum Machines Manager object
    :param qm: Quantum Machine object
    :param prog: The QUA program
    :param octave_config: octave_config object
    :param external_clock: When False (default) sets the clock to be internal.
    :param calibration: When True (default) calibrates all the elements in the program
    """
    #####################
    # setting the clock #
    #####################
    if external_clock:
        # Change to the relevant external frequency
        qmm.octave_manager.set_clock("octave1", ClockType.External, ClockFrequency.MHZ_10)
        # If using a clock from the OPT, use this command instead
        # qmm.octave_manager.set_clock(octave, ClockType.Buffered, ClockFrequency.MHZ_1000)
    else:
        qmm.octave_manager.set_clock("octave1", ClockType.Internal, ClockFrequency.MHZ_10)

    ##############################################################
    # extracting octave elements and their LO and IF frequencies #
    ##############################################################
    octave_elements = get_elements_used_in_octave(qm=qm, config=config, octave_config=octave_config, prog=prog)
    lo_freq = [config["elements"][octave_elements[i]]["mixInputs"]["lo_frequency"] for i in range(len(octave_elements))]

    #################################
    # setting up-converters modules #
    #################################
    for i in range(len(octave_elements)):
        qm.octave.set_lo_source(octave_elements[i], OctaveLOSource.Internal)  # Can change to external
        qm.octave.set_lo_frequency(octave_elements[i], lo_freq[i])
        qm.octave.set_rf_output_gain(octave_elements[i], 0)
        # Set the behaviour of the RF switch to be on. Can change it to : off, trig_normal, trig_inverse
        qm.octave.set_rf_output_mode(octave_elements[i], RFOutputMode.on)

    ###################################
    # setting down-converters modules #
    ###################################
    for elements in octave_elements:
        element_i = qm.elements[elements]
        if isinstance(element_i, ElementWithOctave):
            # This assumes that: FR1in measures RF1's output (which is connected to Analog output 1 and 2), FR2in measures RF2's output (which is connected to Analog output 3 and 4)
            if (element_i.q_port == 1 or element_i.q_port == 2) and 'outputs' in \
                    config['elements'][octave_elements[i]].keys():
                qm.octave.set_qua_element_octave_rf_in_port(octave_elements[i], "octave1", 1)
                qm.octave.set_downconversion(octave_elements[i], lo_source=RFInputLOSource.Internal,
                                             if_mode_i=IFMode.direct, if_mode_q=IFMode.direct)
            if (element_i.q_port == 3 or element_i.q_port == 4) and 'outputs' in \
                    config['elements'][octave_elements[i]].keys():
                qm.octave.set_qua_element_octave_rf_in_port(octave_elements[i], "octave1", 2)
                qm.octave.set_downconversion(octave_elements[i], lo_source=RFInputLOSource.Dmd2LO,
                                             if_mode_i=IFMode.direct,
                                             if_mode_q=IFMode.direct)  # Don't forget to connect external LO to Dmd2LO or Synth2 from back panel

    #########################################################################
    # calibrate all the elements in the program that are used by the octave #
    #########################################################################
    if calibration:
        if_freq = [
            config["elements"][octave_elements[i]]["intermediate_frequency"] for i in range(len(octave_elements))
        ]
        for i in range(len(octave_elements)):
            print("-" * 37 + f" Calibrates {octave_elements[i]}")
            qm.octave.calibrate_element(octave_elements[i], [(float(lo_freq[i]), float(if_freq[i]))])
            qm = qmm.open_qm(config)
    return qmm, qm

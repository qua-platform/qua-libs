"""
set_octave.py: script for setting all the octave parameters
"""
from qm.octave import *
import os
import numpy as np
from qm import generate_qua_script
from qm.octave import QmOctaveConfig
import re
# from configuration import *


def get_elements_used_in_octave(qm=None, octave_config=None, prog=None):
    """
    Extract the elements used in program that are connected to the octave
    :param qm: Quantum Machine object
    :param octave_config: octave configuration
    :param prog: QUA program object
    :return: a list of elements which are used in the program and are connected to the octave
    """

    if qm is None:
        raise ('Can not find qm object')
    if octave_config is None:
        raise ('Can not find octave configuration')
    if prog is None:
        raise ('Can not find QUA program object')

    # make a list of all the elements in the program
    elements_in_prog = []
    for element in list(qm.get_config()['elements'].keys()):
        if re.search(f'(?<="){element}', generate_qua_script(prog)) is not None and re.search(f'{element}(?=")',
                                                                                              generate_qua_script(
                                                                                                  prog)) is not None:
            elements_in_prog.append(element)

    # get the elements that are connected to the octave
    elements_used_in_octave = []
    config = qm.get_config()
    for element in elements_in_prog:
        if 'mixInputs' in config['elements'][element].keys() and qm.octave._get_element_opx_output(element)[0] in list(
                QmOctaveConfig.get_opx_octave_port_mapping(octave_config)):
            elements_used_in_octave.append(element)

    return np.array(elements_used_in_octave)


def octave_configuration(default_port_mapping=True, more_than_one_octave=False, octave_ip=None, octave_port=None):
    """
    Initiate octave_config class, set the calibration file, add octaves info and set the port mapping between the OPX and the octaves.

    :param default_port_mapping: When set to True (default) the default port mapping will be set
    :param more_than_one_octave: When set to False (default) octave_config object will get information for one octave devive
    :return: octave_config ocject
    """
    if octave_ip is None:
        raise ('Please inser octave ip')
    if octave_port is None:
        raise ('Please inser octave port')

    octave_config = QmOctaveConfig()
    octave_config.set_calibration_db(os.getcwd()) # Saves the `calibration_db` file in the current working directory. Can chnage this by giving a spesipic path
    octave_config.add_device_info('octave1', octave_ip, octave_port)

    ##########################################################
    # define the port mapping between the OPX and the octave #
    ##########################################################
    if default_port_mapping:
        octave_config.set_opx_octave_mapping([('con1', 'octave1')])
    else:
        portmap = {('con1', 1): ('octave1', 'I1'), ('con1', 2): ('octave1', 'Q1'), ('con1', 3): ('octave1', 'I2'),
                   ('con1', 4): ('octave1', 'Q2'), ('con1', 5): ('octave1', 'I3'), ('con1', 6): ('octave1', 'Q3'),
                   ('con1', 7): ('octave1', 'I4'), ('con1', 8): ('octave1', 'Q4'), ('con1', 9): ('octave1', 'I5'),
                   ('con1', 10): ('octave1', 'Q5')}
        octave_config.add_opx_octave_port_mapping(portmap)
    #################################################
    # add the other octaves to octave_config object #
    #################################################
    if more_than_one_octave:
        octave_2_port = None # Insert the relevant port
        octave_config.add_device_info('octave2', octave_ip, octave_2_port) # if having more than one octave
        default_port_mapping_octave_2 = True
        if default_port_mapping_octave_2:
            octave_config.set_opx_octave_mapping([('con2', 'octave2')])
        else:
            portmap = {('con1', 1): ('octave2', 'I1'), ('con1', 2): ('octave2', 'Q1'), ('con1', 3): ('octave2', 'I2'),
                       ('con1', 4): ('octave2', 'Q2'), ('con1', 5): ('octave2', 'I3'), ('con1', 6): ('octave2', 'Q3'),
                       ('con1', 7): ('octave2', 'I4'), ('con1', 8): ('octave2', 'Q4'), ('con1', 9): ('octave2', 'I5'),
                       ('con1', 10): ('octave2', 'Q5')}
            octave_config.add_opx_octave_port_mapping(portmap)

    return octave_config


def octave_settings(qmm, qm, prog, octave_config, external_clock=False, calibration=True):
    """
    Set all the octave settings including: clock, up-converters modules, down-converters modules

    :param qmm: Quantum Machines Manager object
    :param qm: Quantum Machine object
    :param prog: The QUA program
    :param octave_config: octave_config object
    :param external_clock: When False (default) sets the clock to be internal.
                        When external_clock='10MHz' sets the clock to be external and expects to get 10MHz
                        When external_clock='100MHz' sets the clock to be external and expects to get 100MHz
                        When external_clock='1000MHz' or external_clock='1GHz' sets the clock to be external and expects to get 1GHz
    :param calibration: When True (default) calibrates all the elements in the program
    :return: doesn't return anything
    """
    #####################
    # setting the clock #
    #####################
    if external_clock == '10MHz':
        qmm.octave_manager.set_clock("octave1", ClockType.External, ClockFrequency.MHZ_10)
    elif external_clock == '100MHz':
        qmm.octave_manager.set_clock("octave1", ClockType.External, ClockFrequency.MHZ_100)
    elif external_clock == '1000MHz' or external_clock == '1GHz':
        qmm.octave_manager.set_clock("octave1", ClockType.External, ClockFrequency.MHZ_1000)
    else:
        qmm.octave_manager.set_clock("octave1", ClockType.Internal, ClockFrequency.MHZ_10)

    ##############################################################
    # extracting octave elements and their LO and IF frequencies #
    ##############################################################
    octave_elements = get_elements_used_in_octave(qm=qm, octave_config=octave_config, prog=prog)
    config = qm.get_config()
    lo_freq = [config['elements'][octave_elements[i]]['mixInputs']['lo_frequency'] for i in range(len(octave_elements))]

    #################################
    # setting up-converters modules #
    #################################
    for i in range(len(octave_elements)):
        qm.octave.set_lo_source(octave_elements[i], OctaveLOSource.Internal) # Can change to external
        qm.octave.set_lo_frequency(octave_elements[i], lo_freq[i])
        qm.octave.set_rf_output_gain(octave_elements[i], 0)
        qm.octave.set_rf_output_mode(octave_elements[i], RFOutputMode.on) # Set the behaviour of the RF switch to be on. Can change it to : off, trig_normal, trig_inverse

    ###################################
    # setting down-converters modules #
    ###################################
    for i in range(len(octave_elements)):
        # This assumes that: FR1in measures RF1's output, FR2in measures RF2's output
        if qm.octave._get_element_octave_output_port(octave_elements[i])[1] == 1 and 'outputs' in config['elements'][octave_elements[i]].keys():
            qm.octave.set_qua_element_octave_rf_in_port(octave_elements[i], "octave1", 1)
            qm.octave.set_downconversion(octave_elements[i])
            qm.octave.set_downconversion(octave_elements[i], lo_source=RFInputLOSource.Internal) # Can change to Dmd1LO
        if qm.octave._get_element_octave_output_port(octave_elements[i])[1] == 2 and 'outputs' in config['elements'][octave_elements[i]].keys():
            qm.octave.set_qua_element_octave_rf_in_port(octave_elements[i], "octave1", 2)
            qm.octave.set_downconversion(octave_elements[i])
            qm.octave.set_downconversion(octave_elements[i], lo_source=RFInputLOSource.Dmd2LO) # Don't forget to connect external LO to Dmd2LO or Synth2 from back panel

    #########################################################################
    # calibrate all the elements in the program that are used by the octave #
    #########################################################################
    if calibration:
        if_freq = [config['elements'][octave_elements[i]]['intermediate_frequency'] for i in
                   range(len(octave_elements))]
        for i in range(len(octave_elements)):
            qm.octave.calibrate_element(octave_elements[i], [(float(lo_freq[i]), float(if_freq[i]))])
            qm = qmm.open_qm(config)

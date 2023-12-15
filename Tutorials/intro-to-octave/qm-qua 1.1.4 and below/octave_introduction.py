"""
octave_introduction.py: shows the basic commands to control the octave's clock, synthesizers, up-converters, triggers,
down-converters and calibration
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.octave import *
from qm.octave.octave_manager import ClockMode
from qm.qua import *
import os
import time
import matplotlib.pyplot as plt
from qualang_tools.units import unit

# Flags to switch between different modes defined below
check_up_converters = False
check_triggers = False
check_down_converters = False
calibration = False

#################################
# Step 0 : Octave configuration #
#################################
opx_ip = "172.0.0.1"
opx_port = 80
octave_ip = "172.0.0.1"
octave_port = 50
con = "con1"
octave = "octave1"
# The elements used to test the ports of the Octave
elements = ["qe1", "qe2", "qe3", "qe4", "qe5"]
IF = 50e6  # The IF frequency
LO = 6e9  # The LO frequency
# The configuration used here
config = {
    "version": 1,
    "controllers": {
        con: {
            "analog_outputs": {
                1: {"offset": 0.0},
                2: {"offset": 0.0},
                3: {"offset": 0.0},
                4: {"offset": 0.0},
                5: {"offset": 0.0},
                6: {"offset": 0.0},
                7: {"offset": 0.0},
                8: {"offset": 0.0},
                9: {"offset": 0.0},
                10: {"offset": 0.0},
            },
            "digital_outputs": {
                1: {},
                2: {},
                3: {},
                4: {},
                5: {},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
            },
        }
    },
    "elements": {
        "qe1": {
            "mixInputs": {
                "I": (con, 1),
                "Q": (con, 2),
                "lo_frequency": LO,
                "mixer": f"octave_{octave}_1",  # a fixed name, do not change.
            },
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
                "cw_wo_trig": "const_wo_trig",
                "readout": "readout_pulse",
            },
            "digitalInputs": {
                "switch": {
                    "port": (con, 1),
                    "delay": 136,
                    "buffer": 0,
                },
            },
            "outputs": {
                "out1": (con, 1),
                "out2": (con, 2),
            },
            "time_of_flight": 24,
            "smearing": 0,
        },
        "qe2": {
            "mixInputs": {
                "I": (con, 3),
                "Q": (con, 4),
                "lo_frequency": LO,
                "mixer": f"octave_{octave}_2",  # a fixed name, do not change.
            },
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
                "cw_wo_trig": "const_wo_trig",
                "readout": "readout_pulse",
            },
            "digitalInputs": {
                "switch": {
                    "port": (con, 2),
                    "delay": 136,
                    "buffer": 0,
                },
            },
            "outputs": {
                "out1": (con, 1),
                "out2": (con, 2),
            },
            "time_of_flight": 24,
            "smearing": 0,
        },
        "qe3": {
            "mixInputs": {
                "I": (con, 5),
                "Q": (con, 6),
                "lo_frequency": LO,
                "mixer": f"octave_{octave}_3",  # a fixed name, do not change.
            },
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
                "cw_wo_trig": "const_wo_trig",
            },
            "digitalInputs": {
                "switch": {
                    "port": (con, 3),
                    "delay": 136,
                    "buffer": 0,
                },
            },
        },
        "qe4": {
            "mixInputs": {
                "I": (con, 7),
                "Q": (con, 8),
                "lo_frequency": LO,
                "mixer": f"octave_{octave}_4",  # a fixed name, do not change.
            },
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
                "cw_wo_trig": "const_wo_trig",
            },
            "digitalInputs": {
                "switch": {
                    "port": (con, 4),
                    "delay": 136,
                    "buffer": 0,
                },
            },
        },
        "qe5": {
            "mixInputs": {
                "I": (con, 9),
                "Q": (con, 10),
                "lo_frequency": LO,
                "mixer": f"octave_{octave}_5",  # a fixed name, do not change.
            },
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
                "cw_wo_trig": "const_wo_trig",
            },
            "digitalInputs": {
                "switch": {
                    "port": (con, 5),
                    "delay": 136,
                    "buffer": 0,
                },
            },
        },
    },
    "pulses": {
        "const": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
            "digital_marker": "ON",
        },
        "const_wo_trig": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": 1000,
            "waveforms": {
                "I": "readout_wf",
                "Q": "zero_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
                "minus_sin": "minus_sine_weights",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "zero_wf": {
            "type": "constant",
            "sample": 0.0,
        },
        "const_wf": {
            "type": "constant",
            "sample": 0.125,
        },
        "readout_wf": {
            "type": "constant",
            "sample": 0.125,
        },
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
        "OFF": {"samples": [(0, 0)]},
    },
    "integration_weights": {
        "cosine_weights": {
            "cosine": [(1.0, 1000)],
            "sine": [(0.0, 1000)],
        },
        "sine_weights": {
            "cosine": [(0.0, 1000)],
            "sine": [(1.0, 1000)],
        },
        "minus_sine_weights": {
            "cosine": [(0.0, 1000)],
            "sine": [(-1.0, 1000)],
        },
    },
    "mixers": {
        f"octave_{octave}_1": [
            {
                "intermediate_frequency": IF,
                "lo_frequency": LO,
                "correction": (1, 0, 0, 1),
            },
        ],
        f"octave_{octave}_2": [
            {
                "intermediate_frequency": IF,
                "lo_frequency": LO,
                "correction": (1, 0, 0, 1),
            },
        ],
        f"octave_{octave}_3": [
            {
                "intermediate_frequency": IF,
                "lo_frequency": LO,
                "correction": (1, 0, 0, 1),
            },
        ],
        f"octave_{octave}_4": [
            {
                "intermediate_frequency": IF,
                "lo_frequency": LO,
                "correction": (1, 0, 0, 1),
            },
        ],
        f"octave_{octave}_5": [
            {
                "intermediate_frequency": IF,
                "lo_frequency": LO,
                "correction": (1, 0, 0, 1),
            },
        ],
    },
}

# Create the octave config object
octave_config = QmOctaveConfig()
# Specify where to store the outcome of the calibration (correction matrix, offsets...)
octave_config.set_calibration_db(os.getcwd())
# Add an Octave called 'octave1' with the specified IP and port
octave_config.add_device_info(octave, octave_ip, octave_port)
# Define the connectivity between our Octave to the OPX(s) ports. The 'default'
# connectivity is as follows:
#    +- OCTAVE (octave1) --------------------------------------------------------+
#    |                                      I1    I2    I3    I4    I5           |
#    |             o    o    o    o    o     o<+   o<+   o<+   o<+   o<+     o   |
#    |   o    o                           QM   |     |     |     |     |         |
#    |             o    o    o    o    o   +>o | +>o | +>o | +>o | +>o |     o   |
#    |                                     |Q1 | |Q2 | |Q3 | |Q4 | |Q5 |         |
#    +-------------------------------------|---|-|---|-|---|-|---|-|---|---------+
#                                          |   | |   | |   | |   | |   |
#                                          |   | |   | |   | |   | |   |
#    +- OPX (con1) ------------------------|---|-|---|-|---|-|---|-|---|---------+
#    |                                     | 1 | | 3 | | 5 | | 7 | | 9 |         |
#    |     o     o     o     o     o    |  | o<+ | o<+ | o<+ | o<+ | o<+ |   o   |
#    |                                     |     |     |     |     |             |
#    |     o     o     o     o     o    |  +>o   +>o   +>o   +>o   +>o   |   o   |
#    |                                       2     4     6     8    10           |
#    +---------------------------------------------------------------------------+
octave_config.set_opx_octave_mapping([(con, octave)])
# The last command is equivalent to
# octave_config.add_opx_octave_port_mapping({
#     ('con1',  1) : ('octave1', 'I1'),
#     ('con1',  2) : ('octave1', 'Q1'),
#     ('con1',  3) : ('octave1', 'I2'),
#     ('con1',  4) : ('octave1', 'Q2'),
#     ('con1',  5) : ('octave1', 'I3'),
#     ('con1',  6) : ('octave1', 'Q3'),
#     ('con1',  7) : ('octave1', 'I4'),
#     ('con1',  8) : ('octave1', 'Q4'),
#     ('con1',  9) : ('octave1', 'I5'),
#     ('con1', 10) : ('octave1', 'Q5'),
# })

# Open the QuantumMachineManager for the OPX and Octave
qmm = QuantumMachinesManager(host=opx_ip, port=opx_port, octave=octave_config)
# Open a quantum machine to calibrate the ports or play signals
qm = qmm.open_qm(config)

# Simple test program that plays a continuous wave through all ports
with program() as hello_octave:
    with infinite_loop_():
        for el in elements:
            play("cw", el)

###########################
# Step 1 : clock settings #
###########################
external_clock = False
if external_clock:
    # Change to the relevant external frequency
    qm.octave.set_clock(octave, clock_mode=ClockMode.External_10MHz)
else:
    qm.octave.set_clock(octave, clock_mode=ClockMode.Internal)
# You can connect clock out from rear panel to a spectrum analyzer  to see the 1GHz signal

#########################################
# Step 2 : set LO, RF gain, and RF mode #
#########################################
external_LO = False
# For external LO only:
if external_LO:
    # OctaveLOSource.LO1,2,3,4,5 to use an external LO source connected to the corresponding ports in the rear panel.
    qm.octave.set_lo_source(elements[1], OctaveLOSource.LO2)
    qm.octave.set_rf_output_gain(elements[1], 0)  # can set the gain from -10dB to 20dB
    qm.octave.set_rf_output_mode(elements[1], RFOutputMode.on)  # set the behaviour of the RF switch to be 'on'.
# For internal LO only:
else:
    for el in elements:
        qm.octave.set_lo_source(el, OctaveLOSource.Internal)  # Use the internal synthetizer to generate the LO.
        qm.octave.set_lo_frequency(el, LO)  # assign the LO inside the octave to element
        qm.octave.set_rf_output_gain(el, 0)  # can set the gain from -10dB to 20dB
        qm.octave.set_rf_output_mode(el, RFOutputMode.on)  # set the behaviour of the RF switch to be 'on'.
        # 'RFOutputMode' can be : on, off, trig_normal or trig_inverse
        # You can check the internal LOs by connecting the ports synth1, 2 and 3 to a spectrum analyzer.

#######################################
# Step 3 : checking the up-converters #
#######################################
if check_up_converters:
    print("-" * 37 + " Checking up-converters")
    job = qm.execute(hello_octave)
    time.sleep(60)  # The program will run for 1 minute
    job.halt()
    # You can connect RF1, RF2, RF3, RF4, RF5 to a spectrum analyzer and check the 3 peaks before calibration:
    # 1. LO-IF, 2. LO, 3. LO+IF

##################################
# Step 4 : checking the triggers #
##################################
if check_triggers:
    print("-" * 37 + " Checking triggers")
    # Connect RF1, RF2, RF3, RF4, RF5 to a spectrum analyzer and check that you get a signal for 4sec then don't get a signal fo 4 sec and so on.
    for el in elements:
        # set the behaviour of the RF switch to be on only when triggered
        qm.octave.set_rf_output_mode(el, RFOutputMode.trig_normal)

    with program() as hello_octave_trigger:
        with infinite_loop_():
            for el in elements:
                play("cw", el, duration=1e9)
                play("cw_wo_trig", el, duration=1e9)
    job = qm.execute(hello_octave_trigger)
    time.sleep(60)  #  The program will run for 1 minute
    job.halt()

#########################################
# Step 5 : checking the down-converters #
#########################################
if check_down_converters:
    print("-" * 37 + " Checking down-converters")
    # Connect RF1 -> RF1In, RF2 -> RF2In
    # Connect IFOUT1 -> AI1 , IFOUT2 -> AI2
    check_down_converter_1 = True
    check_down_converter_2 = False
    u = unit()
    if check_down_converter_1:
        # Reduce the Octave gain to avoid saturating the OPX ADC
        qm.octave.set_rf_output_gain(elements[0], -10)
        # Set the Octave down-conversion port to be RF1In1 for the first element "qe1"
        qm.octave.set_qua_element_octave_rf_in_port(elements[0], octave, 1)
        # Set the source of the down-conversion LO to be internal (only for RF1In1)
        qm.octave.set_downconversion(
            elements[0], lo_source=RFInputLOSource.Internal, if_mode_i=IFMode.direct, if_mode_q=IFMode.direct
        )
        with program() as hello_octave_readout_1:
            raw_ADC_1 = declare_stream(adc_trace=True)
            measure("readout", elements[0], raw_ADC_1)
            with stream_processing():
                raw_ADC_1.input1().save("adc_1")
                raw_ADC_1.input2().save("adc_2")
        # Execute the program
        job = qm.execute(hello_octave_readout_1)
        res = job.result_handles
        res.wait_for_all_values()
        # Plot the results
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Inputs from down conversion 1")
        adc_1 = u.raw2volts(res.get("adc_1").fetch_all())
        ax1.plot(adc_1, label="Input 1")
        ax1.set_title("amp VS time Input 1")
        ax1.set_xlabel("Time [ns]")
        ax1.set_ylabel("Signal amplitude [V]")

        adc_2 = u.raw2volts(res.get("adc_2").fetch_all())
        ax2.plot(adc_2, label="Input 2")
        ax2.set_title("amp VS time Input 2")
        ax2.set_xlabel("Time [ns]")
        ax2.set_ylabel("Signal amplitude [V]")
        plt.tight_layout()

    if check_down_converter_2:
        # Reduce the Octave gain to avoid saturating the OPX ADC
        qm.octave.set_rf_output_gain(elements[1], -10)
        # Set the Octave down-conversion port to be RF1In2 for the second element "qe2"
        qm.octave.set_qua_element_octave_rf_in_port(elements[1], octave, 2)
        # Set the source of the down-conversion LO to be external from the port Dmd2LO in the rear panel
        qm.octave.set_downconversion(
            elements[1], lo_source=RFInputLOSource.Dmd2LO, if_mode_i=IFMode.direct, if_mode_q=IFMode.direct
        )
        with program() as hello_octave_readout_2:
            raw_ADC_2 = declare_stream(adc_trace=True)
            measure("readout", elements[1], raw_ADC_2)
            with stream_processing():
                raw_ADC_2.input1().save("adc_1")
                raw_ADC_2.input2().save("adc_2")
        # Execute the program
        job = qm.execute(hello_octave_readout_2)
        res = job.result_handles
        res.wait_for_all_values()
        # Plot the results
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Inputs from down conversion 2")
        adc_1 = u.raw2volts(res.get("adc_1").fetch_all())
        ax1.plot(adc_1, label="Input 1")
        ax1.set_title("amp VS time Input 1")
        ax1.set_xlabel("Time [ns]")
        ax1.set_ylabel("Signal amplitude [V]")

        adc_2 = u.raw2volts(res.get("adc_2").fetch_all())
        ax2.plot(adc_2, label="Input 2")
        ax2.set_title("amp VS time Input 2")
        ax2.set_xlabel("Time [ns]")
        ax2.set_ylabel("Signal amplitude [V]")
        plt.tight_layout()

#################################
# Step 6 : checking calibration #
#################################
if calibration:
    print("-" * 37 + " Play before calibration")
    # Step 5.1: Connect RF1 and run these lines in order to see the uncalibrated signal first
    job = qm.execute(hello_octave)
    time.sleep(10)  # The program will run for 10 seconds
    job.halt()
    # Step 5.2: Run this in order to calibrate
    for i in range(len(elements)):
        print("-" * 37 + f" Calibrates {elements[i]}")
        qm.octave.calibrate_element(elements[i], [(LO, IF)])  # can provide many pairs of LO & IFs.
        qm = qmm.open_qm(config)
    # Step 5.3: Run these and look at the spectrum analyzer and check if you get 1 peak at LO+IF (i.e. 6.05GHz)
    print("-" * 37 + " Play after calibration")
    job = qm.execute(hello_octave)
    time.sleep(30)  # The program will run for 30 seconds
    job.halt()

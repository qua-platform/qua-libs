import pyvisa as visa
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import numpy as np
import matplotlib.pyplot as plt

######################################
#          HELPER FUNCTIONS          #
######################################
# QDAC2 instrument class
class QDACII():
    def __init__(self, communication_type:str, IP_address:str = None, port:int = 5025, USB_device:int = None, lib:str = '@py'):
        """
        Open the communication to a QDAC2 instrument with python. The communication can be enabled via either Ethernet or USB.

        :param communication_type: Can be either "Ethernet" or "USB".
        :param IP_address: IP address of the instrument - required only for Ethernet communication.
        :param port: port of the instrument, 5025 by default - required only for Ethernet communication.
        :param USB_device: identification number of the device - required only for USB communication.
        :param lib: use '@py' to use pyvisa-py backend (default).
        """
        rm = visa.ResourceManager(lib)  # To use pyvisa-py backend, use argument '@py'
        if communication_type == "Ethernet":
            self._visa = rm.open_resource(f"TCPIP::{IP_address}::{port}::SOCKET")
            self._visa.baud_rate = 921600
            # self._visa.send_end = False
        elif communication_type == "USB":
            self._visa = rm.open_resource(f"ASRL{USB_device}::INSTR")

        self._visa.write_termination = '\n'
        self._visa.read_termination = '\n'
        print(self._visa.query("*IDN?"))
        print(self._visa.query("syst:err:all?"))

    def query(self, cmd):
        return self._visa.query(cmd)

    def write(self, cmd):
        self._visa.write(cmd)

    def write_binary_values(self, cmd, values):
        self._visa.write_binary_values(cmd, values)

    def __exit__(self):
        self.close()

# load list of voltages to the relevant QDAC2 channel
def load_voltage_list(qdac, channel:int, dwell:float, trigger_port:str, output_range:str, output_filter:str, voltage_list:list):
    """
    Configure a QDAC2 channel to play a set of voltages from a given list and step through it according to an external trigger given by an OPX digital marker, using pyvisa commands.

    :param qdac: the QDAC2 object.
    :param channel: the QDAC2 channel that will output the voltage from the voltage list.
    :param dwell: dwell time at each voltage level in seconds - must be smaller than the trigger spacing and larger than 2e-6.
    :param trigger_port: external trigger port to which a digital marker from the OPX is connected - must be in ["ext1", "ext2", "ext3", "ext4"].
    :param output_range: the channel output range that can be either "low" (+/-2V) or "high" (+/-10V).
    :param output_filter: the channel output filter that can be either "dc" (10Hz), "med" (10kHz) or "high" (300kHZ).
    :param voltage_list: list containing the desired voltages to output - the size of the list must not exceed 65536 items.
    :return:
    """
    # Load the list of voltages
    qdac.write_binary_values(f"sour{channel}:dc:list:volt ", voltage_list)
    # Ensure that the output voltage will start from the beginning of the list.
    qdac.write(f"sour{channel}:dc:init:cont off")
    # Set the minimum time spent on each voltage level. Must be between 2µs and the time between two trigger events.
    qdac.write(f"sour{channel}:dc:list:dwell {dwell}")
    # Step through the voltage list on the event of a trigger
    qdac.write(f"sour{channel}:dc:list:tmode stepped")
    # Set the external trigger port. Must be in ["ext1", "ext2", "ext3", "ext4"]
    qdac.write(f"sour{channel}:dc:trig:sour {trigger_port}")
    # Listen continuously to trigger
    qdac.write(f"sour{channel}:dc:init:cont on")
    # Make sure that the correct DC mode (LIST) is set, as opposed to FIXed.
    qdac.write(f"sour{channel}:dc:mode LIST")
    # Set the channel output range
    qdac.write(f"sour{channel}:rang {output_range}")
    # Set the channel output filter
    qdac.write(f"sour{channel}:filt {output_filter}")
    print(f"Set-up QDAC2 channel {channel} to step voltages from a list of {len(voltage_list)} items on trigger events from the {trigger_port} port with a {qdac.query(f'sour{channel}:dc:list:dwell?')} s dwell time.")

######################################
#        SET UP THE EXPERIMENT       #
######################################

# Create the qdac instrument
qdac = QDACII("Ethernet", IP_address="172.16.33.100", port=5025)  # Using Ethernet protocol
qdac.write("*rst")
# qdac = QDACII("USB", USB_device=4)  # Using USB protocol

# Open a Quantum Machine Manager
qmm = QuantumMachinesManager(host="172.16.33.100", cluster_name="Cluster_81")

# Define a QUA program that will trigger the QDAC and measure the output voltage with some averaging
n_avg = 100  # Number of averaging loops
# Voltage values in Volt
voltage_values = list(np.linspace(-0.4, 0.4, 101))

with program() as qdac_1d_sweep:
    i = declare(int)
    n = declare(int)
    data = declare(fixed)
    data_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(i, 0, i < len(voltage_values), i + 1):
            # Wait before sending the trigger - can be replaced by any sequence
            wait(10_000 // 4, "qdac_trigger1", "readout_element")
            # Trigger the QDAC channel
            play("trig", "qdac_trigger1")
            # Measure with the OPX
            measure("readout", "readout_element", None, integration.full("const_weight", data, "out1"))
            # Send the result to the stream processing
            save(data, data_st)

    with stream_processing():
        # Average all the data and save only the last value into "data".
        data_st.buffer(len(voltage_values)).average().save("data")

# Set up the qdac and load the voltage list
load_voltage_list(qdac, channel=1, dwell=2e-6, trigger_port="ext1", output_range="low", output_filter="med", voltage_list=voltage_values)

######################################
#         RUN THE EXPERIMENT         #
######################################
qm = qmm.open_qm(config)
job = qm.execute(qdac_1d_sweep)
job.result_handles.wait_for_all_values()
data = job.result_handles.get("data").fetch_all()
plt.figure()
plt.plot(voltage_values, data, ".")

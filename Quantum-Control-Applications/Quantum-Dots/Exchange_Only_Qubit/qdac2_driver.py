from time import sleep
import pyvisa as visa
from typing import Union
from numpy.typing import NDArray


# QDAC2 instrument class
class QDACII:
    def __init__(
        self,
        communication_type: str,
        IP_address: str = None,
        port: int = 5025,
        USB_device: int = None,
        lib: str = "@py",
    ):
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

        self._visa.write_termination = "\n"
        self._visa.read_termination = "\n"
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
def load_voltage_list(
    qdac,
    channel: int,
    dwell: float,
    slew_rate: float,
    trigger_port: str,
    output_range: str,
    output_filter: str,
    voltage_list: Union[NDArray, list],
):
    """
    Configure a QDAC2 channel to play a set of voltages from a given list and step through it according to an external trigger given by an OPX digital marker, using pyvisa commands.

    :param qdac: the QDAC2 object.
    :param channel: the QDAC2 channel that will output the voltage from the voltage list.
    :param dwell: dwell time at each voltage level in seconds - must be smaller than the trigger spacing and larger than 2e-6.
    :param slew_rate: the rate at which the voltage can change in Volt per seconds to avoid transients from abruptly stepping the voltage. Must be within [0.01; 2e7].
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
    # Set the maximum voltage slope in V/s
    qdac.write(f"sour{channel}:dc:volt:slew {slew_rate}")
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
    sleep(1)
    print(
        f"Set-up QDAC2 channel {channel} to step voltages from a list of {len(voltage_list)} items on trigger events from the {trigger_port} port with a {qdac.query(f'sour{channel}:dc:list:dwell?')} s dwell time."
    )
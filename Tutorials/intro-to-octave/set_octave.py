"""
set_octave.py: script for initializing the octave
"""
import os
from qm.octave import QmOctaveConfig


class OctaveUnit:
    """Class for keeping track of OctavesSettings in inventory."""

    def __init__(
        self,
        name: str,
        ip: str,
        port: int = 50,
        con: str = "con1",
    ):
        """Class for keeping track of OctavesSettings in inventory.

        :param name: Name of the Octave.
        :param ip: IP address of the router to which the Octave is connected.
        :param port: Port of the Octave.
        :param con: Controller to which the Octave is connected. Only used when port mapping set to default.
        """
        self.name = name
        self.ip = ip
        self.port = port
        self.con = con


def octave_declaration(octaves: list = ()):
    """
    Initiate octave_config class, set the calibration file and add octaves info.

    :param octaves: objects that holds the information about octave's name, the controller that is connected to this octave, octave's ip and octave's port.
    """
    octave_config = QmOctaveConfig()
    octave_config.set_calibration_db(os.getcwd())
    for i in range(len(octaves)):
        if octaves[i].name is None:
            raise TypeError(f"Please insert the octave name for the {i}'s octave")
        if octaves[i].con is None:
            raise TypeError(f"Please insert the controller that is connected to the {i}'s octave")
        if octaves[i].ip is None:
            raise TypeError(f"Please insert the octave ip for the {i}'s octave")
        if octaves[i].port is None:
            raise TypeError(f"Please insert the octave port for the {i}'s octave")
        octave_config.add_device_info(octaves[i].name, octaves[i].ip, octaves[i].port)

    return octave_config

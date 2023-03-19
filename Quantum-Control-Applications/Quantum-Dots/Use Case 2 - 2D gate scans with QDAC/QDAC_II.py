import pyvisa as visa

VISA_ADDR="TCPIP::172.16.2.107::5025::SOCKET"
class QDACII():
    def __init__(self, visa_addr=VISA_ADDR, lib='@py'):
        rm = visa.ResourceManager(lib)  # To use pyvisa-py backend, use argument '@py'
        self._visa = rm.open_resource(visa_addr)
        self._visa.write_termination = '\n'
        self._visa.read_termination = '\n'
        # Set baudrate and stuff for serial communication only
        if (visa_addr.find("ASRL") != -1):
            self._visa.baud_rate = 921600
            self._visa.send_end = False

    def query(self, cmd):
        return self._visa.query(cmd)

    def write(self, cmd):
        self._visa.write(cmd)

    def write_binary_values(self, cmd, values):
        self._visa.write_binary_values(cmd, values)

    def __exit__(self):
        self.close()

    def setup_qdac_channels_for_triggered_list(self, channels, trigger_sources, dwell_s_vals):

        for channel, trigger, dwell_s in zip(channels, trigger_sources, dwell_s_vals):
            # Setup LIST connect to external trigger
            # ! Remember to set FIXed mode if you later want to set a voltage directly
            self.write(f"sour{channel}:dc:list:dwell {dwell_s}")
            self.write(f"sour{channel}:dc:list:tmode stepped")  # point by point trigger mode
            self.write(f"sour{channel}:dc:trig:sour {trigger}")
            self.write(f"sour{channel}:dc:init:cont on")
            # Always make sure that you are in the correct DC mode (LIST) in case you have switched to FIXed
            self.write(f"sour{channel}:dc:mode LIST")


class FakeQDAC():
    def __init__(self, visa_addr=VISA_ADDR, lib='@py'):

        print('initialised fake qdac')

    def query(self, cmd):
        print(f'queried {cmd}')

    def write(self, cmd):
        print(f'wrote {cmd}')

    def write_binary_values(self, cmd, values):
        self.write(cmd + values)

    def __exit__(self):
        self.write('closed')

    def setup_qdac_channels_for_triggered_list(self, channels, trigger_sources, dwell_s_vals):

        for channel, trigger, dwell_s in zip(channels, trigger_sources, dwell_s_vals):
            # Setup LIST connect to external trigger
            # ! Remember to set FIXed mode if you later want to set a voltage directly
            self.write(f"sour{channel}:dc:list:dwell {dwell_s}")
            self.write(f"sour{channel}:dc:list:tmode stepped")  # point by point trigger mode
            self.write(f"sour{channel}:dc:trig:sour {trigger}")
            self.write(f"sour{channel}:dc:init:cont on")
            # Always make sure that you are in the correct DC mode (LIST) in case you have switched to FIXed
            self.write(f"sour{channel}:dc:mode LIST")


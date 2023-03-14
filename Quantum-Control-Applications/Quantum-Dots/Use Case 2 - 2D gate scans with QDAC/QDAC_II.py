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

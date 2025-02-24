import pyvisa as visa
from qcodes_contrib_drivers.drivers.QDevil import QDAC2

####################################################################################################################################################
# This script assumes that the QDAC's IP is not known. Follow the steps to find it through the USB connection and switch to Ethernet communication #
####################################################################################################################################################

#####################
# USB communication #
#####################
# 1. Connect the QDAC USB port to the PC and check the USB connection.
# 2. Look for the relevant connection by listing the visa resources.
rm = visa.ResourceManager("")
rm.list_resources()

# 3. Insert serial port found with rm.list_resources() and pen communication through USB port
qdac_serial_addr = (
    "ASRL/dev/cu.usbserial-14210::INSTR"  # This is typical for a mac, for windows it will look like 'ASRL5::INSTR'
)
qdac = QDAC2.QDac2("QDAC", visalib="@py", address=qdac_serial_addr)
# 4. check the communication with the QDAC
print(qdac.IDN())  # query the QDAC's identity
print(qdac.errors())  # read and clear all errors from the QDAC's error queue

# 5. To connect to ethernet afterwards, check the QDAC's IP
print(qdac.ask("syst:comm:lan:ipad?"))  # print the IP of the QDAC
# Once you have the QDAC's IP, take the IP and set an Ethernet communication. It is faster than USB.


# 6. Close to qdac instance so you can create it again.
qdac.close()

##########################
# Ethernet communication #
##########################
# 7. Insert IP
qdac_ipaddr = "169.254.55.17"
# 8. Open communication through Ethernet port
qdac = QDAC2.QDac2("QDAC", visalib="@py", address=f"TCPIP::{qdac_ipaddr}::5025::SOCKET")
# 9. Check the communication with the QDAC
print(qdac.IDN())  # query the QDAC's identity
print(qdac.errors())  # read and clear all errors from the QDAC's error queue

# 10. Close to qdac instance so you can create it again.
qdac.close()

from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
import pyvisa as visa
from qcodes_contrib_drivers.drivers.QDevil import QDAC2


##########################
# Ethernet communication #
##########################
# insert IP
qdac_ipaddr = "169.254.55.17"
# open communication
qdac = QDAC2.QDac2("QDAC", visalib="@py", address=f"TCPIP::{qdac_ipaddr}::5025::SOCKET")
# check the communication with the QDAC
print(qdac.IDN())  # query the QDAC's identity
print(qdac.errors())  # read and clear all errors from the QDAC's error queue

##################
# Leakage matrix #
##################

# For testing, connect resistors: 5M6 over ch 5, 33M between ch 3 & 4, and 5G over ch 1
arrangement = qdac.arrange(contacts={"G1": 1, "G2": 2, "G3": 3, "G4": 4, "O5": 5})
# Set initial voltage values
arrangement.set_virtual_voltages({"G1": 0.0, "G2": 0.0, "G3": 0.0, "G4": 0.0, "O5": 0.0})
sleep(0.5)

# Measure leakage by raising the voltage by 5 mV on each channel in turn.
modulation_mV = 5
powerline_cycles = 20
leakage_matrix_Ohm = arrangement.leakage(modulation_V=modulation_mV / 1000, nplc=powerline_cycles)

leakage_megaohm = leakage_matrix_Ohm / 1e6

# plot
# Show the leakage matrix but cap it off at 100 MΩ
fig, ax = plt.subplots()
plt.title(f"Gate Leakage ({modulation_mV}mV)")
img = ax.imshow(leakage_megaohm, interpolation="none", vmin=0, vmax=100)
ticks = np.arange(len(arrangement.contact_names))
minorticks = np.arange(-0.5, len(ticks), 1)
ax.set_xticks(ticks, labels=arrangement.contact_names)
ax.set_yticks(ticks, labels=arrangement.contact_names)
ax.set_xticks(minorticks, minor=True)
ax.set_yticks(minorticks, minor=True)
ax.grid(which="minor", color="grey", linewidth=1.5)
plt.gca().invert_yaxis()
colorbar = fig.colorbar(img)
colorbar.set_label("Resistance (MΩ)")

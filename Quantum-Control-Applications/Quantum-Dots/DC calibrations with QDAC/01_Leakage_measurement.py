from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
import pyvisa as visa
from qcodes_contrib_drivers.drivers.QDevil import QDAC2


##########################
# Ethernet communication #
##########################
# Insert IP
qdac_ipaddr = "127.0.0.1"  # Write the QDAC IP address
# Open communication
qdac = QDAC2.QDac2("QDAC", visalib="@py", address=f"TCPIP::{qdac_ipaddr}::5025::SOCKET")
# Check the communication with the QDAC
print(qdac.IDN())  # query the QDAC's identity
print(qdac.errors())  # read and clear all errors from the QDAC's error queue

##################
# Leakage matrix #
##################

# Connect resistors P1 to channel 1, P2 to channel 2, B1 to channel 3, B2 to channel 4 and O1 to channel 5
arrangement = qdac.arrange(contacts={"P1": 1, "P2": 2, "B1": 3, "B2": 4, "O1": 5})
# Set initial voltage values
arrangement.set_virtual_voltages({"P1": 0.0, "P2": 0.0, "B1": 0.0, "B2": 0.0, "O1": 0.0})
sleep(0.5)

# Measure leakage by raising the voltage by 5 mV on each channel in turn.
modulation_mV = 5
powerline_cycles = 20
leakage_matrix_Ohm = arrangement.leakage(modulation_V=modulation_mV / 1000, nplc=powerline_cycles)

leakage_megaohm = leakage_matrix_Ohm / 1e6

# Plot
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

# Free all internal triggers, 12 internal triggers are available
qdac.free_all_triggers()
# Close to qdac instance so you can create it again.
qdac.close()

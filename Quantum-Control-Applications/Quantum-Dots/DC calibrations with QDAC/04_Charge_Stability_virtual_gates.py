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

####################
# 2D voltage sweep #
####################

inner_steps = 21  # define the voltage steps
inner_V = np.linspace(-0.3, 0.4, inner_steps)
outer_steps = 21  # define the voltage steps
outer_V = np.linspace(-0.2, 0.5, outer_steps)
inner_step_time = 20e-3

# Define the plunger gates
arrangement = qdac.arrange(
    # QDAC channels 2 and 3 connected to p1 and p2 respectively
    contacts={"p1": 2, "p2": 3},
    # Internal trigger for measuring current
    internal_triggers={"inner"},
)

# Add virtual gates corrections
arrangement.initiate_correction("p1", [0.9, 0.1])
arrangement.initiate_correction("p2", [-0.2, 0.97])


sweep = arrangement.virtual_sweep2d(
    inner_contact="p1",
    inner_voltages=inner_V,
    outer_contact="p2",
    outer_voltages=outer_V,
    inner_step_time_s=inner_step_time,
    inner_step_trigger="inner",
)

# Define the sensor parameters
sensor_channel = 5  # define sensing channel
sensor_integration_time = (
    15e-3  # define time [s] to integrate current over #NOTE: sensor integration time should be <= step_time
)
sensing_range = "low"  # low (max 150 nA, noise level ~10 pA) or high (max 10 mA, noise level ~1 uA) current range

# Set up the current sensor
sensor = qdac.channel(sensor_channel)  # choose the sensor channel
sensor.measurement_aperture_s(sensor_integration_time)  # choose the sensing time
sensor.measurement_range(sensing_range)  # choose the sensing range
sensor.clear_measurements()  # clear any remaining buffer of measurements
measurement = sensor.measurement()  # create a measurement instance for the sensor
measurement.start_on(arrangement.get_trigger_by_name("inner"))  # set the trigger that will start a measurement

# Set to starting voltage
arrangement.set_virtual_voltage("p1", inner_V[0])
arrangement.set_virtual_voltage("p2", outer_V[0])
sleep(0.5)

# Start sweep
sweep.start()
sleep(inner_steps * outer_steps * inner_step_time + 0.5)

# Stop current flow
arrangement.set_virtual_voltage("p1", 0)
arrangement.set_virtual_voltage("p2", 0)


raw = measurement.available_A()
available = list(map(lambda x: float(x), raw[-(outer_steps * inner_steps) :]))

# Plot
currents = np.reshape(available, (-1, inner_steps)) * 1000
fig, ax = plt.subplots()
plt.title("diodes (Ge) back-to-back")
extent = [inner_V[0], inner_V[-1], outer_V[0], outer_V[-1]]
img = ax.imshow(currents, cmap="plasma", interpolation="nearest", extent=extent)
ax.set_xlabel("Volt")
ax.set_ylabel("Volt")
colorbar = fig.colorbar(img)
colorbar.set_label("mA")

# Free all internal triggers, 12 internal triggers are available
qdac.free_all_triggers()
# Close to qdac instance so you can create it again.
qdac.close()

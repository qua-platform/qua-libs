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
# 1D voltage sweep #
####################

# Define the sweep parameters
arrangement = qdac.arrange(
    # QDAC channels 1, 2 and 10 are connected to p1, p2 and b1 respectively
    contacts={"p1": 1, "p2": 2, "b1": 10},
    # Internal trigger for measuring current
    internal_triggers={"inner"},  # name the channels
)
n_steps = 21  # define the number of voltage steps
V_list = np.linspace(-0.3, 0.4, n_steps)  # define the voltage sweep
step_time = 20e-3  # time [s] per voltage step

# Create the sweep
sweep = arrangement.virtual_sweep(
    contact="p1",  # choose the contact channel
    voltages=V_list,  # choose the voltage list
    step_time_s=step_time,  # choose the time per voltage
    step_trigger="inner",
)  # choose the trigger for the measurement

# Define the sensor parameters
sensor_channel = 5  # define sensing channel
sensor_integration_time = (
    15e-3  # define time [s] to integrate current over #NOTE: sensor_integration_time should be <= step_time
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
arrangement.set_virtual_voltage("p1", V_list[0])
sleep(0.5)

# Start sweep
sweep.start()  # this starts the voltage sweep and the attached triggers will trigger the current sensing
sleep(n_steps * step_time + 0.5)

# Return voltages to their dc value
arrangement.set_virtual_voltage("p1", 0)

raw = measurement.available_A()  # fetch the current measurements from the buffer
available = list(map(lambda x: float(x), raw[-n_steps:]))

# Plot
currents = np.array(available) * 1000  # current in mA
fig, ax = plt.subplots()
plt.title("IV measurement")
ax.plot(V_list, currents)
ax.set_xlabel("Voltage [V]")
ax.set_ylabel("Current [mA]")

# Free all internal triggers, 12 internal triggers are available
qdac.free_all_triggers()
# Close to qdac instance so you can create it again.
qdac.close()

from qcodes_contrib_drivers.drivers.QDevil import QDAC2
import os
import qcodes as qc
from time import time
from qcodes import initialise_or_create_database_at, load_or_create_experiment
from qcodes.utils.dataset.doNd import do2d, do1d, do0d
from qcodes import Parameter
from qm.qua import *
from qualang_tools.external_frameworks.qcodes.opx_driver import OPX
from configuration import *

#####################################
#           Qcodes set-up           #
#####################################
db_name = "QM_demo.db"  # Database name
sample_name = "demo"  # Sample name
exp_name = "OPX_QDAC2_integration"  # Experiment name

# Initialize qcodes database
db_file_path = os.path.join(os.getcwd(), db_name)
qc.config.core.db_location = db_file_path
initialise_or_create_database_at(db_file_path)
# Initialize qcodes experiment
experiment = load_or_create_experiment(experiment_name=exp_name, sample_name=sample_name)
# Initialize the qcodes station to which instruments will be added
station = qc.Station()
# Create the OPX instrument class
opx_instrument = OPX(config, name="OPX_demo", host="172.16.33.100", cluster_name="Cluster_81")
# Add the OPX instrument to the qcodes station
station.add_component(opx_instrument)

# Create the QDAC2 instrument class
qdac2 = QDAC2.QDac2("QDAC", visalib="@py", address=f"TCPIP::172.16.33.100::5025::SOCKET")
# Add the QDAC2 instrument to the qcodes station
station.add_component(qdac2)

# Parameters used in all scans below
n_avg = 100  # Number of averaging loops
# Voltage values in Volt
voltage_values1 = list(np.linspace(-0.4, 0.4, 101))
voltage_values2 = list(np.linspace(-2.5, 2.5, 101))
qdac_offset = 3.752

######################################
#       Fast 1d sweep with do0d      #
######################################
"""
Define a QUA program that will trigger the QDAC and measure the output voltage with some averaging
"""


### OPX section
def qdac_1d_sweep_fast(simulate=False):
    with program() as prog:
        i = declare(int)
        n = declare(int)
        data = declare(fixed)
        data_st = declare_stream()

        with infinite_loop_():
            if not simulate:
                pause()
            with for_(n, 0, n < n_avg, n + 1):
                with for_(i, 0, i < len(voltage_values1), i + 1):
                    # Wait before sending the trigger - can be replaced by any sequence
                    wait(10_000 // 4, "qdac_trigger1", "readout_element")
                    # Trigger the QDAC channel
                    play("trig", "qdac_trigger1")
                    # Measure with the OPX
                    measure("readout", "readout_element", None, integration.full("const", data, "out1"))
                    # Send the result to the stream processing
                    save(data, data_st)

        with stream_processing():
            # Average all the data and save the values into "data".
            data_st.buffer(len(voltage_values1)).buffer(n_avg).map(FUNCTIONS.average()).save_all("data")
    return prog


# Pass the readout length (in ns) to the class to convert the demodulated/integrated data into Volts
opx_instrument.readout_pulse_length(readout_len)
# Axis1 is the most inner loop
opx_instrument.set_sweep_parameters("axis1", voltage_values1, "V", "Vg1")
# Add the custom sequence to the OPX
opx_instrument.qua_program = qdac_1d_sweep_fast(simulate=False)

### QDAC2 section
qdac2.reset()
Vg1 = qdac2.channel(1)  # Define the QDAC2 channel
# Set the current range ("high" or "low") and filter ("dc": 10Hz ,  "med": 10khz,  "high": 300kHz)
Vg1.output_mode(range="low", filter="med")
# Define the voltage list, stepping mode and dwell time
Vg1_list = Vg1.dc_list(voltages=voltage_values1, stepped=True, dwell_s=5e-6)
# Set the trigger mode to external with input port "ext1"
Vg1_list.start_on_external(trigger=1)

### Run the experiment
experiment1 = load_or_create_experiment("Fast_1D_sweep_do0d", sample_name)

start_time = time()
do0d(
    opx_instrument.run_exp,
    opx_instrument.resume,
    opx_instrument.get_measurement_parameter(),
    opx_instrument.halt,
    do_plot=True,
    exp=experiment1,
)
print(f"Elapsed time: {time() - start_time:.2f} s")

######################################
#       Slow 1d sweep with do1d      #
######################################
"""
Define a QUA program that will measure the output voltage with single point averaging while qcodes will step the QDAC2 voltage
"""


### OPX section
def qdac_1d_sweep_slow(simulate=False):
    with program() as prog:
        i = declare(int)
        n = declare(int)
        data = declare(fixed)
        data_st = declare_stream()

        with infinite_loop_():
            if not simulate:
                pause()
            with for_(n, 0, n < n_avg, n + 1):
                # Wait before measuring - can be replaced by any sequence
                wait(1_000_000 // 4, "readout_element")
                # Measure with the OPX
                measure("readout", "readout_element", None, integration.full("const", data, "out1"))
                # Send the result to the stream processing
                save(data, data_st)

        with stream_processing():
            # Average all the data and save the values into "data".
            data_st.buffer(n_avg).map(FUNCTIONS.average()).save_all("data")
    return prog


# Pass the readout length (in ns) to the class to convert the demodulated/integrated data into Volts
opx_instrument.readout_pulse_length(readout_len)
# Add the custom sequence to the OPX
opx_instrument.qua_program = qdac_1d_sweep_slow(simulate=False)

### QDAC2 section
qdac2.reset()  # Reset the qdac parameters
ch1 = qdac2.channel(1)  # Define the QDAC2 channel
# Set the current range ("high" or "low") and filter ("dc": 10Hz ,  "med": 10khz,  "high": 300kHz)
ch1.output_mode(range="low", filter="med")
Vg1 = ch1.dc_constant_V  # Define the voltage parameter for this channel

### Run the experiment
experiment2 = load_or_create_experiment("Slow_1D_sweep_do1d", sample_name)

start_time = time()
do1d(
    Vg1,
    voltage_values1[0],
    voltage_values1[-1],
    len(voltage_values1),
    0.001,
    opx_instrument.resume,
    opx_instrument.get_measurement_parameter(),
    enter_actions=[opx_instrument.run_exp],
    exit_actions=[opx_instrument.halt],
    show_progress=True,
    do_plot=True,
    exp=experiment2,
)
print(f"Elapsed time: {time() - start_time:.2f} s")


############################################################################
#       2D map combining the QDAC (slow axis) and the OPX (fast axis)      #
############################################################################
"""
Define a QUA program that will step the voltage along the fast axis and measure with single point averaging (inner loop) 
while qcodes will step the QDAC2 voltage along the slow axis. 
It is equivalent to raster scan for acquiring a charge stability map.
"""

### QDAC2 section
qdac2.reset()  # Reset the qdac parameters
# Channel 1
qdac_offset = 3.752  # V
ch1 = qdac2.channel(1)
ch1.output_mode(range="high", filter="dc")
Vg1 = ch1.dc_constant_V(qdac_offset)  # Set an offset that will be combined with the OPX fast scan (3.752 +/- 0.3V)
# Channel 2
ch2 = qdac2.channel(2)
ch2.output_mode(range="low", filter="med")
Vg2 = ch2.dc_constant_V  # Define the voltage parameter for this channel


### OPX section
def qdac_opx_combined(simulate=False):
    with program() as prog:
        i = declare(int)
        n = declare(int)
        data = declare(fixed)
        data_st = declare_stream()

        with infinite_loop_():
            if not simulate:
                pause()
            with for_(i, 0, i < len(voltage_values1), i + 1):
                # Step the voltage along the fast axis
                play("step", "gate")
                # Wait before measuring
                wait(10_000 // 4, "readout_element")
                with for_(n, 0, n < n_avg, n + 1):
                    # Measure with the OPX
                    measure("readout", "readout_element", None, integration.full("const", data, "out1"))
                    # Send the result to the stream processing
                    save(data, data_st)
            # Bring the voltage back to zero
            ramp_to_zero("gate")

        with stream_processing():
            # Average all the data and save the values into "data".
            data_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(len(voltage_values1)).save_all("data")
    return prog


# Pass the readout length (in ns) to the class to convert the demodulated/integrated data into Volts
opx_instrument.readout_pulse_length(readout_len)
# Add the custom sequence to the OPX
opx_instrument.qua_program = qdac_opx_combined(simulate=False)
# Axis1 is the most inner non-averaging loop
opx_instrument.set_sweep_parameters("axis1", voltage_values1 + qdac_offset, "V", "Vg1")

### Run the experiment
experiment2 = load_or_create_experiment("Combined_2D_sweep_do1d", sample_name)

start_time = time()
do1d(
    Vg2,
    voltage_values2[0],
    voltage_values2[-1],
    len(voltage_values2),
    0.001,
    opx_instrument.resume,
    opx_instrument.get_measurement_parameter(),
    enter_actions=[opx_instrument.run_exp],
    exit_actions=[opx_instrument.halt],
    show_progress=True,
    do_plot=True,
    exp=experiment2,
)
print(f"Elapsed time: {time() - start_time:.2f} s")

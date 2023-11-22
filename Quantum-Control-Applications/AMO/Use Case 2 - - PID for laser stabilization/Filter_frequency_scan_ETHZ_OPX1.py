from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration_cavity_locking_ETHZ_OPX1 import *
from qualang_tools.addons.variables import assign_variables_to_element
from qualang_tools.results import fetching_tool
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


###################
# The QUA program #
###################
angle = 0.0  # Phase angle between the sideband and demodulation in units of 2pi
# Scanned offset values in Volt
offsets = np.linspace(-0.3, 0.3, 101) - setpoint_filter_cavity_1
step = np.mean(np.diff(offsets))
n_avg = 100

with program() as prog:
    n = declare(int)
    offset = declare(fixed)  # 3.28
    # Results variables
    I = declare(fixed)
    Q = declare(fixed)
    single_shot_DC = declare(fixed)
    single_shot_AC = declare(fixed)
    # Streams
    single_shot_DC_st = declare_stream()
    single_shot_AC_st = declare_stream()

    # Ensure that the results variables are assigned to the measurement elements
    assign_variables_to_element("detector_DC", single_shot_DC)
    assign_variables_to_element("detector_AC", I, Q, single_shot_AC)

    play("offset" * amp(offsets[0] * 4), "filter_cavity_1")
    with for_(*from_array(offset, offsets)):
        with for_(n, 0, n<n_avg, n+1):
            # Ensure that the two digital oscillators will start with the same phase
            reset_phase("phase_modulator")
            reset_phase("detector_AC")
            # Adjust the phase delay between the two
            frame_rotation_2pi(angle, "detector_AC")
            # Sync all the elements
            align()
            # Play the PDH sideband
            play("cw", "phase_modulator")
            # Measure and integrate the signal received by the detector --> DC measurement
            measure("readout", "detector_DC", None, integration.full("constant", single_shot_DC, "out1"))
            # Measure and demodulate the signal received by the detector --> AC measurement sqrt(I**2 + Q**2)
            measure("readout", "detector_AC", None, demod.full("constant", I, "out1"), demod.full("constant", Q, "out1"))
            # assign(single_shot_AC, Math.sqrt(I*I + Q*Q))
            assign(single_shot_AC, I)

            # Save the desired variables
            save(single_shot_DC, single_shot_DC_st)
            save(single_shot_AC, single_shot_AC_st)

            # Wait between each iteration
            wait(100)
        # Update the filter cavity offset
        with if_(offset < offsets[-1]):
            play("offset" * amp(step * 4), "filter_cavity_1")

    with stream_processing():
        single_shot_DC_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(len(offsets)).save("single_shot_DC")
        single_shot_AC_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(len(offsets)).save("single_shot_AC")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=9510)
# qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)

###########################
# Run or Simulate Program #
###########################

simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, prog, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()
else:
    import time
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(prog)
    results = fetching_tool(job, ["single_shot_DC", "single_shot_AC"])
    single_shot_DC, single_shot_AC = results.fetch_all()
    single_shot_DC, single_shot_AC = u.demod2volts(single_shot_DC, readout_len), u.demod2volts(single_shot_AC, readout_len)*np.sqrt(2)
    fig = plt.figure()
    plt.subplot(211)
    plt.plot(offsets, single_shot_DC, "-")
    plt.xlabel("Offset [V]")
    plt.ylabel("Reflection signal [V]")
    plt.subplot(212)
    plt.plot(offsets, single_shot_AC, "-")
    plt.xlabel("Offset [V]")
    plt.ylabel("Error signal [V]")
    plt.tight_layout()

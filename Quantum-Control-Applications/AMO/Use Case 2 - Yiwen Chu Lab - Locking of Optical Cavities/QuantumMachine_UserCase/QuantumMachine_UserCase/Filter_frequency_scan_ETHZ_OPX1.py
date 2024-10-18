from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration_cavity_locking_ETHZ_OPX1 import *
from qualang_tools.addons.variables import assign_variables_to_element
from qualang_tools.results import fetching_tool
from qualang_tools.loops import from_array, qua_linspace
import matplotlib.pyplot as plt
import warnings
from scipy.signal import find_peaks
import time

warnings.filterwarnings("ignore")


###################
# The QUA program #
###################
def measure_macro(single_shot_DC_st,single_shot_AC_st,I,Q):
    n = declare(int)
    single_shot_DC = declare(fixed)
    single_shot_AC = declare(fixed)
    with for_(n, 0, n<n_avg, n+1):
        # Ensure that the two digital oscillators will start with the same phase
        reset_phase("phase_modulator")
        reset_phase("detector_AC")
        # Adjust the phase delay between the two
        # frame_rotation_2pi(angle, "detector_AC")
        frame_rotation_2pi(angle, "phase_modulator")
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
        reset_frame("phase_modulator") #reset the phase to undo the angle rotation above

        # Save the desired variables
        save(single_shot_DC, single_shot_DC_st)
        save(single_shot_AC, single_shot_AC_st)
        # Wait between each iteration
        wait(100)

def update_offset():
    nst=declare(int)
    with if_(use_small_steps):
        with for_(nst, 0, nst<N_ss, nst+1):
            play("offset" * amp(small_step * 4), "filter_cavity_1")
            wait(small_step_time)
    with else_():
        play("offset" * amp(step * 4), "filter_cavity_1") #single step
        wait(step_time) #delay before measuring to allow the cavity to settle

def sweep_offset_macro(offsets,single_shot_DC_st,single_shot_AC_st,I,Q):
    offset = declare(fixed)
    with for_(*from_array(offset, offsets)):
        measure_macro(single_shot_DC_st,single_shot_AC_st,I,Q)
        update_offset()


def initialize_sweep_macro(offset_0):
    # Move to start of the scan
    play("offset" * amp(offset_0 * 4), "filter_cavity_1")
    wait(step_time) #delay before measuring to allow the cavity to settle

def SlowLock(offsets, n_avg):
    with program() as prog:
        # n = declare(int)
        # nst= declare(int)
        # offset = declare(fixed)  # 3.28
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

        initialize_sweep_macro(offsets[0])
        sweep_offset_macro(offsets,single_shot_DC_st,single_shot_AC_st,I,Q)

        with stream_processing():
            single_shot_DC_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(len(offsets)).save("single_shot_DC")
            single_shot_AC_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(len(offsets)).save("single_shot_AC")

    return prog

if __name__ == '__main__':
    #####################################
    #  Open Communication with the QOP  #
    #####################################

    qmm = QuantumMachinesManager(host=qop_ip, port=9510)
    # qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
    qm = qmm.open_qm(config)

    ################
    #  Parameters  #
    ################

    angle = 0.25*config_angle # Phase angle between the sideband and demodulation in units of 2pi

    # Scanned offset values in Volt
    offsets = np.linspace(-0.5, 0.5, 5001) # This way, we sweep around setpoint

    step = np.mean(np.diff(offsets))
    n_avg = 1
    step_time = 20000 #time that a single step takes in clock cycles (4 ns)
    use_small_steps = True #use small steps to avoid ringing
    N_ss= 60 #number of small steps (target)

    show_av_AC = False #show the average AC signal
    exclude_pts = [0,1] #indices of points to exclude in the plot

    #set small step size
    small_step = step/N_ss #small step size
    if np.abs(small_step) < 2**-16: # 2**-16 is the smallest step size possible
        if step<0:
            N_ss = int(np.ceil(np.abs(step)/2**-16))
        elif step>0:
            N_ss = int(np.floor(np.abs(step)/2**-16))
        small_step = step/N_ss
        print(f"small step size too small, using {N_ss} small steps instead")
    print(small_step)
    small_step_time = int(step_time/N_ss) #time that a small step takes in clock cycles (4 ns)

    # Open a quantum machine to execute the QUA program

    ###############
    # Run Program #
    ###############

    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    prog=SlowLock(offsets, n_avg)
    job = qm.execute(prog)

    #################
    # Fetch results #
    #################

    results = fetching_tool(job, ["single_shot_DC", "single_shot_AC"])
    single_shot_DC, single_shot_AC  = results.fetch_all()
    #filter AC signal
    single_shot_AC_filt = single_shot_AC[~np.isin(np.arange(len(single_shot_AC)), exclude_pts)]
    single_shot_DC_filt = single_shot_DC[~np.isin(np.arange(len(single_shot_AC)), exclude_pts)]
    offsets_filt = offsets[~np.isin(np.arange(len(single_shot_AC)), exclude_pts)]
    single_shot_DC_v = -u.demod2volts(single_shot_DC_filt, readout_len) # invert DC because the OPX ADCs are inverted. We keep AC the same to see the real error signal we use.
    
    ###################
    # Analyse results #
    ###################

    #calculate average
    AC_avg = np.mean(single_shot_AC_filt)
    # #get the resonance offset from peaks in the DC signal
    peak_indices=find_peaks(-single_shot_DC_v+np.max(single_shot_DC_v),height =0.5*(np.max(single_shot_DC_v)-np.min(single_shot_DC_v)))[0]
    # Select the peak closest to zero volt offset (away from OPX range edges)
    bestpeak_index = peak_indices[np.argmin(np.abs(offsets_filt[peak_indices]))]
    resonance_offset = offsets_filt[bestpeak_index]
    axis_cut = single_shot_AC_filt[bestpeak_index] # Target value of the Error Signal to lock to the minimum

    ########
    # Plot #
    ########

    fig, axs=plt.subplots(2,1,figsize=(12,8), sharex=True,layout='tight')
    #plot DC signal
    axs[0].plot(offsets_filt, single_shot_DC_v, "--", marker="o",markersize=4)
    axs[0].axvline(resonance_offset, color="k", linestyle="--", label=f"resonance offset= {resonance_offset*1e3:.1f} mV")
    axs[0].set_ylabel("Reflection signal [V]")
    axs[0].legend()
    # plot AC signal in demodulated units
    axs[1].plot(offsets_filt, single_shot_AC_filt, "--", label="Error signal", marker="o", markersize=4)
    axs[1].axhline(axis_cut, color = "orange", linestyle="-", label=f"Set Target= {axis_cut:.5f}")
    axs[1].axvline(resonance_offset, color="k", linestyle="--", label=f"resonance offset= {resonance_offset*1e3:.1f} mV")
    axs[1].set_xlabel("Offset [V]")
    axs[1].set_ylabel("Error signal [demod. units]")
    axs[1].legend()
    print(f"AC signal average in demod. units : {AC_avg:.5f}")
    plt.show()

    ############
    # Close QM #
    ############
    qmm.close() # Close the Quantum Machine

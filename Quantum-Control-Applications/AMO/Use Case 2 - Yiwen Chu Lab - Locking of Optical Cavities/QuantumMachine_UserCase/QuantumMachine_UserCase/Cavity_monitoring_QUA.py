from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration_cavity_locking_ETHZ_OPX1 import *
from qualang_tools.addons.variables import assign_variables_to_element
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
import matplotlib.pyplot as plt
import warnings
import time
from decimal import Decimal
from datetime import datetime

warnings.filterwarnings("ignore")

def PID_derivation(input_signal, bitshift_scale_factor, gain_P, gain_I, gain_D, alpha, target):
    error = declare(fixed)
    integrator_error = declare(fixed)
    derivative_error = declare(fixed)
    old_error = declare(fixed)
    corr=declare(fixed)
    # input_signal_V = declare(fixed)

    # Convert input signal from demod unit into Volt
    # assign(input_signal_V, input_signal*4096/readout_len)
    # calculate the error
    assign(error, (target - input_signal) << bitshift_scale_factor) #bitshift the error
    # calculate the integrator error with exponentially decreasing weights with coefficient alpha, see https://en.wikipedia.org/wiki/Exponential_smoothing
    assign(integrator_error, (1.0 - alpha) * integrator_error + alpha * error)
    # calculate the derivative error
    assign(derivative_error, old_error - error)
    # save old error to be error
    assign(old_error, error)

    #TestSam
    assign(corr, gain_P * error + gain_I * integrator_error + gain_D * derivative_error)
    # return gain_P * error + gain_I * integrator_error + gain_D * derivative_error, error, integrator_error, derivative_error
    return corr, error, integrator_error, derivative_error


###################
# The QUA program #
###################


def PID_monitor_prog(bitshift_scale_factor=9, gain_P=0.0, gain_I=0.0, gain_D=0.0, alpha=0.0, target=0.0, N_outer_repeat=1, lock_time=4, wait_time=4):
    with program() as prog:
        # Results variables
        I = declare(fixed)
        Q = declare(fixed)
        single_shot_DC = declare(fixed)
        single_shot_AC = declare(fixed)
        n_outer_repeat = declare(int)
        n_inner_repeat = declare(int)
        N_inner_repeat = declare(int)
        pid_on = declare(bool)

        # PID variables
        bitshift_scale_factor_qua = declare(int, value=bitshift_scale_factor)  ## scale_factor = 2**bitshift_scale_factor
        gain_P_qua = declare(fixed, value=gain_P)
        gain_I_qua = declare(fixed, value=gain_I)
        gain_D_qua = declare(fixed, value=gain_D)
        alpha_qua = declare(fixed, value=alpha)
        target_qua = declare(fixed, value=target)
        dc_offset_1 = declare(fixed)
        correction=declare(fixed)

        # Streams
        single_shot_st = declare_stream()
        error_st = declare_stream()
        integrator_error_st = declare_stream()
        derivative_error_st = declare_stream()
        offset_st = declare_stream()
        single_shot_AC_st = declare_stream()

        # Debugging: Show PID Parameters and Target
        gain_P_st = declare_stream()
        gain_I_st = declare_stream()
        gain_D_st = declare_stream()
        target_st = declare_stream()
        # Ensure that the results variables are assigned to the measurement elements
        assign_variables_to_element("detector_DC", single_shot_DC)
        assign_variables_to_element("detector_AC", I, Q, single_shot_AC)

        # Repeat the experiment N_outer_repeat times
        with for_(n_outer_repeat, 0, n_outer_repeat < N_outer_repeat, n_outer_repeat + 1):
            # Toggle between PID "on" and "off"
            with for_each_(pid_on, [True, False]): #just changed the order of the True and False for the locking to happen before the unlocking 
                # Update the PID parameters based on pid_on.
                with if_(pid_on):
                    assign(gain_P_qua, gain_P)
                    assign(gain_I_qua, gain_I)
                    # Update the N_inner_repeat based on the readout length in ns (or the duration of the most inner sequence)
                    assign(N_inner_repeat, int(lock_time*1e9 / readout_len))
                with else_():
                    assign(gain_P_qua, 0.0)
                    assign(gain_I_qua, 0.0)
                    assign(N_inner_repeat, int(wait_time*1e9 / readout_len))

                # Measure N_inner_repeat times for a given set of PID gains
                with for_(n_inner_repeat, 0, n_inner_repeat < N_inner_repeat, n_inner_repeat + 1):

                    # Ensure that the two digital oscillators will start with the same phase
                    reset_phase("phase_modulator")
                    reset_phase("detector_AC")
            
                    # Adjust the phase delay between the two
                    # frame_rotation_2pi(angle, "detector_AC") #rotate the detector phase
                    frame_rotation_2pi(angle, "phase_modulator") #or rotate the phase modulator phase
                    # Sync all the elements
                    align()
                    # Play the PDH sideband
                    play("cw", "phase_modulator")
                    # Measure and integrate the signal received by the detector --> DC measurement
                    measure("readout", "detector_DC", None, integration.full("constant", single_shot_DC, "out1"))
                    # Measure and demodulate the signal received by the detector --> AC measurement sqrt(I**2 + Q**2)
                    measure("readout", "detector_AC", None, demod.full("constant", I, "out1"), demod.full("constant", Q, "out1")) #still inverted
            
                    assign(single_shot_AC, I)

                    # PID correction signal
                    correction, error, integrator_error, derivative_error = PID_derivation(single_shot_AC, bitshift_scale_factor_qua, gain_P_qua, gain_I_qua, gain_D_qua, alpha_qua, target_qua)
                    # Update the DC offset
                    assign(dc_offset_1, dc_offset_1+correction) # The way "offset" works is that it adds up correction to the previous value, because it's a "sticky". See "sticky" QUA documentation

                    # Apply the correction
                    play("offset" * amp(correction * 4), "filter_cavity_1") # Note that the previously sent analog signal at the end of the pulse is played until the next pulse is sent, done through using a QUA sticky element. Search "sticky" in the configuration_cavity_locking_ETHZ_OPX1.py file.

                    # Save the desired variables
                    save(single_shot_DC, single_shot_st)
                    save(dc_offset_1, offset_st)
                    # save(correction, offset_st)

                    save(error, error_st)
                    save(derivative_error, derivative_error_st)
                    save(integrator_error, integrator_error_st)
                    save(single_shot_AC, single_shot_AC_st)

                    # Debugging: Show PID Parameters and Target
                    save(gain_P_qua, gain_P_st)
                    save(gain_I_qua, gain_I_st)
                    save(gain_D_qua, gain_D_st)
                    save(target_qua, target_st)

        with stream_processing():

            # Get the data from pid_on False and True on the same buffer
            error_st.buffer(int(lock_time*1e9 / readout_len) + int(wait_time*1e9 / readout_len)).save_all("error")
            integrator_error_st.buffer(int(lock_time*1e9 / readout_len) + int(wait_time*1e9 / readout_len)).save_all("integration_error")
            derivative_error_st.buffer(int(lock_time*1e9 / readout_len) + int(wait_time*1e9 / readout_len)).save_all("derivative_error")
            single_shot_st.buffer(int(lock_time*1e9 / readout_len) + int(wait_time*1e9 / readout_len)).save_all("single_shot")
            offset_st.buffer(int(lock_time*1e9 / readout_len) + int(wait_time*1e9 / readout_len)).save_all("offset")
            single_shot_AC_st.buffer(int(lock_time*1e9 / readout_len) + int(wait_time*1e9 / readout_len)).save_all("single_shot_AC")

            gain_P_st.buffer(int(lock_time*1e9 / readout_len) + int(wait_time*1e9 / readout_len)).save_all("gain_P")
            gain_I_st.buffer(int(lock_time*1e9 / readout_len) + int(wait_time*1e9 / readout_len)).save_all("gain_I")
            gain_D_st.buffer(int(lock_time*1e9 / readout_len) + int(wait_time*1e9 / readout_len)).save_all("gain_D")
            target_st.buffer(int(lock_time*1e9 / readout_len) + int(wait_time*1e9 / readout_len)).save_all("target_fetch")

            gain_P_st.timestamps().buffer(int(lock_time*1e9 / readout_len) + int(wait_time*1e9 / readout_len)).save_all("timestamps")
            # Alternatively this would create a 2d array gain_P_st.buffer(N_inner_repeat).buffer(2).save("gain_P")

    return prog

#####################################
#  Open Communication with the QOP  #
#####################################

if __name__ == '__main__':

    # The parameters for the PID_prog function call. 
    # These are the initial values, adjusting while measuring is done with the Cavity_locking_update_parameters.py script.
    set_bitshift = 0 # Bitshift scale factor
    set_alpha = 0.02  # Exponential weight for the integral error
    set_target = 8e-5 # Set-point to which the PID should converge (demodulation units)

    angle = 0*config_angle  # Phase angle between the sideband and demodulation in units of 2pi
    wait_time = 1000 # 4 is the lowest possible value for wait_time

    qmm = QuantumMachinesManager(host=qop_ip, port=9510)
    lockTime=2
    waitTime=6
    gain_P,gain_I=-2e-2,0.0 # WORKING VALUES
    gain_D=0.0
    N_repeat=10
    prog = PID_monitor_prog(bitshift_scale_factor=set_bitshift, gain_P=gain_P, gain_I=gain_I, gain_D=gain_D, alpha=set_alpha, target=set_target, N_outer_repeat=N_repeat, lock_time=lockTime, wait_time=waitTime)
    # clear the results to free the memory
    qmm.clear_all_job_results()

    qm = qmm.open_qm(config)

    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(prog)
    
    # Debugging: Show PID Parameters and Target
    results = fetching_tool(job, ["error", "integration_error", "derivative_error", "single_shot", "offset", "single_shot_AC", "gain_P", "gain_I", "gain_D", "target_fetch","timestamps"], mode="live")
    
    
    fig = plt.figure(figsize=(12, 8))
    interrupt_on_close(fig, job) #ensures that the infinite loop is stopped when the figure is closed

    # Initialize data arrays for later saving
    error_data = np.array([])
    integration_error_data = np.array([])
    derivative_error_data = np.array([])
    single_shot_DC_V_data = np.array([])
    offset_data = np.array([])
    variance_data = np.array([])
    single_shot_AC_data = np.array([])
    gain_P_data = np.array([])
    gain_I_data = np.array([])
    timestamps_data=np.array([])

    t0=time.time()
    while results.is_processing():
        
        error, integration_error, derivative_error, single_shot_DC, offset, single_shot_AC, gain_P_fetched, gain_I_fetched, gain_D_fetched, target_fetched,timestamps_fetched = results.fetch_all()

        #convert single_shot to voltage
        single_shot_DC_volts = -u.demod2volts(single_shot_DC, readout_len) # invert DC because the OPX ADCs are inverted.
        single_shot_AC_volts = u.demod2volts(single_shot_AC, readout_len)*np.sqrt(2) #convert to voltage, still inverted

        # Accumulate data for later saving
        error_data = np.concatenate(error)
        integration_error_data = np.concatenate(integration_error)
        derivative_error_data = np.concatenate(derivative_error)
        single_shot_DC_V_data = np.concatenate(single_shot_DC_volts)
        offset_data = np.concatenate(offset)
        single_shot_AC_data =  np.concatenate(single_shot_AC_volts)
        gain_P_data = np.concatenate(gain_P_fetched)
        gain_I_data = np.concatenate(gain_I_fetched)
        timestamps_data = np.concatenate(timestamps_fetched)

        n_min = min([len(single_shot_DC_V_data), len(timestamps_data)])
        samples=1 # plot every "samples" number of samples
        len_plot=100
        plot_live=False
        if plot_live:
            plt.cla()
            plt.plot(1e-9*(np.array(timestamps_data[:n_min]) - timestamps_data[0]), single_shot_DC_V_data[:n_min])
            plt.pause(0.1)
    # Saving data for external analysis   
    #date = str(datetime.now()).replace(":","_")
    params_dict_data={"P" : gain_P, "I" : gain_I, "waitTime" : waitTime, "lockTime" : lockTime, "N_repeat" : N_repeat }
    date = datetime.now().strftime('%Y-%m-%d %Hh%M_%S')

    print('Saving data...')

    np.savez(f"QuantumMachine_UserCase/Cavity Locking Data/PID_monitoring_QUA {date}Gain_P{np.round(1e2*gain_P,decimals=2)}e-2_Gain_I{np.round(gain_I*1e4,decimals=2)}e-4_CustomAmp", error=error_data, integration_error=integration_error_data, derivative_error=derivative_error_data, single_shot_DC=single_shot_DC_V_data, offset=offset_data, single_shot_AC=single_shot_AC_data,gain_P=gain_P_data,gain_I=gain_I_data,timestamps=timestamps_data, params_dict = params_dict_data)
    
    print('Saved')
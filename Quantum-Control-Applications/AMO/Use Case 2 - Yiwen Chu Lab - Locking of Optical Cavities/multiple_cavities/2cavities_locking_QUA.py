from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration_cavity_locking_ETHZ_OPX1 import *
from qualang_tools.addons.variables import assign_variables_to_element
from qualang_tools.results import fetching_tool
from qualang_tools.loops import from_array, qua_linspace
import matplotlib.pyplot as plt
import warnings
from scipy.special import erf_zeros
from scipy.signal import find_peaks
import time

warnings.filterwarnings("ignore")


###################
# The QUA program #
###################
def PID_derivation(input_signal,corr,error,integrator_error,derivative_error,old_error, bitshift_scale_factor, gain_P, gain_I, gain_D, alpha, target):
    # # input_signal_V = declare(fixed)
    # corr=declare(fixed)
    # error = declare(fixed)
    # integrator_error = declare(fixed)
    # derivative_error = declare(fixed)
    # old_error = declare(fixed)
    
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
    return(corr)

def switches_preset_locking(cavity,settle_time=20e-3):
    '''Sets RF and optical switches states for a time settle_time, in order to account for non-zero switching times'''

    n_settle = declare(int)
    N_settle=declare(int)
    assign(N_settle, int(settle_time*1e9 / readout_len))
    
    with for_(n_settle, 0, n_settle < N_settle, n_settle + 1):
        align()
        # Select which photodiode signal is sent to the OPX input, using SP4T RF swich
        with if_(cavity=="filter_cavity_1"):
            play("switch", "RFswitch_pin0")
        with elif_(cavity=="filter_cavity_2"):
            play("switch", "RFswitch_pin1")
        with elif_(cavity=="filter_cavity_3"):
            play("switch", "RFswitch_pin0") #With multiplexer
            play("switch", "RFswitch_pin1")
        # Configure laser path with optical switches
        # play("switch", "opticalswitch_23") #A single optical switch element is compatible with the table above

def measure_macro(cavity,I,Q,N_avg=1):
    n_avg = declare(int)
    single_shot_DC = declare(fixed)
    single_shot_AC = declare(fixed)
    #with for_(n_avg, 0, n_avg<N_avg, n_avg+1):
    # Ensure that the two digital oscillators will start with the same phase
    reset_phase("phase_modulator")
    reset_phase("detector_AC")
    # Adjust the phase delay between the two
    # frame_rotation_2pi(angle, "detector_AC")
    frame_rotation_2pi(angle, "phase_modulator")
    # Sync all the elements
    align("phase_modulator", "detector_AC", "detector_DC") # Wait for previous block of pulses (from previous loop iteration) to be finished
    
    # Select which photodiode signal is sent to the OPX input, using SP4T RF swich
    # with if_(cavity_lookup[cavity]==1):
    #     play("switchON", "RFswitch_pin0")
    # with elif_(cavity=="filter_cavity_2"):
    #     play("switch", "RFswitch_pin1")
    # with elif_(cavity=="filter_cavity_3"):
    #     play("switch", "RFswitch_pin0") #With multiplexer
    #     play("switch", "RFswitch_pin1")
    # Configure laser path with optical switches
    # play("switch", "opticalswitch_23") #A single optical switch element is compatible with the table above
    # Play the PDH sideband
    play("cw", "phase_modulator")
    # Measure and integrate the signal received by the detector --> DC measurement
    measure("readout", "detector_DC", None, integration.full("constant", single_shot_DC, "out1"))
    # Measure and demodulate the signal received by the detector --> AC measurement sqrt(I**2 + Q**2)
    measure("readout", "detector_AC", None, demod.full("constant", I, "out1"), demod.full("constant", Q, "out1"))
    # assign(single_shot_AC, Math.sqrt(I*I + Q*Q))
    assign(single_shot_AC, I)
    reset_frame("phase_modulator") #reset the phase to undo the angle rotation above
    
    return(single_shot_AC)

def measure_macro_noRFswitch(cavity,I,Q,N_avg=1):
    n_avg = declare(int)
    single_shot_DC = declare(fixed)
    single_shot_AC = declare(fixed)
    with for_(n_avg, 0, n_avg<N_avg, n_avg+1):
        # Ensure that the two digital oscillators will start with the same phase
        reset_phase("phase_modulator")
        reset_phase("detector_AC")
        # Adjust the phase delay between the two
        # frame_rotation_2pi(angle, "detector_AC")
        frame_rotation_2pi(angle, "phase_modulator")
        # Sync all the elements
        align() # Wait for previous block of pulses (from previous loop iteration) to be finished
        
        # Select which photodiode signal is sent to the OPX input, using SP4T RF swich
        # with if_(cavity_lookup[cavity]==1):
        #    play("switchOFF", "RFswitch_pin0")
        # with elif_(cavity=="filter_cavity_2"):
        #     play("switch", "RFswitch_pin1")
        # with elif_(cavity=="filter_cavity_3"):
        #     play("switch", "RFswitch_pin0") #With multiplexer
        #     play("switch", "RFswitch_pin1")
        # Configure laser path with optical switches
        # play("switch", "opticalswitch_23") #A single optical switch element is compatible with the table above
        # Play the PDH sideband
        play("cw", "phase_modulator")
        # Measure and integrate the signal received by the detector --> DC measurement
        measure("readout", "detector_DC", None, integration.full("constant", single_shot_DC, "out1"))
        # Measure and demodulate the signal received by the detector --> AC measurement sqrt(I**2 + Q**2)
        measure("readout", "detector_AC", None, demod.full("constant", I, "out1"), demod.full("constant", Q, "out1"))
        # assign(single_shot_AC, Math.sqrt(I*I + Q*Q))
        assign(single_shot_AC, I)
        reset_frame("phase_modulator") #reset the phase to undo the angle rotation above
        
    return(single_shot_AC)

def Wait(cavity,I,Q,wait_time):
    n_wait_repeat = declare(int)
    N_wait_repeat = declare(int)
    assign(N_wait_repeat, int(wait_time*1e9 / readout_len))
    with for_(n_wait_repeat, 0, n_wait_repeat < N_wait_repeat, n_wait_repeat + 1):
        _=measure_macro_noRFswitch(cavity,I,Q)

def FastLock(cavity,I,Q,offset_opti,correction,error,int_error,der_error,old_error,gain_P_qua,gain_I_qua,gain_D_qua,lock_time,alpha_qua,target_qua=0.0,bitshift_scale_factor_qua=3):
    '''Engages PID on error signal from low freq PDH'''
    n_fastlock_repeat = declare(int)
    N_fastlock_repeat = declare(int)
    assign(N_fastlock_repeat, int(lock_time*1e9 / readout_len))
    # play("offset" * amp(offset_opti), cavity) #Start at the previous setpoint
    # wait(pause_time)
    n_digital_repeat = declare(int)

    with if_(cavity_lookup[cavity]==1):
        with for_(n_digital_repeat, 0, n_digital_repeat < N_fastlock_repeat, n_digital_repeat + 1):
            play("switchON", "RFswitch_pin0")
    with for_(n_fastlock_repeat, 0, n_fastlock_repeat < N_fastlock_repeat, n_fastlock_repeat + 1):
        single_shot_AC=measure_macro(cavity,I,Q)
        correction= PID_derivation(single_shot_AC,correction,error,int_error,der_error,old_error, bitshift_scale_factor_qua, gain_P_qua, gain_I_qua, gain_D_qua, alpha_qua, target_qua)
        assign(offset_opti, offset_opti+correction)
        play("offset" * amp(correction * 4), cavity) #sticky element, adds correction to previous value


def FullLock(N_repeat,gain_P,gain_I,gain_D,locktime,waittime,alpha,target,bitshift_scale_factor):
    with program() as prog:
        # Results variables
        I = declare(fixed)
        Q = declare(fixed)
        single_shot_DC = declare(fixed)
        single_shot_AC = declare(fixed)
        n_repeat=declare(int)

        # PID variables
        bitshift_scale_factor_qua = declare(int, value=bitshift_scale_factor)  ## scale_factor = 2**bitshift_scale_factor
        gain_P_qua = declare(fixed, value=gain_P)
        gain_I_qua = declare(fixed, value=gain_I)
        gain_D_qua = declare(fixed, value=gain_D)
        alpha_qua = declare(fixed, value=alpha)
        target_qua = declare(fixed, value=target)

        dc_offset_cav1 = declare(fixed)
        correction_cav1=declare(fixed)
        error_cav1=declare(fixed)
        integrator_error_cav1=declare(fixed)
        derivative_error_cav1=declare(fixed)
        old_error_cav1=declare(fixed)

        # dc_offset_cav2 = declare(fixed)
        # correction_cav2=declare(fixed)
        # error_cav2=declare(fixed)
        # integrator_error_cav2=declare(fixed)
        # derivative_error_cav2=declare(fixed)
        # old_error_cav2=declare(fixed)

        # dc_offset_cav3 = declare(fixed)
        # correction_cav3=declare(fixed)
        # error_cav3=declare(fixed)
        # integrator_error_cav3=declare(fixed)    
        # derivative_error_cav3=declare(fixed)
        # old_error_cav3=declare(fixed)

        # Ensure that the results variables are assigned to the measurement elements
        assign_variables_to_element("detector_DC", single_shot_DC)
        assign_variables_to_element("detector_AC", I, Q, single_shot_AC)

        with for_(n_repeat, 0, n_repeat < N_repeat, n_repeat + 1):
            FastLock("filter_cavity_1",I,Q,
                     dc_offset_cav1,correction_cav1,error_cav1,integrator_error_cav1,derivative_error_cav1,old_error_cav1,
                     gain_P_qua,gain_I_qua,gain_D_qua,locktime,alpha_qua,target_qua,bitshift_scale_factor_qua)
            Wait("filter_cavity_1",I,Q,
                 waittime)

            # FastLock("filter_cavity_2",I,Q,
            #          dc_offset_cav2,correction_cav2,error_cav2,integrator_error_cav2,derivative_error_cav2,old_error_cav2,
            #          locktime, gain_P_qua,gain_I_qua,gain_D_qua,alpha_qua,target_qua,bitshift_scale_factor_qua)
            # FastLock("filter_cavity_3",I,Q,
            #          dc_offset_cav3,correction_cav3,error_cav3,integrator_error_cav3,derivative_error_cav3,old_error_cav3,
            #          locktime, gain_P_qua,gain_I_qua,gain_D_qua,alpha_qua,target_qua,bitshift_scale_factor_qua)

    return prog


simulate_y_n = True
sim_duration = 100_000

if __name__ == '__main__':
    #####################################
    #  Open Communication with the QOP  #
    #####################################

    # qmm = QuantumMachinesManager(host=qop_ip, port=9510)
    qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
    # clear the results to free the memory
    qmm.clear_all_job_results()
    qm = qmm.open_qm(config)

    ################
    #  Parameters  #
    ################
    # The parameters for the PID_prog function call. 
    # These are the initial values, adjusting while measuring is done with the Cavity_locking_update_parameters.py script.
    set_bitshift = 3 # Bitshift scale factor
    set_alpha = 0.02  # Exponential weight for the integral error
    set_target = 0 # Set-point to which the PID should converge (demodulation units)

    angle = 0*config_angle  # Phase angle between the sideband and demodulation in units of 2pi
    pause_time = 1000 # 4 is the lowest possible value for wait_time

    lockTime=20
    waitTime=1
    gain_P,gain_I=-1e-2,0.0e-3  # WORKING VALUES
    N_repeat=10

    ###############
    # Run Program #
    ###############

    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    prog=FullLock(N_repeat,gain_P,gain_I,0.0,lockTime,waitTime,set_alpha,set_target,set_bitshift)

    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    if simulate_y_n:
        fig = plt.figure()
        job = qmm.simulate(config, prog, SimulationConfig(duration=sim_duration,))
        samples = job.get_simulated_samples()
        samples.con1.plot()
        plt.show()

    else:
        job = qm.execute(prog)




        
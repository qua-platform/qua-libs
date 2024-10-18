from configuration_cavity_locking_ETHZ_OPX1 import *
from configuration_cavity_locking_ETHZ_OPX1 import RFswitch_lookup
from qm.qua import *


def update_offset(cavity,small_step,small_step_time,N_ss=60):
    '''Update the offset of the cavity. Does it in small steps to avoid big voltage jumps.'''
    nst=declare(int)
    with for_(nst, 0, nst<N_ss, nst+1):
        play("offset" * amp(small_step * 4), cavity)
        wait(small_step_time)

def PID_derivation(input_signal,corr,error,integrator_error,derivative_error,old_error, bitshift_scale_factor, gain_P, gain_I, gain_D, alpha, target):
    # calculate the error
    assign(error, (target - input_signal) << bitshift_scale_factor) #bitshift the error
    # calculate the integrator error with exponentially decreasing weights with coefficient alpha, see https://en.wikipedia.org/wiki/Exponential_smoothing
    assign(integrator_error, (1.0 - alpha) * integrator_error + alpha * error)
    # calculate the derivative error
    assign(derivative_error, old_error - error)
    # save old error to be error
    assign(old_error, error)

    assign(corr, gain_P * error + gain_I * integrator_error + gain_D * derivative_error)
    return(corr)

def opticalswitches_control(config_number):
    '''Plays digital output pulses to optical switches, given the string config_number, in the form "4321".
    For instance, config_number="0101" means that optical switches OS1 and OS3 are ON (logical HIGh state on their corresponding digital output port)'''
    # Select which optical switch is ON (black fiber connected to red fiber)
    with switch_(config_number):
        with case_(0): # Not strictly necessary
            play("switchOFF", "opticalswitch_1") 
            play("switchOFF", "opticalswitch_2")
            play("switchOFF", "opticalswitch_3")
            play("switchOFF", "opticalswitch_4")
        with case_(1):
            play("switchON", "opticalswitch_1")
        with case_(2):
            play("switchON", "opticalswitch_2")
        with case_(3):
            play("switchON", "opticalswitch_1")
            play("switchON", "opticalswitch_2")
        with case_(4):
            play("switchON", "opticalswitch_3")
        with case_(5):
            play("switchON", "opticalswitch_1")
            play("switchON", "opticalswitch_3")
        with case_(6):
            play("switchON", "opticalswitch_2")
            play("switchON", "opticalswitch_3")
        with case_(7):
            play("switchON", "opticalswitch_1")
            play("switchON", "opticalswitch_2")
            play("switchON", "opticalswitch_3")
        with case_(8):
            play("switchON", "opticalswitch_4")
        with case_(9):
            play("switchON", "opticalswitch_1")
            play("switchON", "opticalswitch_4")
        with case_(10):
            play("switchON", "opticalswitch_2")
            play("switchON", "opticalswitch_4")
        with case_(11):
            play("switchON", "opticalswitch_1")
            play("switchON", "opticalswitch_2")
            play("switchON", "opticalswitch_4")
        with case_(12):
            play("switchON", "opticalswitch_3")
            play("switchON", "opticalswitch_4")
        with case_(13):
            play("switchON", "opticalswitch_1")
            play("switchON", "opticalswitch_3")
            play("switchON", "opticalswitch_4")
        with case_(14):
            play("switchON", "opticalswitch_2")
            play("switchON", "opticalswitch_3")
            play("switchON", "opticalswitch_4")
        with case_(15):
            play("switchON", "opticalswitch_1")
            play("switchON", "opticalswitch_2")
            play("switchON", "opticalswitch_3")
            play("switchON", "opticalswitch_4")

def RFswitch_control(config_number):
    '''Select which photodiode signal is sent to the OPX input, using SP8T RF switch'''
    with switch_(config_number):
        with case_(0): # Not strictly necessary
            play("switchOFF", "RFswitch_pin0") 
            play("switchOFF", "RFswitch_pin1")
            play("switchOFF", "RFswitch_pin2")
        with case_(1):
            play("switchON", "RFswitch_pin0")
        with case_(2):
            play("switchON", "RFswitch_pin1")
        with case_(3):
            play("switchON", "RFswitch_pin0")
            play("switchON", "RFswitch_pin1")
        with case_(4):
            play("switchON", "RFswitch_pin2")
        with case_(5):
            play("switchON", "RFswitch_pin0")
            play("switchON", "RFswitch_pin2")
        with case_(6):
            play("switchON", "RFswitch_pin1")
            play("switchON", "RFswitch_pin2")
        with case_(7):
            play("switchON", "RFswitch_pin0")
            play("switchON", "RFswitch_pin1")
            play("switchON", "RFswitch_pin2")

def switches_preset_locking(RFconfig,opticalconfig,settle_time=10e-3):
    '''Sets RF and optical switches states for a time settle_time, in order to account for non-zero switching times'''
    RFconfig_number=declare(int,value=RFswitch_lookup[RFconfig])
    opticalconfig_number=declare(int,value=opticalswitches_lookup[opticalconfig])
    n_settle = declare(int)
    N_settle=declare(int)
    assign(N_settle, int(settle_time*1e9 / readout_len))
    
    with for_(n_settle, 0, n_settle < N_settle, n_settle + 1):
        align()
        # Configure electrical signal path through SP8T switch
        RFswitch_control(RFconfig_number)
        # Configure laser path with optical switches
        opticalswitches_control(opticalconfig_number)

def measure_macro(RFconfig,single_shot_DC_st,single_shot_AC_st,I,Q,savedata,N_avg=1,angle=0.0):
    '''Measure macro used in FastLock. Plays the digital pulses at the same time as the PDH modulator drive pulse.
    single_shot_DC/AC are not arguments, they're local variables declared inside this function.'''
    n_avg = declare(int)
    single_shot_DC = declare(fixed)
    single_shot_AC = declare(fixed)
    RFconfig_number=declare(int,value=RFswitch_lookup[RFconfig])
    opticalconfig_number=declare(int,value=opticalswitches_lookup["0110"]) # Only OS2 and OS3 are ON

    # assign(cavity_number,RFswitch_lookup[cavity])
    with for_(n_avg, 0, n_avg<N_avg, n_avg+1):
        # Ensure that the two digital oscillators will start with the same phase
        reset_phase("phase_modulator")
        reset_phase("detector_AC")
        # Adjust the phase delay between the two
        # frame_rotation_2pi(angle, "detector_AC")
        frame_rotation_2pi(angle, "phase_modulator")
        # Sync all the elements
        align() # Wait for previous block of pulses (from previous loop iteration) to be finished
        
        # Select which photodiode signal is sent to the OPX input, using SP8T RF swich
        RFswitch_control(RFconfig_number)
        # Configure laser path with optical switches
        opticalswitches_control(opticalconfig_number)
        # Play the PDH sideband
        play("cw", "phase_modulator")
        # Measure and integrate the signal received by the detector --> DC measurement
        measure("readout", "detector_DC", None, integration.full("constant", single_shot_DC, "out1"))
        # Measure and demodulate the signal received by the detector --> AC measurement sqrt(I**2 + Q**2)
        measure("readout", "detector_AC", None, demod.full("constant", I, "out1"), demod.full("constant", Q, "out1"))
        # assign(single_shot_AC, Math.sqrt(I*I + Q*Q))
        assign(single_shot_AC, I)
        reset_frame("phase_modulator") #reset the phase to undo the angle rotation above

    with if_(savedata==True):
        save(single_shot_DC, single_shot_DC_st)
        save(single_shot_AC, single_shot_AC_st)
        
    return(single_shot_AC)

def measure_macro_withavg(RFconfig,single_shot_DC_st,single_shot_AC_st,I,Q,savedata,N_avg=2,angle=0.0):
    '''Measure macro used in FastLock. Plays the digital pulses at the same time as the PDH modulator drive pulse.
    single_shot_DC/AC are not arguments, they're local variables declared inside this function.'''
    n_avg = declare(int)
    single_shot_DC = declare(fixed)
    single_shot_DC_temp = declare(fixed)
    single_shot_AC = declare(fixed)
    single_shot_AC_temp = declare(fixed)
    RFconfig_number=declare(int,value=RFswitch_lookup[RFconfig])
    opticalconfig_number=declare(int,value=opticalswitches_lookup["0110"]) #OS2 and OS3 are ON

    # assign(cavity_number,RFswitch_lookup[cavity])
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
        RFswitch_control(RFconfig_number)
        # Configure laser path with optical switches
        opticalswitches_control(opticalconfig_number)
        # play("switch", "opticalswitch_23") #A single optical switch element is compatible with the table above
        # Play the PDH sideband
        play("cw", "phase_modulator")
        # Measure and integrate the signal received by the detector --> DC measurement
        measure("readout", "detector_DC", None, integration.full("constant", single_shot_DC_temp, "out1"))
        # Measure and demodulate the signal received by the detector --> AC measurement sqrt(I**2 + Q**2)
        measure("readout", "detector_AC", None, demod.full("constant", I, "out1"), demod.full("constant", Q, "out1"))
        # assign(single_shot_AC, Math.sqrt(I*I + Q*Q))
        assign(single_shot_AC_temp, I)
        with if_(n_avg==0):
            assign(single_shot_DC,single_shot_DC_temp)
            assign(single_shot_AC,single_shot_AC_temp)
        with else_():
            assign(single_shot_DC, single_shot_DC+single_shot_DC_temp)
            assign(single_shot_AC, single_shot_DC+single_shot_AC_temp)
        reset_frame("phase_modulator") #reset the phase to undo the angle rotation above
        
    assign(single_shot_DC, single_shot_DC/N_avg)
    assign(single_shot_AC, single_shot_AC/N_avg)
    with if_(savedata==True):
        save(single_shot_DC, single_shot_DC_st)
        save(single_shot_AC, single_shot_AC_st)
        
    return(single_shot_AC)

def measure_macro_slowlock(RFconfig, I,Q,single_shot_DC,single_shot_AC,N_avg=1,angle=0.0):
    '''Measure macro used in SlowLock. 
    Does not save data.
    single_shot_DC/AC are arguments, they're not declared inside this function. Indeed, we want them to be modified outside.'''
    n_avg = declare(int)
    RFconfig_number=declare(int,value=RFswitch_lookup[RFconfig])
    opticalconfig_number=declare(int,value=opticalswitches_lookup["0110"]) #OS2 and OS3 are ON

    with for_(n_avg, 0, n_avg<N_avg, n_avg+1):
        # Ensure that the two digital oscillators will start with the same phase
        reset_phase("phase_modulator")
        reset_phase("detector_AC")
        # Adjust the phase delay between the two
        # frame_rotation_2pi(angle, "detector_AC")
        frame_rotation_2pi(angle, "phase_modulator")
        # Sync all the elements
        align()
        # Select which photodiode signal is sent to the OPX input, using SP4T RF swich
        RFswitch_control(RFconfig_number)
        # Configure laser path with optical switches
        opticalswitches_control(opticalconfig_number)

        # Play the PDH sideband
        play("cw", "phase_modulator")
        # Measure and integrate the signal received by the detector --> DC measurement
        measure("readout", "detector_DC", None, integration.full("constant", single_shot_DC, "out1"))
        # Measure and demodulate the signal received by the detector --> AC measurement sqrt(I**2 + Q**2)
        measure("readout", "detector_AC", None, demod.full("constant", I, "out1"), demod.full("constant", Q, "out1"))
        # assign(single_shot_AC, Math.sqrt(I*I + Q*Q))
        assign(single_shot_AC, I)
        #reset_frame("phase_modulator") #reset the phase to undo the angle rotation above
    return(single_shot_DC)

def slowLock(cavity,dc_offset_cav,step):
    ''' Perform slow lock'''
    I = declare(fixed)
    Q = declare(fixed)
    single_shot_DC = declare(fixed)
    single_shot_AC = declare(fixed)
    start_AC=declare(fixed)
    offset = declare(fixed)

    # Measure starting point. single_shot_DC and AC are modified inside the function, which is why they are passed as arguments
    measure_macro_slowlock(cavity,I,Q,single_shot_DC,single_shot_AC)

    assign(start_AC, single_shot_AC) #single_shot_AC has been modified by the function above
    #Sweep offset until the AC signal changes sign
    # with while_(single_shot_AC*start_AC > -1*start_AC*start_AC): #stop when the AC signal changes sign by a significant amount, in unit of |start_AC|^2
    with while_(single_shot_AC*start_AC > 0):
        measure_macro_slowlock(cavity,I,Q,single_shot_DC,single_shot_AC)
        update_offset(cavity,small_step,small_step_time,N_ss)
        assign(offset,offset+step)
        #wait(100)
    assign(dc_offset_cav,offset)

def wait_monitor(cavity,single_shot_DC_st,single_shot_AC_st,I,Q,wait_time):
    '''Dummy function that waits and does not apply PID correction'''
    n_wait_repeat = declare(int)
    N_wait_repeat = declare(int)
    assign(N_wait_repeat, int(wait_time*1e9 / readout_len))
    with for_(n_wait_repeat, 0, n_wait_repeat < N_wait_repeat, n_wait_repeat + 1):
        _=measure_macro(cavity,single_shot_DC_st,single_shot_AC_st,I,Q,savedata=True)
        # _=measure_macro_withavg(cavity,single_shot_DC_st,single_shot_AC_st,I,Q,savedata=True)

def fastLock(cavity,single_shot_DC_st,single_shot_AC_st,I,Q,offset_opti,correction,error,int_error,der_error,old_error,gain_P_qua,gain_I_qua,gain_D_qua,lock_time,alpha_qua,target_qua=0.0,bitshift_scale_factor_qua=3,savedata=True):
    '''Engages PID on error signal from low freq PDH'''
    n_fastlock_repeat = declare(int)
    N_fastlock_repeat = declare(int)
    assign(N_fastlock_repeat, int(lock_time*1e9 / readout_len))
    # play("offset" * amp(offset_opti), cavity) #Start at the previous setpoint
    # wait(pause_time)
    with for_(n_fastlock_repeat, 0, n_fastlock_repeat < N_fastlock_repeat, n_fastlock_repeat + 1):
        single_shot_AC=measure_macro(cavity,single_shot_DC_st,single_shot_AC_st,I,Q,savedata)
        # single_shot_AC=measure_macro_withavg(cavity,single_shot_DC_st,single_shot_AC_st,I,Q,savedata)
        correction= PID_derivation(single_shot_AC,correction,error,int_error,der_error,old_error, bitshift_scale_factor_qua, gain_P_qua, gain_I_qua, gain_D_qua, alpha_qua, target_qua)
        assign(offset_opti, offset_opti+correction)
        play("offset" * amp(correction * 4), cavity) #sticky element, adds correction to previous value

def fullock(locktime,dc_offset_cav1,dc_offset_cav2,dc_offset_cav3,single_shot_DC_cav1_st,single_shot_DC_cav2_st,single_shot_DC_cav3_st,single_shot_AC_cav1_st,single_shot_AC_cav2_st,single_shot_AC_cav3_st,correction_cav1,correction_cav2,correction_cav3,error_cav1,error_cav2,error_cav3,integrator_error_cav1,integrator_error_cav2,integrator_error_cav3,derivative_error_cav1,derivative_error_cav2,derivative_error_cav3,old_error_cav1,old_error_cav2,old_error_cav3,gain_P_qua,gain_I_qua,gain_D_qua,alpha_qua,target_qua,bitshift_scale_factor_qua,savedata=True):
    '''Uses fastLock to lock three cavities in series, one after another'''
    
    savedata_qua=declare(bool,value=savedata)

    I = declare(fixed)
    Q = declare(fixed)
    single_shot_DC = declare(fixed)
    single_shot_AC = declare(fixed)
    assign_variables_to_element("detector_DC", single_shot_DC)
    assign_variables_to_element("detector_AC", I, Q, single_shot_AC)

    # Cavity 1
    fastLock("filter_cavity_1",single_shot_DC_cav1_st,single_shot_AC_cav1_st,I,Q,
                dc_offset_cav1,correction_cav1,error_cav1,integrator_error_cav1,derivative_error_cav1,old_error_cav1,
                gain_P_qua,gain_I_qua,gain_D_qua,locktime,alpha_qua,target_qua,bitshift_scale_factor_qua,savedata_qua)
    # Cavity 2
    fastLock("filter_cavity_2",single_shot_DC_cav2_st,single_shot_AC_cav2_st,I,Q,
                dc_offset_cav2,correction_cav2,error_cav2,integrator_error_cav2,derivative_error_cav2,old_error_cav2,
                gain_P_qua,gain_I_qua,gain_D_qua,locktime,alpha_qua,target_qua,bitshift_scale_factor_qua,savedata_qua)
    # Cavity 3
    fastLock("filter_cavity_3",single_shot_DC_cav3_st,single_shot_AC_cav3_st,I,Q,
                dc_offset_cav3,correction_cav3,error_cav3,integrator_error_cav3,derivative_error_cav3,old_error_cav3,
                gain_P_qua,gain_I_qua,gain_D_qua,locktime,alpha_qua,target_qua,bitshift_scale_factor_qua,savedata_qua)
            
def prelock(dc_offset_cav1,dc_offset_cav2,dc_offset_cav3,gain_P_qua,gain_I_qua,gain_D_qua,alpha_qua,target_qua,bitshift_scale_factor_qua,step,lock_firsttime=1, N_repeat_firsttime=20):
    '''Perform slow locks when starting far from resonance'''

    I = declare(fixed)
    Q = declare(fixed)
    single_shot_DC = declare(fixed)
    single_shot_AC = declare(fixed)
    assign_variables_to_element("detector_DC", single_shot_DC)
    assign_variables_to_element("detector_AC", I, Q, single_shot_AC)
    
    correction_cav1=declare(fixed)
    error_cav1=declare(fixed)
    integrator_error_cav1=declare(fixed)
    derivative_error_cav1=declare(fixed)
    old_error_cav1=declare(fixed)

    correction_cav2=declare(fixed)
    error_cav2=declare(fixed)
    integrator_error_cav2=declare(fixed)
    derivative_error_cav2=declare(fixed)
    old_error_cav2=declare(fixed)

    correction_cav3=declare(fixed)
    error_cav3=declare(fixed)
    integrator_error_cav3=declare(fixed)
    derivative_error_cav3=declare(fixed)
    old_error_cav3=declare(fixed)

    single_shot_DC_cav1_st = declare_stream()
    single_shot_AC_cav1_st = declare_stream()
    single_shot_DC_cav2_st = declare_stream()
    single_shot_AC_cav2_st = declare_stream()
    single_shot_DC_cav3_st = declare_stream()
    single_shot_AC_cav3_st = declare_stream()

    n_repeat=declare(int) #Dummy variables for loops
    slowLock("filter_cavity_1",dc_offset_cav1,step)

    with for_(n_repeat, 0, n_repeat < N_repeat_firsttime, n_repeat + 1):
        fullock(lock_firsttime,dc_offset_cav1,dc_offset_cav2,dc_offset_cav3,
                    single_shot_DC_cav1_st,single_shot_DC_cav2_st,single_shot_DC_cav3_st,
                    single_shot_AC_cav1_st,single_shot_AC_cav2_st,single_shot_AC_cav3_st,
                    correction_cav1,correction_cav2,correction_cav3,
                    error_cav1,error_cav2,error_cav3,
                    integrator_error_cav1,integrator_error_cav2,integrator_error_cav3,
                    derivative_error_cav1,derivative_error_cav2,derivative_error_cav3,
                    old_error_cav1,old_error_cav2,old_error_cav3,
                    gain_P_qua,gain_I_qua,gain_D_qua,alpha_qua,target_qua,bitshift_scale_factor_qua,False)
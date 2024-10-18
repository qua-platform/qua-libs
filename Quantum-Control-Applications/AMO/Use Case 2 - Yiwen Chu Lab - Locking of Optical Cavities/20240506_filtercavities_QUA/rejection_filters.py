from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration_cavity_locking_ETHZ_OPX1 import *
from filter_cavities_lib import *
from qualang_tools.addons.variables import assign_variables_to_element
from qualang_tools.results import fetching_tool
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import time

warnings.filterwarnings("ignore")


###################
# The QUA program #
###################

def main(N_repeat,gain_P,gain_I,gain_D,locktime,alpha,target,bitshift_scale_factor,lock_firsttime=0.5,N_repeat_firsttime=20):
    with program() as prog:
        '''Declare a bunch of variables'''
        # Results variables
        I = declare(fixed)
        Q = declare(fixed)
        single_shot_DC = declare(fixed)
        single_shot_AC = declare(fixed)
        meas_power_time=0.1
        #Loop variables
        n_repeat=declare(int) #Dummy variables for loops

        # PID variables
        #Common variables
        bitshift_scale_factor_qua = declare(int, value=bitshift_scale_factor)  ## scale_factor = 2**bitshift_scale_factor
        gain_P_qua = declare(fixed, value=gain_P)
        gain_I_qua = declare(fixed, value=gain_I)
        gain_D_qua = declare(fixed, value=gain_D)
        alpha_qua = declare(fixed, value=alpha)
        target_qua = declare(fixed, value=target)
        #Cavity1 variables
        dc_offset_cav1 = declare(fixed)
        correction_cav1=declare(fixed)
        error_cav1=declare(fixed)
        integrator_error_cav1=declare(fixed)
        derivative_error_cav1=declare(fixed)
        old_error_cav1=declare(fixed)
        #Cavity2 variables
        dc_offset_cav2 = declare(fixed)
        correction_cav2=declare(fixed)
        error_cav2=declare(fixed)
        integrator_error_cav2=declare(fixed)
        derivative_error_cav2=declare(fixed)
        old_error_cav2=declare(fixed)
        #Cavity3 variables
        dc_offset_cav3 = declare(fixed)
        correction_cav3=declare(fixed)
        error_cav3=declare(fixed)
        integrator_error_cav3=declare(fixed)    
        derivative_error_cav3=declare(fixed)
        old_error_cav3=declare(fixed)

        # Streams
        DC_cav1_st = declare_stream()
        AC_cav1_st = declare_stream()
        DC_cav2_st = declare_stream()
        AC_cav2_st = declare_stream()
        DC_cav3_st = declare_stream()
        AC_cav3_st = declare_stream()

        DC_transmission_beforelock_st = declare_stream()
        AC_transmission_beforelock_st = declare_stream()
        DC_transmission_afterlock_st = declare_stream()
        AC_transmission_afterlock_st = declare_stream()

        # Ensure that the results variables are assigned to the measurement elements
        assign_variables_to_element("detector_DC", single_shot_DC)
        assign_variables_to_element("detector_AC", I, Q, single_shot_AC)

        '''Perform slow locks when starting far from resonance'''
        
        prelock(dc_offset_cav1,dc_offset_cav2,dc_offset_cav3,gain_P_qua,gain_I_qua,gain_D_qua,alpha_qua,target_qua,bitshift_scale_factor_qua,step,lock_firsttime, N_repeat_firsttime)

        '''Loop on the measurement routine'''
        with for_(n_repeat, 0, n_repeat < N_repeat, n_repeat + 1):
            
            # Measure power @PM2
            assign(IO1,2)
            pause() 

            # Measure filter transmission @ PM3 before and after relock
            wait_monitor("filters_transmission",DC_transmission_beforelock_st,AC_transmission_beforelock_st,I,Q,meas_power_time)

            fullock(locktime,dc_offset_cav1,dc_offset_cav2,dc_offset_cav3,
                    DC_cav1_st,DC_cav2_st,DC_cav3_st,
                    AC_cav1_st,AC_cav2_st,AC_cav3_st,
                    correction_cav1,correction_cav2,correction_cav3,
                    error_cav1,error_cav2,error_cav3,
                    integrator_error_cav1,integrator_error_cav2,integrator_error_cav3,
                    derivative_error_cav1,derivative_error_cav2,derivative_error_cav3,
                    old_error_cav1,old_error_cav2,old_error_cav3,
                    gain_P_qua,gain_I_qua,gain_D_qua,alpha_qua,target_qua,bitshift_scale_factor_qua)
            
            wait_monitor("filters_transmission",DC_transmission_afterlock_st,AC_transmission_afterlock_st,I,Q,meas_power_time)

            # Measure power @PM1
            assign(IO1,1)
            pause()

            # Carefully set optical switches in the right order : SPD protection (OS4) last
            switches_preset_locking("cavity_reflection","0001",settle_time=10e-3) 
            switches_preset_locking("cavity_reflection","1001",settle_time=10e-3)

            # Launch SPD measurement
            assign(IO1,3)
            pause() 

            # Wait for SPD measurement to be finished
            assign(SPD_meas_done, IO2) # Should be zero
            with while_(SPD_meas_done==0):
                switches_preset_locking("cavity_reflection","1001",settle_time=10e-3)
                assign(SPD_meas_done, IO2)

            # Reset IO2 such that program enters the while_(SPD_meas_done==0) loop on next iteration
            assign(IO2,0)

        with stream_processing():
            # Save the streams
            DC_cav1_st.buffer(int(locktime*1e9 / readout_len)).save_all("DC_cav1")
            AC_cav1_st.buffer(int(locktime*1e9 / readout_len)).save_all("AC_cav1")
            DC_cav1_st.timestamps().buffer(int(locktime*1e9 / readout_len)).save_all("timestamps_cav1")
            
            DC_cav2_st.buffer(int(locktime*1e9 / readout_len)).save_all("DC_cav2")
            AC_cav2_st.buffer(int(locktime*1e9 / readout_len)).save_all("AC_cav2")
            DC_cav2_st.timestamps().buffer(int(locktime*1e9 / readout_len)).save_all("timestamps_cav2")

            DC_cav3_st.buffer(int(locktime*1e9 / readout_len)).save_all("DC_cav3")
            AC_cav3_st.buffer(int(locktime*1e9 / readout_len)).save_all("AC_cav3")
            DC_cav3_st.timestamps().buffer(int(locktime*1e9 / readout_len)).save_all("timestamps_cav3")

            DC_transmission_beforelock_st.buffer(int(meas_power_time*1e9 / readout_len)).save_all("DC_transmission_beforelock")
            AC_transmission_beforelock_st.buffer(int(meas_power_time*1e9 / readout_len)).save_all("AC_transmission_beforelock")
            DC_transmission_beforelock_st.timestamps().buffer(int(meas_power_time*1e9 / readout_len)).save_all("timestamps_transmission_beforelock")
                                                                                                                           
            DC_transmission_afterlock_st.buffer(int(meas_power_time*1e9 / readout_len)).save_all("DC_transmission_afterlock")
            AC_transmission_afterlock_st.buffer(int(meas_power_time*1e9 / readout_len)).save_all("AC_transmission_afterlock")
            DC_transmission_afterlock_st.timestamps().buffer(int(meas_power_time*1e9 / readout_len)).save_all("timestamps_transmission_afterlock")
    
    return prog

def launch_SPD_dataacquisition(acqtime):
    print("SPD data acqusition launched")
    time.sleep(acqtime)
    qm.set_io2(1)

def results_processing(results):
    '''Continuously fetch data from QUA program streams, and saves data at the end'''
    while results.is_processing():
        #Process data
        DC_cav1,AC_cav1,timestamps_cav1_fetched,DC_cav2,AC_cav2,timestamps_cav2_fetched,DC_cav3,AC_cav3,timestamps_cav3_fetched,DC_transmission_beforelock,AC_transmission_beforelock,timestamps_transmission_beforelock_fetched,DC_transmission_afterlock,AC_transmission_afterlock,timestamps_transmission_afterlock_fetched = results.fetch_all()
        DC_cav1_volts = -u.demod2volts(DC_cav1, readout_len) # invert DC because the OPX ADCs are inverted.
        AC_cav1_volts = u.demod2volts(AC_cav1, readout_len)*np.sqrt(2) #convert to voltage, still inverted

        DC_cav1_data = np.concatenate(DC_cav1_volts)
        AC_cav1_data =  np.concatenate(AC_cav1_volts)
        timestamps_cav1_data = np.concatenate(timestamps_cav1_fetched)

        DC_cav2_volts = -u.demod2volts(DC_cav2, readout_len) # invert DC because the OPX ADCs are inverted.
        AC_cav2_volts = u.demod2volts(AC_cav2, readout_len)*np.sqrt(2) #convert to voltage, still inverted

        DC_cav2_data = np.concatenate(DC_cav2_volts)
        AC_cav2_data =  np.concatenate(AC_cav2_volts)
        timestamps_cav2_data = np.concatenate(timestamps_cav2_fetched)

        DC_cav3_volts = -u.demod2volts(DC_cav3, readout_len) # invert DC because the OPX ADCs are inverted.
        AC_cav3_volts = u.demod2volts(AC_cav3, readout_len)*np.sqrt(2) #convert to voltage, still inverted

        DC_cav3_data = np.concatenate(DC_cav3_volts)
        AC_cav3_data =  np.concatenate(AC_cav3_volts)
        timestamps_cav3_data = np.concatenate(timestamps_cav3_fetched)

        DC_transmission_beforelock_volts = -u.demod2volts(DC_transmission_beforelock, readout_len) # invert DC because the OPX ADCs are inverted.
        AC_transmission_beforelock_volts = u.demod2volts(AC_transmission_beforelock, readout_len)*np.sqrt(2) #convert to voltage, still inverted

        DC_transmission_beforelock_data = np.concatenate(DC_transmission_beforelock_volts)
        AC_transmission_beforelock_data =  np.concatenate(AC_transmission_beforelock_volts)
        timestamps_transmission_beforelock_data = np.concatenate(timestamps_transmission_beforelock_fetched)

        DC_transmission_afterlock_volts = -u.demod2volts(DC_transmission_afterlock, readout_len) # invert DC because the OPX ADCs are inverted.
        AC_transmission_afterlock_volts = u.demod2volts(AC_transmission_afterlock, readout_len)*np.sqrt(2) #convert to voltage, still inverted

        DC_transmission_afterlock_data = np.concatenate(DC_transmission_afterlock_volts)
        AC_transmission_afterlock_data =  np.concatenate(AC_transmission_afterlock_volts)
        timestamps_transmission_afterlock_data = np.concatenate(timestamps_transmission_afterlock_fetched)

    date = datetime.now().strftime('%Y-%m-%d %Hh%M_%S')
    print('Saving data...')
    np.savez(f"FilterData/SPDcounts_{date}",DC_cav1=DC_cav1_data,AC_cav1=AC_cav1_data,timestamps_cav1=timestamps_cav1_data,DC_cav2=DC_cav2_data,AC_cav2=AC_cav2_data,timestamps_cav2=timestamps_cav2_data,DC_cav3=DC_cav3_data,AC_cav3=AC_cav3_data,timestamps_cav3=timestamps_cav3_data,DC_transmission_beforelock=DC_transmission_beforelock_data,AC_transmission_beforelock=AC_transmission_beforelock_data,timestamps_transmission_beforelock=timestamps_transmission_beforelock_data,DC_transmission_afterlock=DC_transmission_afterlock_data,AC_transmission_afterlock=AC_transmission_afterlock_data,timestamps_transmission_afterlock=timestamps_transmission_afterlock_data)    


if __name__ == '__main__':
    #####################################
    #  Open Communication with the QOP  #
    #####################################

    qmm = QuantumMachinesManager(host=qop_ip, port=9510)
    # clear the results to free the memory
    qmm.clear_all_job_results()
    qm = qmm.open_qm(config)

    ################
    #  Parameters  #
    ################

    # The parameters for the PID_prog function call. 
    set_bitshift = 0 # Bitshift scale factor
    set_alpha = 0.02  # Exponential weight for the integral error
    set_target = 0 # Set-point to which the PID should converge (demodulation units)
    gain_P,gain_I,gain_D=-1e-1,0.0,0.0  # WORKING VALUES
  
    #Lock / release parameters
    lockTime=0.01
    acqTime=1
    N_repeat=200
    
    ######################
    # Python data arrays #
    ######################
    powers_PM2=np.zeros(N_repeat)
    powers_PM3_beforelock=np.zeros(N_repeat)
    powers_PM3_afterlock=np.zeros(N_repeat)
    powers_PM1=np.zeros(N_repeat)

    ###########################
    # Connect to power meters #
    ###########################
    PM1=PM100D(PM1_address)
    PM2=PM100D(PM2_address)
    # PM3=PM100D(PM3_address)

    ###############
    # Run Program #
    ###############

    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    prog=main(N_repeat,gain_P,gain_I,gain_D,lockTime,set_alpha,set_target,set_bitshift)

    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    if simulate_y_n:
        fig = plt.figure()
        job = qmm.simulate(config, prog, SimulationConfig(
        duration=sim_duration,))                            # duration of simulation in units of 4ns
        samples = job.get_simulated_samples()
        samples.con1.plot()
        plt.show()

    else:
        # Initialize IO variables
        qm.set_io1(0)
        qm.set_io2(0)
        # Execute program
        job = qm.execute(prog)
        # Declare results fetching tool and launch the results processing thread
        results = fetching_tool(job, ["DC_cav1","AC_cav1","timestamps_cav1","DC_cav2","AC_cav2","timestamps_cav2","DC_cav3","AC_cav3","timestamps_cav3","DC_transmission_beforelock","AC_transmission_beforelock","timestamps_transmission_beforelock","DC_transmission_afterlock","AC_transmission_afterlock","timestamps_transmission_afterlock",], mode="live")
        results_processing_thread=ThreadWithReturnValue(target=results_processing, args=(results,))
        results_processing_thread.start()
        # Loop over number of repetitions
        j=0
        while j<N_repeat:
            # Check is job is paused : QUA program reached a pause statement
            while not job.is_paused():
                time.sleep(0.001)
            # Depending on the value of IO1, decide what to do (it tells us where we are in the QUA program loop)
            whattodo=get_io1_value()
            if whattodo==1:
                # Measure power @PM1
                powers_PM1[j]=PM1.read_value()
            elif whattodo==2:
                # Measure power @PM2
                powers_PM2[j]=PM2.read_value()
            elif whattodo==3:
                # Start SPD count rate acquisition
                SPDcount_thread = ThreadWithReturnValue(target=launch_SPD_dataacquisition, args=(acqTime,))
                SPDcount_thread.start()
                SPDcount_thread.join() # Wait for thread target to be finished
                # Add 1 to the loop counter. We want N_repeat acquisition windows
                j+=1
            # Independent of what we did, we need to tell the QUA program to resume
            resume()

        results_processing_thread.join() # Stop data streaming thread


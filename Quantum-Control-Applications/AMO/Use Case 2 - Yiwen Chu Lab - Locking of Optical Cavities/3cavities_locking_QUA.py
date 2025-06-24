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

warnings.filterwarnings("ignore")


###################
# The QUA program #
###################


def main(
    N_repeat,
    gain_P,
    gain_I,
    gain_D,
    locktime,
    waittime,
    alpha,
    target,
    bitshift_scale_factor,
    lock_firsttime=0.5,
    N_repeat_firsttime=20,
    N_monitor_repeat=20,
):
    with program() as prog:
        # Results variables
        I = declare(fixed)
        Q = declare(fixed)
        single_shot_DC = declare(fixed)
        single_shot_AC = declare(fixed)

        # Loop variables
        n_repeat = declare(int)  # Dummy variables for loops

        # PID variables
        # Common variables
        bitshift_scale_factor_qua = declare(
            int, value=bitshift_scale_factor
        )  ## scale_factor = 2**bitshift_scale_factor
        gain_P_qua = declare(fixed, value=gain_P)
        gain_I_qua = declare(fixed, value=gain_I)
        gain_D_qua = declare(fixed, value=gain_D)
        alpha_qua = declare(fixed, value=alpha)
        target_qua = declare(fixed, value=target)
        # Cavity1 variables
        dc_offset_cav1 = declare(fixed)
        correction_cav1 = declare(fixed)
        error_cav1 = declare(fixed)
        integrator_error_cav1 = declare(fixed)
        derivative_error_cav1 = declare(fixed)
        old_error_cav1 = declare(fixed)
        # Cavity2 variables
        dc_offset_cav2 = declare(fixed)
        correction_cav2 = declare(fixed)
        error_cav2 = declare(fixed)
        integrator_error_cav2 = declare(fixed)
        derivative_error_cav2 = declare(fixed)
        old_error_cav2 = declare(fixed)
        # Cavity3 variables
        dc_offset_cav3 = declare(fixed)
        correction_cav3 = declare(fixed)
        error_cav3 = declare(fixed)
        integrator_error_cav3 = declare(fixed)
        derivative_error_cav3 = declare(fixed)
        old_error_cav3 = declare(fixed)

        # Streams
        single_shot_DC_cav1_st = declare_stream()
        single_shot_AC_cav1_st = declare_stream()
        single_shot_DC_cav2_st = declare_stream()
        single_shot_AC_cav2_st = declare_stream()
        single_shot_DC_cav3_st = declare_stream()
        single_shot_AC_cav3_st = declare_stream()

        single_shot_DC_transmission_st = declare_stream()
        single_shot_AC_transmission_st = declare_stream()
        # Ensure that the results variables are assigned to the measurement elements
        assign_variables_to_element("detector_DC", single_shot_DC)
        assign_variables_to_element("detector_AC", I, Q, single_shot_AC)

        """Perform slow locks when starting far from resonance"""

        prelock(
            dc_offset_cav1,
            dc_offset_cav2,
            dc_offset_cav3,
            gain_P_qua,
            gain_I_qua,
            gain_D_qua,
            alpha_qua,
            target_qua,
            bitshift_scale_factor_qua,
            step,
            lock_firsttime,
            N_repeat_firsttime,
        )

        """Loop on the fast lock of successive cavities"""
        with for_(n_repeat, 0, n_repeat < N_repeat, n_repeat + 1):

            fullock(
                locktime,
                dc_offset_cav1,
                dc_offset_cav2,
                dc_offset_cav3,
                single_shot_DC_cav1_st,
                single_shot_DC_cav2_st,
                single_shot_DC_cav3_st,
                single_shot_AC_cav1_st,
                single_shot_AC_cav2_st,
                single_shot_AC_cav3_st,
                correction_cav1,
                correction_cav2,
                correction_cav3,
                error_cav1,
                error_cav2,
                error_cav3,
                integrator_error_cav1,
                integrator_error_cav2,
                integrator_error_cav3,
                derivative_error_cav1,
                derivative_error_cav2,
                derivative_error_cav3,
                old_error_cav1,
                old_error_cav2,
                old_error_cav3,
                gain_P_qua,
                gain_I_qua,
                gain_D_qua,
                alpha_qua,
                target_qua,
                bitshift_scale_factor_qua,
            )

            switches_preset_locking("filters_transmission", settle_time=200e-6)
            wait_monitor(
                "filters_transmission", single_shot_DC_transmission_st, single_shot_AC_transmission_st, I, Q, waittime
            )

        with stream_processing():
            # Save the streams

            single_shot_DC_cav1_st.buffer(int(locktime * 1e9 / readout_len)).save_all("single_shot_DC_cav1")
            single_shot_AC_cav1_st.buffer(int(locktime * 1e9 / readout_len)).save_all("single_shot_AC_cav1")
            single_shot_DC_cav1_st.timestamps().buffer(int(locktime * 1e9 / readout_len)).save_all("timestamps_cav1")

            single_shot_DC_cav2_st.buffer(int(locktime * 1e9 / readout_len)).save_all("single_shot_DC_cav2")
            single_shot_AC_cav2_st.buffer(int(locktime * 1e9 / readout_len)).save_all("single_shot_AC_cav2")
            single_shot_DC_cav2_st.timestamps().buffer(int(locktime * 1e9 / readout_len)).save_all("timestamps_cav2")

            single_shot_DC_cav3_st.buffer(int(locktime * 1e9 / readout_len)).save_all("single_shot_DC_cav3")
            single_shot_AC_cav3_st.buffer(int(locktime * 1e9 / readout_len)).save_all("single_shot_AC_cav3")
            single_shot_DC_cav3_st.timestamps().buffer(int(locktime * 1e9 / readout_len)).save_all("timestamps_cav3")

            single_shot_DC_transmission_st.buffer(int(waittime * 1e9 / readout_len)).save_all(
                "single_shot_DC_transmission"
            )
            single_shot_AC_transmission_st.buffer(int(waittime * 1e9 / readout_len)).save_all(
                "single_shot_AC_transmission"
            )
            single_shot_DC_transmission_st.timestamps().buffer(int(waittime * 1e9 / readout_len)).save_all(
                "timestamps_transmission"
            )

    return prog


simulate_y_n = False
sim_duration = 100000

if __name__ == "__main__":
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
    set_bitshift = 0  # Bitshift scale factor
    set_alpha = 0.02  # Exponential weight for the integral error
    set_target = 0  # Set-point to which the PID should converge (demodulation units)
    gain_P, gain_I, gain_D = -1e-1, 0.0, 0.0  # WORKING VALUES

    # Lock / release parameters
    lockTime = 0.01
    waitTime = 5
    N_repeat = 50

    ###############
    # Run Program #
    ###############

    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    prog = main(N_repeat, gain_P, gain_I, gain_D, lockTime, waitTime, set_alpha, set_target, set_bitshift)

    if simulate_y_n:
        fig = plt.figure()
        job = qmm.simulate(
            config,
            prog,
            SimulationConfig(
                duration=sim_duration,
            ),
        )  # duration of simulation in units of 4ns
        samples = job.get_simulated_samples()
        samples.con1.plot()
        plt.show()

    else:
        job = qm.execute(prog)
        results = fetching_tool(
            job,
            [
                "single_shot_DC_cav1",
                "single_shot_AC_cav1",
                "timestamps_cav1",
                "single_shot_DC_cav2",
                "single_shot_AC_cav2",
                "timestamps_cav2",
                "single_shot_DC_cav3",
                "single_shot_AC_cav3",
                "timestamps_cav3",
                "single_shot_DC_transmission",
                "single_shot_AC_transmission",
                "timestamps_transmission",
            ],
            mode="live",
        )

        while results.is_processing():
            # Process data
            (
                single_shot_DC_cav1,
                single_shot_AC_cav1,
                timestamps_cav1_fetched,
                single_shot_DC_cav2,
                single_shot_AC_cav2,
                timestamps_cav2_fetched,
                single_shot_DC_cav3,
                single_shot_AC_cav3,
                timestamps_cav3_fetched,
                single_shot_DC_transmission,
                single_shot_AC_transmission,
                timestamps_transmission_fetched,
            ) = results.fetch_all()

            # Cavities reflection data
            single_shot_DC_cav1_volts = -u.demod2volts(
                single_shot_DC_cav1, readout_len
            )  # invert DC because the OPX ADCs are inverted.
            single_shot_AC_cav1_volts = u.demod2volts(single_shot_AC_cav1, readout_len) * np.sqrt(
                2
            )  # convert to voltage, still inverted

            single_shot_DC_cav1_data = np.concatenate(single_shot_DC_cav1_volts)
            single_shot_AC_cav1_data = np.concatenate(single_shot_AC_cav1_volts)
            timestamps_cav1_data = np.concatenate(timestamps_cav1_fetched)

            single_shot_DC_cav2_volts = -u.demod2volts(
                single_shot_DC_cav2, readout_len
            )  # invert DC because the OPX ADCs are inverted.
            single_shot_AC_cav2_volts = u.demod2volts(single_shot_AC_cav2, readout_len) * np.sqrt(
                2
            )  # convert to voltage, still inverted

            single_shot_DC_cav2_data = np.concatenate(single_shot_DC_cav2_volts)
            single_shot_AC_cav2_data = np.concatenate(single_shot_AC_cav2_volts)
            timestamps_cav2_data = np.concatenate(timestamps_cav2_fetched)

            single_shot_DC_cav3_volts = -u.demod2volts(
                single_shot_DC_cav3, readout_len
            )  # invert DC because the OPX ADCs are inverted.
            single_shot_AC_cav3_volts = u.demod2volts(single_shot_AC_cav3, readout_len) * np.sqrt(
                2
            )  # convert to voltage, still inverted

            single_shot_DC_cav3_data = np.concatenate(single_shot_DC_cav3_volts)
            single_shot_AC_cav3_data = np.concatenate(single_shot_AC_cav3_volts)
            timestamps_cav3_data = np.concatenate(timestamps_cav3_fetched)

            # Filter cascade transmission data
            single_shot_DC_transmission_volts = -u.demod2volts(
                single_shot_DC_transmission, readout_len
            )  # invert DC because the OPX ADCs are inverted.
            single_shot_AC_transmission_volts = u.demod2volts(single_shot_AC_transmission, readout_len) * np.sqrt(
                2
            )  # convert to voltage, still inverted

            single_shot_DC_transmission_data = np.concatenate(single_shot_DC_transmission_volts)
            single_shot_AC_transmission_data = np.concatenate(single_shot_AC_transmission_volts)
            timestamps_transmission_data = np.concatenate(timestamps_transmission_fetched)

        date = datetime.now().strftime("%Y-%m-%d %Hh%M_%S")
        print("Saving data...")
        np.savez(
            f"Cavity Locking Data/FullLock_data/3Cavities/3cavities_transmission_monitoring {date}Gain_P{np.round(1e2*gain_P,decimals=2)}e-2_Gain_I{np.round(gain_I*1e4,decimals=2)}e-4_CustomAmp",
            single_shot_DC_cav1=single_shot_DC_cav1_data,
            single_shot_AC_cav1=single_shot_AC_cav1_data,
            timestamps_cav1=timestamps_cav1_data,
            single_shot_DC_cav2=single_shot_DC_cav2_data,
            single_shot_AC_cav2=single_shot_AC_cav2_data,
            timestamps_cav2=timestamps_cav2_data,
            single_shot_DC_cav3=single_shot_DC_cav3_data,
            single_shot_AC_cav3=single_shot_AC_cav3_data,
            timestamps_cav3=timestamps_cav3_data,
            single_shot_DC_transmission=single_shot_DC_transmission_data,
            single_shot_AC_transmission=single_shot_AC_transmission_data,
            timestamps_transmission=timestamps_transmission_data,
        )

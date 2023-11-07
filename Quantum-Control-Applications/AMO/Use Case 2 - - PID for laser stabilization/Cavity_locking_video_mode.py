from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration_cavity_locking import *
from qualang_tools.addons.variables import assign_variables_to_element
from qualang_tools.results import fetching_tool
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


###################
# The QUA program #
###################
target = -0.001
# bitshift_scale_factor = 9  ## scale_factor = 2**bitshift_scale_factor
# gain_P = -1e-4
# gain_I = 0.0
# gain_D = 0.0
# alpha = 0.1
angle = 0.0
N_shots = 1000000
variance_window = 100
variance_threshold = 0.0001

def PID_derivation(input_signal, bitshift_scale_factor, gain_P, gain_I, gain_D, alpha, target):
    integrator_error = declare(fixed)
    derivative_error = declare(fixed)
    old_error = declare(fixed)
    error = declare(fixed)

    # calculate the error
    assign(error, (target - input_signal) << bitshift_scale_factor)
    # calculate the integrator error with exponentially decreasing weights with coefficient alpha
    assign(integrator_error, (1.0 - alpha) * integrator_error + alpha * error)
    # calculate the derivative error
    assign(derivative_error, old_error - error)

    return gain_P * error + gain_I * integrator_error + gain_D * derivative_error, error

def PID_prog(bitshift_scale_factor=9, gain_P=-1e-4, gain_I=0.0, gain_D=0.0, alpha=0.0, target=0.0):
    with program() as prog:
        # adc_st = declare_stream(adc_trace=True)
        n = declare(int)
        single_shot_DC = declare(fixed)
        I = declare(fixed)
        Q = declare(fixed)
        single_shot_AC = declare(fixed)
        single_shot_st = declare_stream()
        amplitude = declare(fixed, value=1.0)
        correction = declare(fixed)
        dc_offset_1 = declare(fixed)
        error = declare(fixed, value=0)
        amp_st = declare_stream()


        total_error = declare(fixed)
        integrator_error = declare(fixed)
        derivative_error = declare(fixed)
        old_error = declare(fixed)
        error_st = declare_stream()
        integrator_error_st = declare_stream()
        derivative_error_st = declare_stream()
        dc_st = declare_stream()

        variance_vector = declare(fixed, value=[7 for _ in range(variance_window)])
        variance_index = declare(int, value=0)
        variance_st = declare_stream()

        bitshift_scale_factor_qua = declare(int, value=bitshift_scale_factor)  ## scale_factor = 2**bitshift_scale_factor
        gain_P_qua = declare(fixed, value=gain_P)
        gain_I_qua = declare(fixed, value=gain_I)
        gain_D_qua = declare(fixed, value=gain_D)
        alpha_qua = declare(fixed, value=alpha)
        target_qua = declare(fixed, value=target)

        param = declare(int)

        a = declare(fixed)

        assign_variables_to_element("detector_DC", single_shot_DC)
        assign_variables_to_element("detector_AC", I, Q, single_shot_AC)
        # Phase modulator
        assign(a, 1)
        set_dc_offset("filter_cavity_1", "single", 0.0)
        # with infinite_loop_():
        #     play("cw"*amp(a), "phase_modulator")
        with for_(n, 0, n < N_shots, n + 1):
            # Update the PID parameters based on the user input.
            # IO1 specifies the parameter to be updated and IO2 the corresponding value
            assign(param, IO1)
            with if_(param==1):
                assign(bitshift_scale_factor_qua, IO2)
            with elif_(param==2):
                assign(gain_P_qua, IO2)
            with elif_(param==3):
                assign(gain_I_qua, IO2)
            with elif_(param==4):
                assign(gain_D_qua, IO2)
            with elif_(param==5):
                assign(alpha_qua, IO2)
            with elif_(param == 6):
                assign(target_qua, IO2)
        # with while_((Math.abs(Math.max(variance_vector)) - np.abs(target) > variance_threshold) | (Math.abs(Math.min(variance_vector)) - np.abs(target) > variance_threshold)):
            # Ensure that the two digital oscillators will have the same phase
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
            # Measure and demodulate the signal received by the detector --> AC measurement I**2 + Q**2
            measure("readout", "detector_AC", None, demod.full("constant", I, "out1"), demod.full("constant", Q, "out1"))
            assign(single_shot_AC, I*I + Q*Q)
            # PID correction signal
            correction, error = PID_derivation(single_shot_DC, bitshift_scale_factor_qua, gain_P_qua, gain_I_qua, gain_D_qua, alpha_qua,
                           target_qua)
            # assign(correction, PID_derivation(single_shot_DC, bitshift_scale_factor_qua, gain_P_qua, gain_I_qua, gain_D_qua, alpha_qua, target))
            # Apply the correction by updating the DC offset
            assign(dc_offset_1, dc_offset_1 + correction)
            # Handle saturation
            with if_(dc_offset_1 > 0.5 - phase_mod_amplitude):
                assign(dc_offset_1, 0.5 - phase_mod_amplitude)
            with if_(dc_offset_1 < -0.5 + phase_mod_amplitude):
                assign(dc_offset_1, -0.5 + phase_mod_amplitude)
            set_dc_offset("filter_cavity_1", "single", dc_offset_1)
            save(dc_offset_1, dc_st)
            # save old error to be error
            assign(old_error, error)

            # Estimate variance (actually simply max distance from target)
            with if_(variance_index==variance_window):
                assign(variance_index, 0)
            assign(variance_vector[variance_index], single_shot_DC)
            save(variance_vector[variance_index], variance_st)
            assign(variance_index, variance_index+1)

            save(single_shot_DC, single_shot_st)
            save(error, error_st)
            save(dc_offset_1, amp_st)
            save(derivative_error, derivative_error_st)
            save(integrator_error, integrator_error_st)
            wait(100000)

        with stream_processing():
            single_shot_st.buffer(1000).save("single_shot")
            dc_st.buffer(1000).save("variance")
            # amp_st.save_all("amplitude")
            error_st.buffer(1000).save("error")
            # integrator_error_st.save_all("integrator_error")
            # derivative_error_st.save_all("derivative_error")
    return prog

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)

prog = PID_prog(bitshift_scale_factor=2, gain_P=-1e-4, gain_I=0.0, gain_D=0.0, alpha=0.0)
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
    qm.set_io1_value(0)
    qm.set_io2_value(0.0)
    time.sleep(1)
    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(prog)
    results = fetching_tool(job, ["error", "single_shot", "variance"], mode="live")
    fig = plt.figure()
    while results.is_processing():
        error, single_shot, variance = results.fetch_all()
        # error = error[-10000:]
        # single_shot = single_shot[-10000:]
        # variance = variance[-10000:]
        plt.subplot(311)
        plt.cla()
        plt.plot(error, ".-")
        plt.title("Error signal")
        plt.xlabel("Time [μs]")
        plt.ylabel("Amplitude Error [arb. units]")
        plt.subplot(312)
        plt.cla()
        plt.plot(single_shot)
        plt.axhline(target, color="k")
        plt.title('Single shot measurement')
        plt.subplot(313)
        plt.cla()
        plt.plot(variance)
        plt.title('Variance')
        plt.pause(0.1)
        print(np.abs(np.max(variance))-np.abs(target))

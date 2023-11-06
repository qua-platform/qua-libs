from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


###################
# The QUA program #
###################
target = -0.001
bitshift_scale_factor = 9  ## scale_factor = 2**bitshift_scale_factor
gain_P = -2.0*1.85 /2/2
gain_I = -7
gain_D = -0.2

alpha = 0.1
N_shots = 1000

with program() as PID_prog:
    # adc_st = declare_stream(adc_trace=True)
    n = declare(int)
    single_shot = declare(fixed)
    single_shot_st = declare_stream()
    amplitude = declare(fixed, value=1.0)
    error = declare(fixed, value=0)
    error_st = declare_stream()
    amp_st = declare_stream()
    integrator_error = declare(fixed, value=0)
    integrator_error_st = declare_stream()
    old_error = declare(fixed, value=0.0)
    derivative_error = declare(fixed)
    total_error = declare(fixed)
    derivative_error_st = declare_stream()

    with for_(n, 0, n < N_shots, n + 1):
        # start noise only after 1/4 of the shots
        # The noise is a different element outputting to the same ports and adding a 100kHz or so noise signal
        with if_(n > N_shots / 4):
            play("cw", "noise")

        # play the AOM pulse with a controlled amplitude
        play("cw" * amp(amplitude), "AOM")
        # Measure and integrate the signal received by the photo-diode
        measure("readout", "photo-diode", None, integration.full("constant", single_shot, "out1"))

        # calculate the error
        assign(error, (target - single_shot) << bitshift_scale_factor)
        # calculate the integrator error with exponentially decreasing weights with coefficient alpha
        assign(integrator_error, (1 - alpha) * integrator_error + alpha * error)
        # calculate the derivative error
        assign(derivative_error, old_error - error)
        # stop the correction during 1/4 of shots just to show that it locks to a different setpoint quickly
        with if_((n < int(N_shots / 2)) | (n > int(0.75*N_shots))):
            assign(amplitude, amplitude + (gain_P * error + gain_I * integrator_error + gain_D * derivative_error))

        # save old error to be error
        assign(old_error, error)

        save(single_shot, single_shot_st)
        save(error, error_st)
        save(amplitude, amp_st)
        save(derivative_error, derivative_error_st)
        save(integrator_error, integrator_error_st)

    with stream_processing():
        single_shot_st.save_all("single_shot")
        error_st.save_all("error")
        amp_st.save_all("amplitude")
        integrator_error_st.save_all("integrator_error")
        derivative_error_st.save_all("derivative_error")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)

###########################
# Run or Simulate Program #
###########################

simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, PID_prog, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()
else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(PID_prog)
    #
    job.result_handles.wait_for_all_values()
    res = job.result_handles
    res.wait_for_all_values()
    # adc1 = res.get("adc1").fetch_all()
    error = res.get("error").fetch_all()["value"]
    derivative_error = res.get("derivative_error").fetch_all()["value"]
    single_shot = res.get("single_shot").fetch_all()["value"]
    amplitude = res.get("amplitude").fetch_all()["value"]
    integrator_error = res.get("integrator_error").fetch_all()["value"]

    # I calculated on the scope 0.38us between pulses
    Time = [x * 0.38 * 0 + x for x in range(N_shots)]
    # just checking the std after the lock
    # print(single_shot[N_shots:-1].std())

    # plotting stuff
    plt.figure()
    plt.subplot(311)
    plt.plot(Time, error, ".-")
    plt.title("Error signal")
    plt.xlabel("Time [μs]")
    plt.ylabel("Amplitude Error [arb. units]")

    plt.subplot(312)
    plt.plot(Time, single_shot)
    plt.axhline(target, color="k")
    plt.title('Single shot measurement')

    plt.subplot(313)
    plt.plot(Time, amplitude)
    plt.title("Applied amplitude")
    plt.tight_layout()


    plt.figure()
    plt.subplot(311)
    plt.plot(Time, error, ".-")
    plt.title("Intensity lock error")
    plt.xlabel("Time [μs]")
    plt.ylabel("Amplitude Error [arb. units]")

    plt.subplot(312)
    plt.plot(Time, integrator_error)
    plt.title('integrator error')
    plt.xlabel("Time [μs]")

    plt.subplot(313)
    plt.plot(Time, derivative_error)
    plt.title('derivative error')
    plt.xlabel("Time [μs]")
    plt.tight_layout()

"""
A simple sandbox to showcase different QUA functionalities during the installation.
"""

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
gain_P = -2.0*1.85 /2/2
gain_I = -7
gain_D = -0.2
# gain_I_bitshift=0
# integral_size=10

alpha = 0.1
N_shots = 1000

with program() as PID_prog:
    # adc_st = declare_stream(adc_trace=True)
    n = declare(int)
    integ = declare(fixed)
    integral_st = declare_stream()
    amplitude = declare(fixed, value=1.0)
    error = declare(fixed, value=0)
    error_st = declare_stream()
    amp_st = declare_stream()
    integrator_error = declare(fixed, value=0)
    integrator_error_st = declare_stream()
    old_error = declare(fixed, value=0.0)
    derivative_error = declare(fixed)
    total_error = declare(fixed)
    total_error_st = declare_stream()

    with for_(n, 0, n < N_shots, n + 1):
        # start noise only after 1/4 of the shots
        # The noise is a different element outputting to the same ports and adding a 100kHz or so noise signal
        # on the gaussians. since the pulse processors are phase coherent from shot to shot then the phase of the kHz signal
        # is preserved shot to shot and so we make slow, kHz noise, on a train of gaussians which are 100ns long
        with if_(n > N_shots / 4):
            play("cw", "noise")

        # play the gaussian pulse and measure the intensity on an envelope detector
        # we are playing DC from the OPX and modulating the IQ mixer like a vector modulator (i.e. modulating the LO directly to make 100ns gaussians)
        # the signal is acquired on an envelope detector at the RF port of the mixer and goes into the OPX
        play("cw" * amp(amplitude), "AOM")
        measure("readout", "photo-diode", None, integration.full("constant", integ, "out1"))

        # calculate the error
        assign(error, (target - integ) << 9)
        # calculate the integrator error with exponentially decreasing weights with coefficient alpha
        assign(integrator_error, (1 - alpha) * integrator_error + alpha * error)
        # calculate the derivative error
        assign(derivative_error, old_error - error)
        # start the lock at first then stop to show the noise without lock, then lock on the noise. then keep lock and remove noise to compare the noise suppression to the no noise case and see if we are detection limited

        # apply correction during the first 1/4 of shots just to show that it locks to a different setpoint quickly
        # lower the gain_P to see that the lock becomes softer
        with if_((n < int(N_shots / 2)) | (n > int(0.75*N_shots))):
            assign(amplitude, amplitude + (gain_P * error + gain_I * integrator_error + gain_D * derivative_error))
        # stop the lock while the noise is introduced for another 1/4 of shots to show the noise without lock
        # start the lock again after 1/2 the shots and see the performance
        # with if_(n > int(N_shots / 2)):
        #     assign(amplitude, amplitude + (gain_P * error + gain_I * integrator_error + gain_D * derivative_error))

        # save old error to be error
        assign(old_error, error)

        save(integ, integral_st)
        save(error, error_st)
        save(amplitude, amp_st)
        save(total_error, total_error_st)
        save(integrator_error, integrator_error_st)

    with stream_processing():
        integral_st.save_all("integral")
        error_st.save_all("error")
        amp_st.save_all("amplitude")
        integrator_error_st.save_all("integrator_error")
        total_error_st.save_all("total_error")

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
    total_error = res.get("total_error").fetch_all()["value"]
    integral = res.get("integral").fetch_all()["value"]
    amplitude = res.get("amplitude").fetch_all()["value"]
    integrator_error = res.get("integrator_error").fetch_all()["value"]

    # I calculated on the scope 0.38us between pulses
    Time = [x * 0.38 * 0 + x for x in range(N_shots)]
    # just checking the std after the lock
    # print(integral[N_shots:-1].std())

    # plotting stuff

    plt.figure()
    plt.subplot(221)
    plt.plot(Time, error, ".-")
    plt.title("Intensity lock error")
    plt.xlabel("Time [μs]")
    plt.ylabel("Amplitude Error [arb. units]")

    plt.subplot(222)
    plt.plot(Time, integral)
    plt.axhline(target, color="k")
    plt.title('Intensity')

    plt.subplot(223)
    plt.plot(Time, amplitude)
    plt.title("Applied amplitude")
    plt.tight_layout()
    # plt.figure(figsize=(20,16))
    plt.subplot(224)
    plt.plot(Time, integrator_error)
    plt.title('integrator error')
    # plt.figure(figsize=(20,16))
    # plt.plot(Time, total_error)
    # plt.title('total error')

    # plt.figure()
    # plt.plot(adc1)

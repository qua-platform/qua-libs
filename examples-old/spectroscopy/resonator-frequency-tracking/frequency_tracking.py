from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from qm.qua import *
from config import config
import matplotlib.pyplot as plt


def measure_response_magnitude(qe, pulse, result):
    # wait so to let the resonator to return to rest
    wait(100, qe)

    # temporary variables to dump the measurement demodulation integral into
    I = declare(fixed)
    Q = declare(fixed)
    measure(pulse, qe, None, demod.full("integW1", I), demod.full("integW2", Q))

    # save the magnitude into the result QUA-variable.
    # keep in mind that although we used an I-pulse the response might have Q-pulse part
    assign(result, I * I + Q * Q)


def track_frequency(qe, pulse, current_w, w_step):
    # temporary variables to dump the magnitudes into
    res_w_smaller = declare(fixed)
    res_w = declare(fixed)
    res_w_bigger = declare(fixed)

    # measure the response magnitude at the current frequency
    update_frequency(qe, current_w)
    # Note: the amplitude spoofing won't be needed in a real experiment
    measure_response_magnitude(qe, pulse * amp(LORENTZIAN[steps_position]), res_w)

    # measure the response magnitude at a lower frequency by 1 step
    update_frequency(qe, current_w - w_step)
    # Note: the amplitude spoofing won't be needed in a real experiment
    measure_response_magnitude(qe, pulse * amp(LORENTZIAN[steps_position - 1]), res_w_smaller)

    # measure the response magnitude at a higher frequency by 1 step
    update_frequency(qe, current_w + w_step)
    # Note: the amplitude spoofing won't be needed in a real experiment
    measure_response_magnitude(qe, pulse * amp(LORENTZIAN[steps_position + 1]), res_w_bigger)

    # figure out which frequency at the lowest response magnitude, and update the current frequency accordingly
    best_w = Util.cond(
        res_w_smaller < res_w,
        current_w - w_step,
        Util.cond(res_w <= res_w_bigger, current_w, current_w + w_step),
    )
    assign(current_w, best_w)
    update_frequency(qe, current_w)

    # Note: this line won't be needed in a real experiment because spoofing won't be needed
    # update the current lorentzian index
    assign(
        steps_position,
        Util.cond(
            res_w_smaller < res_w,
            steps_position - 1,
            Util.cond(res_w <= res_w_bigger, steps_position, steps_position + 1),
        ),
    )


qmm = QuantumMachinesManager(host="127.0.0.1")

with program() as w_if_tracking:
    qe = "qe1"
    pulse = "measurement"
    current_w = declare(int, value=int(10e6))
    calibration_step = declare(int, value=int(1e4))

    # Note: this line won't be needed in a real experiment because spoofing won't be needed
    # used for lorenzian response spoofing, this is the way to pre-process and create the Lorentz function as a QUA array
    LORENTZIAN = declare(fixed, value=[1 - 0.5 / (1 + 0.1 * (n - 50) ** 2) for n in range(100)])
    # this spoof variable is the index of the current spoofing
    steps_position = declare(int, value=50)

    loop_iteration_counter = declare(int)
    with for_(
        loop_iteration_counter,
        0,
        loop_iteration_counter < 100,
        loop_iteration_counter + 1,
    ):

        # Note: this line won't be needed in a real experiment because spoofing won't be needed
        # let's assume the real intermediate frequency decreased by 1 step
        assign(steps_position, steps_position + 1)

        # This example function will fix the quantum element intermediate frequency using a basic calibration algorithm
        track_frequency(qe, pulse, current_w, calibration_step)

        ################################
        # rest of experiment code here #
        ################################

        # for exmaple, let's have an experiment that measures the magnitude of response
        response = declare(fixed)
        # Note: the amplitude spoofing won't be needed in a real experiment
        measure_response_magnitude(qe, pulse * amp(LORENTZIAN[steps_position]), response)
        save(response, "response")
        save(current_w, "w")

job = qmm.simulate(
    config,
    w_if_tracking,
    SimulationConfig(
        # duration of simulation in units of 4ns
        duration=30000,
        # Note: the Loopback won't be needed in a real experiment, because the element will output back to the OPX
        simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)]),
    ),
)

response_handle = job.result_handles.get("response")
response_values = response_handle.fetch_all()
plt.figure()
plt.title("Response as a function of time")
plt.plot(
    [time for (magnitude, time) in response_values],
    [magnitude for (magnitude, time) in response_values],
    "--bo",
)
plt.show()

w_handle = job.result_handles.get("w")
w_values = w_handle.fetch_all()
plt.figure()
plt.title("Frequency as a function of time")
plt.plot([time for (w, time) in w_values], [w for (w, time) in w_values], "--bo")
plt.show()

from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt

################################
# Open quantum machine manager #
################################

qmm = QuantumMachinesManager()

########################
# Open quantum machine #
########################

qm = qmm.open_qm(config)

###################
# The QUA program #
###################


n_avg = 1000  # number of averages

# cooldown_time = 50000 // 4  # qubit decay time
cooldown_time = 4  # qubit decay time

with program() as mixer_cal:

    # Declare QUA variables
    ###################
    n = declare(int)  # variable for average loop
    n_st = declare_stream()  # stream for 'n'
    c = declare(int)  # variable for switch case
    I = declare(fixed)  # demodulated and integrated signal
    Q = declare(fixed)  # demodulated and integrated signal
    I_st = declare_stream()  # stream for I
    Q_st = declare_stream()  # stream for Q

    # Pulse sequence
    ################
    with for_(n, 0, n < n_avg, n + 1):

        with for_(c, 0, c <= 2, c + 1):
            with switch_(c):
                with case_(0):  # projection along X
                    wait(cooldown_time, "qubit", "resonator")
                    play("-Y/2", "qubit")
                    align("qubit", "resonator")
                    measure(
                        "readout",
                        "resonator",
                        None,
                        dual_demod.full("cos", "out1", "sin", "out2", I),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                    )
                    save(I, I_st)
                    save(Q, Q_st)
                with case_(1):  # projection along Y
                    wait(cooldown_time, "qubit")
                    play("X/2", "qubit")
                    align("qubit", "resonator")
                    measure(
                        "readout",
                        "resonator",
                        None,
                        dual_demod.full("cos", "out1", "sin", "out2", I),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                    )
                    save(I, I_st)
                    save(Q, Q_st)
                with case_(2):  # projection along Z
                    wait(cooldown_time, "qubit")
                    measure(
                        "readout",
                        "resonator",
                        None,
                        dual_demod.full("cos", "out1", "sin", "out2", I),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                    )
                    save(I, I_st)
                    save(Q, Q_st)
        save(n, n_st)

    # Stream processing
    ###################
    with stream_processing():
        n_st.save("iteration")
        I_st.buffer(3).average().save("I")
        Q_st.buffer(3).average().save("Q")

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    # simulation properties
    simulate_config = SimulationConfig(
        duration=1000,
        simulation_interface=LoopbackInterface(([("con1", 1, "con1", 1)])),
    )
    job = qmm.simulate(config, mixer_cal, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    job = qm.execute(mixer_cal)  # execute QUA program

    res_handle = job.result_handles

    I_handle = res_handle.get("I")
    I_handle.wait_for_values(1)
    Q_handle = res_handle.get("Q")
    Q_handle.wait_for_values(1)
    iteration_handle = res_handle.get("iteration")
    iteration_handle.wait_for_values(1)

    while res_handle.is_processing():
        try:
            I = I_handle.fetch_all()
            Q = Q_handle.fetch_all()
            iteration = iteration_handle.fetch_all()
            print(iteration)
            print(I, Q)

        except Exception as e:
            pass

    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    print(I, Q)

from aws_utils import *
import matplotlib.pyplot as plt


def XXXX_stabilizer_scheme():
    # prepare ancilla in |+>
    initialize_state("beta_1")

    align(*elements_list)

    # Time step 2
    CNOT("beta_2", "alpha_2")
    stabilize("beta_2")
    stabilize("gamma_1")
    # CNOT("gamma_1", "delta_1")

    align(*elements_list)

    # Time step 3
    CNOT("beta_1", "gamma_1")
    stabilize("beta_2")
    stabilize("alpha_2")
    # CNOT("alpha_1", "delta_1")

    align(*elements_list)

    # Time step 4
    CNOT("beta_2", "gamma_2")
    stabilize("beta_1")
    stabilize("alpha_1")
    # CNOT("alpha_2", "delta_2")

    align(*elements_list)

    #  Time step 5
    CNOT("beta_1", "alpha_1")
    stabilize("beta_1")
    stabilize("gamma_2")
    # CNOT("gamma_2", "delta_2")

    align(*elements_list)

    # non-adiabatic deflation
    play("deflation", "beta_1_buffer")
    play("Pump_Op", "beta_1_ATS", duration=50)

    # SWAP
    update_frequency("beta_1_ATS", omega_p)
    update_frequency("beta_1_buffer", get_IF("beta_1_buffer") - delta_SWAP)
    align("beta_1_buffer", "beta_1_ATS")
    play("swap", "beta_1_ATS", duration=50)
    play("drive", "beta_1_buffer", duration=50)
    update_frequency("beta_1_ATS", get_IF("beta_1_ATS"))
    update_frequency("beta_1_buffer", get_IF("beta_1_buffer"))
    align(*elements_list)


def readout():
    I = declare(fixed)
    se = declare(bool)
    n = declare(int)
    pi2_phase = declare(fixed, value=-1)
    counter = declare(int, value=0)
    MV = declare(bool)
    MV_st = declare_stream()

    # Parity
    wait(4, "t0")
    with for_(n, 0, n < n_parity, n + 1):
        play("X90", "t0")  # unconditional
        wait(t_parity, "t0")
        frame_rotation(np.pi, "t0")
        play("X90" * amp(pi2_phase), "t0")
        align("t0", "rr0")
        measure(
            "Readout_Op",
            "rr0",
            None,
            dual_demod.full("integW_cos", "out1", "integW_sin", "out2", I),
        )

        assign(se, I > th)
        assign(counter, counter + Cast.to_int(se))
        assign(pi2_phase, Util.cond(se, -pi2_phase, pi2_phase))

        # active reset transmon
        play("X", "t0", condition=se)  # se == True => Excited
    # Majority voting
    assign(MV, counter > int(n_parity / 2))
    save(MV, MV_st)


repetitions = 1

with program() as rep_code:
    i = declare(int)

    XXXX_stabilizer_scheme()
    # align()

    with for_(i, 0, i < repetitions, i + 1):
        # error tracking
        # with qrun_():
        readout()
        XXXX_stabilizer_scheme()

qmm = QuantumMachinesManager()

job = qmm.simulate(
    config, rep_code, simulate=SimulationConfig(4500), flags=["auto-element-thread"]
)
samples = job.get_simulated_samples()
plt.figure(1)
plt.plot(samples.con1.analog["1"])
plt.plot(samples.con1.analog["2"])
plt.plot(samples.con1.analog["3"] + 1)
plt.plot(samples.con1.analog["4"] + 1)
plt.plot(samples.con1.analog["5"] + 2)
plt.plot(samples.con1.analog["6"] + 2)
plt.plot(samples.con1.analog["7"] + 3)
plt.plot(samples.con1.analog["8"] + 3)
plt.plot(samples.con1.analog["9"] + 4)
plt.plot(samples.con1.analog["10"] + 4)
plt.plot(samples.con2.analog["1"] + 5)
plt.plot(samples.con2.analog["2"] + 5)
plt.xlabel("Time [ns]")
plt.yticks(
    [0, 1, 2, 3, 4, 5],
    ["Transmon", "Readout", "ATS 1", "buffer 1", "ATS 2", "buffer 2"],
    rotation="vertical",
    va="center",
)

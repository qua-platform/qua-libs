from qm import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *

t = int(1e3)  # Sensing duration


def init_nuclear_spin(qubit, target_state):
    SSR(qubit, state, 1000)
    with while_(state != target_state):
        save(state, "a")
        if qubit == "N":  # Three level nuclear
            # This plays the pi pulse a the specific transition between the current state and the target state
            with if_(state == 1):
                play(f"pi_1_{target_state}", qubit)
            with if_(state == 0):
                play(f"pi_0_{target_state}", qubit)
            with if_(state == -1):
                play(f"pi_-1_{target_state}", qubit)
        else:  # Two level nuclear
            play("pi", qubit)

        SSR(qubit, state, 1000)


def QFT():
    play("Hadamard", "C90")  # 1
    align("NV", "N", "C414", "C90")  # 2

    play("pi_over_two_C90_C414", "NV")  # 3
    align("NV", "N", "C414", "C90")  # 4

    play("pi_over_six_C90_N1", "NV")  # 5 - Can be merged with 6
    play("two_pi_over_six_C90_N2", "NV")  # 6 - Can be merged with 5
    play("Hadamard", "C414")  # 7
    align("NV", "N", "C414", "C90")  # 8

    play("pi_over_three_C414_N1", "NV")  # 9 - Can be merged with 10
    play("two_pi_over_three_C414_N2", "NV")  # 10 - Can be merged with 9
    align("NV", "N", "C414", "C90")  # 11

    play("Chrestenson", "N")  # 12


def iQFT():
    play("Chrestenson", "N")  # 12
    align("NV", "N", "C414", "C90")  # 11

    play("two_pi_over_three_C414_N2", "NV")  # 10 - Can be merged with 9
    play("pi_over_three_C414_N1", "NV")  # 9 - Can be merged with 10
    align("NV", "N", "C414", "C90")  # 8

    play("Hadamard", "C414")  # 7
    play("two_pi_over_six_C90_N2", "NV")  # 6 - Can be merged with 5
    play("pi_over_six_C90_N1", "NV")  # 5 - Can be merged with 6
    align("NV", "N", "C414", "C90")  # 4

    play("pi_over_two_C90_C414", "NV")  # 3
    align("NV", "N", "C414", "C90")  # 2

    play("Hadamard", "C90")  # 1


def SSR(qubit, state, N):
    """Determine the state of a nuclear spin using N repetitions"""
    with for_(ssr_i, 0, ssr_i < 3, ssr_i + 1):
        assign(ssr_count[ssr_i], 0)

    # run N repetitions
    with for_(ssr_i, 0, ssr_i < N, ssr_i + 1):
        # perform pi pulse and optical readout
        play(f"pi_{qubit}+", "NV")
        measure(
            "readout",
            "NV",
            None,
            time_tagging.raw(ssr_res_vec, 300, targetLen=ssr_res_length),
        )
        assign(ssr_count[2], ssr_res_length)

        play(f"pi_{qubit}-", "NV")
        measure(
            "readout",
            "NV",
            None,
            time_tagging.raw(ssr_res_vec, 300, targetLen=ssr_res_length),
        )
        assign(ssr_count[0], ssr_res_length)

        if qubit == "N":  # Three level nuclear
            play(f"pi_{qubit}0", "NV")
            measure(
                "readout",
                "NV",
                None,
                time_tagging.raw(ssr_res_vec, 300, targetLen=ssr_res_length),
            )
            assign(ssr_count[1], ssr_res_length)

    # compare photon count and save result in variable "state"
    if qubit == "N":  # Three level nuclear
        assign(state, Math.argmin(ssr_count) - 1)
    else:  # Two level nuclear
        with if_(ssr_count[2] > ssr_count[0]):
            assign(state, 1)
        with else_():
            assign(state, -1)


def phase_acquisition(qubit, time):
    if qubit == "N":
        play("pi_over_two_N-", "NV")
        wait(time, "NV")
        play("pi_over_two_N0", "NV")
        wait(time, "NV")
        play("pi_over_two_N+", "NV")

    elif qubit == "C414":
        play("pi_over_two_C414-", "NV")
        wait(time, "NV")
        play("pi_over_two_C414+", "NV")

    else:
        play("pi_over_two_C90-", "NV")
        wait(time, "NV")
        play("pi_over_two_C90+", "NV")


with program() as qft_sensing:
    a = declare(int)  # Averages
    state = declare(int)  # Current nuclear spin state
    target_stage = declare(int)  # Nuclear spin desired state
    ssr_i = declare(int)  # Used by SSR for loop index
    ssr_count = declare(int, size=3)  # Used by SSR for count, up to 3 nuclear levels
    ssr_res_length = declare(int)
    ssr_res_vec = declare(int, size=5)

    with for_(a, 0, a < 1e6, a + 1):

        # Init
        init_nuclear_spin("N", -1)
        init_nuclear_spin("C414", -1)
        init_nuclear_spin("C90", -1)

        align("NV", "N", "C414", "C90")
        play("laser", "NV")
        play("pi", "NV")  # To be in the 1 state.

        # QFT
        QFT()

        align("NV", "N", "C414", "C90")

        # Sensing
        phase_acquisition("N", t)
        phase_acquisition("C414", 2 * t)
        phase_acquisition("C90", 4 * t)

        # IQFT
        iQFT()

        align("NV", "N", "C414", "C90")

        # Single Shot Readout
        SSR("N", state, 1000)
        save(state, "N_state")
        SSR("C414", state, 1000)
        save(state, "C414_state")
        SSR("C90", state, 1000)
        save(state, "C90_state")

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

job = qm.simulate(qft_sensing, simulate=SimulationConfig(duration=20000))
job.get_simulated_samples()

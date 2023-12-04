"""
        READOUT SEARCH
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from qualang_tools.addons.variables import assign_variables_to_element
from qualang_tools.bakery import baking

###################
# The QUA program #
###################

n_avg = 100  # Number of averaging loops
drive_frequency = 10 * u.GHz
N = 100
bit_shit_cte = 7
n_avg_ro = 2**bit_shit_cte
Coulomb_pk_to_pk = 0.2
qubit_IFs = np.arange(-50 * u.MHz, 50 * u.MHz, 0.1 * u.MHz)
burst_durations = np.arange(16, 100, 4)

# Bake the Rabi pulses
pi_list = []
for t in burst_durations:  # Create the different baked sequences
    t = int(t)
    with baking(config, padding_method="none") as b:  # don't use padding to assure error if timing is incorrect
        if t == 0:
            wf_I = [0.0] * 16
            wf_Q = [0.0] * 16  # Otherwise the baked pulse will be empty
        else:
            wf_I = [pi_amp_left] * t
            wf_Q = [0.0] * t  # The baked waverforms (only the I quadrature)

        # Add the baked operation to the config
        b.add_op("pi_baked", "qubit", [wf_I, wf_Q])

        gap_to_adjust = 20  # gap to remove due to realtime calculations
        # Time to wait after playing the pulse in order to remain sync with the Coulomb pulse
        wait_time = (2 * bias_length) - (t + gap_to_adjust)
        # zero-pad the baked waveform to match the multiple-of-4ns requirement
        remainder = 4 - (t + wait_time) % 4

        # Baked sequence
        b.wait(16, "qubit")  # Wait before playing the qubit pulse (can be removed or adjusted)
        b.play("pi_baked", "qubit")  # Play the qubit pulse
        b.wait(wait_time + remainder, "qubit")  # Wait after the pulse in order to remain sync with the Coulomb pulse

    # Append the baking object in the list to call it from the QUA program
    pi_list.append(b)

with program() as prog:
    n = declare(int)  # QUA integer used as an index for the outer averaging loop
    n_ro = declare(int)  # QUA integer used as an index for the inner averaging loop
    counter = declare(int)  # QUA integer used as an index for the Coulomb drive
    counter2 = declare(int)  # QUA integer used as an index for the Rabi drive
    f = declare(int)  # QUA integer for sweeping the qubit drive frequency
    baking_index = declare(int)  # QUA integer for sweeping the qubit pulse duration
    I_on = declare(fixed)  # QUA fixed used to store the outcome of the readout when a qubit pulse is played
    I_off = declare(fixed)  # QUA fixed used to store the outcome of the readout without qubit pulse
    I_on_avg = declare(fixed)  # QUA fixed used to store the outcome of the readout when a qubit pulse is played
    I_off_avg = declare(fixed)  # QUA fixed used to store the outcome of the readout without qubit pulse
    I_on_st = declare_stream()  # Stream for I_on
    I_off_st = declare_stream()  # Stream for I_off
    n_st = declare_stream()  # STream for n --> progress counter for live-plotting
    # Ensure that the result variables are assign to the pulse processor used for readout
    assign_variables_to_element("TIA", I_on, I_off)
    assign_variables_to_element("TIA", I_on_avg, I_off_avg)
    # Set the qubit drive frequency

    with for_(n, 0, n < n_avg, n + 1):  # The outer averaging loop
        with for_(*from_array(f, qubit_IFs)):  # The outer averaging loop
            update_frequency("qubit", f)
            with for_(baking_index, 0, baking_index < len(burst_durations), baking_index + 1):
                assign(I_on_avg, 0)
                assign(I_off_avg, 0)
                # Play the Coulomb pulse continuously for a time given by coulomb_drive_length
                with for_(counter, 0, counter < N, counter + 1):
                    # The Coulomb pulse
                    play("bias" * amp(Coulomb_pk_to_pk), "gate_1")
                    play("bias" * amp(-Coulomb_pk_to_pk), "gate_1")

                # Play the qubit sequence in a separated loop to avoid gaps in the Coulomb drive
                with for_(counter2, 0, counter2 < N, counter2 + 1):
                    # switch case to select the baked waveform corresponding to the burst duration
                    with switch_(baking_index, unsafe=True):
                        for ii in range(len(burst_durations)):
                            with case_(ii):
                                pi_list[ii].run()

                with for_(n_ro, 0, n_ro < n_avg_ro, n_ro + 1):  # The inner averaging loop for I_on
                    measure("readout", "TIA", None, integration.full("cos", I_on, "out1"))
                    assign(I_on_avg, (I_on >> bit_shit_cte) + I_on_avg)
                save(I_on_avg, I_on_st)

                align()
                # Play the Coulomb pulse continuously for a time given by coulomb_drive_length without qubit drive
                with for_(counter, 0, counter < N, counter + 1):
                    # The Coulomb pulse
                    play("bias" * amp(Coulomb_pk_to_pk), "gate_1")
                    play("bias" * amp(-Coulomb_pk_to_pk), "gate_1")

                with for_(n_ro, 0, n_ro < n_avg_ro, n_ro + 1):  # The inner averaging loop for I_off
                    measure("readout", "TIA", None, integration.full("cos", I_off, "out1"))
                    assign(I_off_avg, (I_off >> bit_shit_cte) + I_off_avg)
                save(I_off_avg, I_off_st)

        save(n, n_st)

    with stream_processing():
        # Average and stream I_on, I_off, the difference and their sum
        I_on_st.buffer(len(burst_durations)).buffer(len(qubit_IFs)).average().save("I_on")
        I_off_st.buffer(len(burst_durations)).buffer(len(qubit_IFs)).average().save("I_off")
        (I_on_st - I_off_st).buffer(len(burst_durations)).buffer(len(qubit_IFs)).average().save("I_diff")
        (I_on_st + I_off_st).buffer(len(burst_durations)).buffer(len(qubit_IFs)).average().save("I_sum")

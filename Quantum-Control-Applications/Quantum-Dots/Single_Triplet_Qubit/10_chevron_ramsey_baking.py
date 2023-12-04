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

# Baking section
pi_list = []
pi_reference_list = []
for t in idle_times:  # Create the different baked sequences
    t = int(t)
    # Bake the Ramsey sequence
    with baking(
        opx_instrument.config, padding_method="none"
    ) as b:  # don't use padding to assure error if timing is incorrect
        # generate the baked pi_half pulses
        wf_I_pi_half = [pi_amp] * pi_half_duration
        wf_Q_pi_half = [0.0] * pi_half_duration
        gap_to_adjust = 20  # gap to remove due to realtime calculations
        # Time to wait after playing the pulse in order to remain sync with the Coulomb pulse
        wait_time = (2 * bias_length) - (t + 2 * pi_half_duration + gap_to_adjust)
        # zero-pad the baked waveform to match the multiple-of-4ns requirement
        remainder = 4 - (t + 2 * pi_half_duration + wait_time) % 4

        # Add the baked pi_half operation to the config
        b.add_op("pi_half_baked", "qubit", [wf_I_pi_half, wf_Q_pi_half])

        # Baked sequence
        b.wait(16, "qubit")  # Wait before playing the qubit pulse (can be removed or adjusted)
        b.play("pi_half_baked", "qubit")  # Play the 1st pi half pulse
        b.wait(t, "qubit")  # Wait the idle time
        b.frame_rotation_2pi(
            artificial_oscillation_frequency * t * 1e-9, "qubit"
        )  # Dephase the 2nd pi half for virtual Z-rotation
        b.play("pi_half_baked", "qubit")  # Play the pi half pulse
        b.reset_frame("qubit")  # Reset the frame to avoid floating point error accumulation
        b.wait(wait_time + remainder, "qubit")  # Wait after the pulse in order to remain sync with the Coulomb pulse
    # Bake the reference sequence
    with baking(
        opx_instrument.config, padding_method="none"
    ) as b_ref:  # don't use padding to assure error if timing is incorrect
        # generate the baked reference pulses
        wf_I_pi_half = [pi_amp] * pi_half_duration
        wf_Q_pi_half = [0.0] * pi_half_duration
        # Time to wait after playing the pulse in order to remain sync with the Coulomb pulse
        wait_time = (2 * bias_length) - (t + 2 * pi_half_duration + gap_to_adjust)
        # zero-pad the baked waveform to match the multiple-of-4ns requirement
        remainder = 4 - (t + 2 * pi_half_duration + wait_time) % 4

        # Add the baked pi_half operations to the config
        b_ref.add_op("pi_half_baked", "qubit", [wf_I_pi_half, wf_Q_pi_half])

        # Baked sequence
        b_ref.wait(16, "qubit")  # Wait before playing the qubit pulse (can be removed or adjusted)
        b_ref.play("pi_half_baked", "qubit")  # Play the 1st pi half pulse
        b_ref.wait(t, "qubit")  # Wait the idle time
        b_ref.frame_rotation_2pi(artificial_oscillation_frequency * t * 1e-9, "qubit")
        b_ref.play("pi_half_baked", "qubit", amp=-1)  # Play the 2nd reversed pi half pulse
        b_ref.reset_frame("qubit")  # Reset the frame to avoid floating point error accumulation
        b_ref.wait(
            wait_time + remainder, "qubit"
        )  # Wait after the pulse in order to remain sync with the Coulomb pulse

    # Append the baking object in the list to call it from the QUA program
    pi_list.append(b)
    pi_reference_list.append(b_ref)

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
            # with for_(*from_array(baking_index, np.arange(0, len(burst_durations), 1))):  # The outer averaging loop
            with for_(baking_index, 0, baking_index < len(idle_times), baking_index + 1):
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
                        for ii in range(len(idle_times)):
                            with case_(ii):
                                pi_list[ii].run()

                # Wait for the IV converter to reach its steady state and measure for a duration given by total_integration_time
                wait(IV_buffer_len * u.ns, "TIA")
                with for_(n_ro, 0, n_ro < n_avg_ro, n_ro + 1):  # The inner averaging loop for I_on
                    measure("readout", "TIA", None, integration.full("cos", I_on, "out1"))
                    assign(I_on_avg, I_on_avg + (I_on >> n_avg_ro_pow2))
                save(I_on_avg, I_on_st)

                align()
                # Play the Coulomb pulse continuously for a time given by coulomb_drive_length without qubit drive
                with for_(counter, 0, counter < N, counter + 1):
                    # The Coulomb pulse
                    play("bias" * amp(Coulomb_pk_to_pk), "gate_1")
                    play("bias" * amp(-Coulomb_pk_to_pk), "gate_1")

                # Play the reference sequence
                with for_(counter2, 0, counter2 < N, counter2 + 1):
                    with switch_(baking_index, unsafe=True):
                        for ii in range(len(idle_times)):
                            with case_(ii):
                                pi_reference_list[ii].run()

                # Wait for the IV converter to reach its steady state and measure for a duration given by total_integration_time
                wait(IV_buffer_len * u.ns, "TIA")
                with for_(n_ro, 0, n_ro < n_avg_ro, n_ro + 1):  # The inner averaging loop for I_off
                    measure("readout", "TIA", None, integration.full("cos", I_off, "out1"))
                    assign(I_off_avg, I_off_avg + (I_off >> n_avg_ro_pow2))
                save(I_off_avg, I_off_st)
        save(n, n_st)

    with stream_processing():
        # Average and stream I_on, I_off, the difference and their sum
        if live_plotting:
            I_on_st.buffer(len(idle_times)).buffer(len(qubit_IFs)).average().save_all("I_on")
            I_off_st.buffer(len(idle_times)).buffer(len(qubit_IFs)).average().save_all("I_off")
            (I_on_st - I_off_st).buffer(len(idle_times)).buffer(len(qubit_IFs)).average().save_all("I_diff")
            (I_on_st + I_off_st).buffer(len(idle_times)).buffer(len(qubit_IFs)).average().save_all("I_sum")
            n_st.save("iteration")
        else:
            I_on_st.buffer(len(idle_times)).buffer(len(qubit_IFs)).buffer(n_avg).map(FUNCTIONS.average()).save_all(
                "I_on"
            )
            I_off_st.buffer(len(idle_times)).buffer(len(qubit_IFs)).buffer(n_avg).map(FUNCTIONS.average()).save_all(
                "I_off"
            )
            (I_on_st - I_off_st).buffer(len(idle_times)).buffer(len(qubit_IFs)).buffer(n_avg).map(
                FUNCTIONS.average()
            ).save_all("I_diff")
            (I_on_st + I_off_st).buffer(len(idle_times)).buffer(len(qubit_IFs)).buffer(n_avg).map(
                FUNCTIONS.average()
            ).save_all("I_sum")

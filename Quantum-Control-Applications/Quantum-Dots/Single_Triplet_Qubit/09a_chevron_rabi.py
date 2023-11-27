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

###################
# The QUA program #
###################

n_avg = 100  # Number of averaging loops
drive_frequency = 10 * u.GHz
N = 100
bit_shit_cte = 7
n_avg_ro = 2 ** bit_shit_cte
Coulomb_pk_to_pk = 0.2
qubit_IFs = np.arange(-50 * u.MHz, 50 * u.MHz, 0.1 * u.MHz)
burst_durations = np.arange(16, 100, 4)

with program() as prog:
    n = declare(int)  # QUA integer used as an index for the outer averaging loop
    n_ro = declare(int)  # QUA integer used as an index for the inner averaging loop
    counter = declare(int)  # QUA integer used as an index for the Coulomb drive
    counter2 = declare(int)  # QUA integer used as an index for the Rabi drive
    f = declare(int)  # QUA integer for sweeping the qubit drive frequency
    t_burst = declare(int)  # QUA integer for sweeping the qubit pulse duration
    I_on = declare(fixed)  # QUA fixed used to store the outcome of the readout when a qubit pulse is played
    I_off = declare(fixed)  # QUA fixed used to store the outcome of the readout without qubit pulse
    I_on_avg = declare(fixed)  # QUA fixed used to store the outcome of the readout when a qubit pulse is played
    I_off_avg = declare(fixed)  # QUA fixed used to store the outcome of the readout without qubit pulse
    I_on_st = declare_stream()  # Stream for I_on
    I_off_st = declare_stream()  # Stream for I_off
    # Ensure that the result variables are assign to the pulse processor used for readout
    assign_variables_to_element("TIA", I_on, I_off)

    with for_(n, 0, n < n_avg, n + 1):  # The outer averaging loop

        with for_(*from_array(f, qubit_IFs)):  # The outer averaging loop

            # Set the qubit drive frequency
            update_frequency("qubit", f)

            with for_(*from_array(t_burst, burst_durations // 4)):  # The outer averaging loop
                # Play the Coulomb pulse continuously for a time given by coulomb_drive_length
                with for_(counter, 0, counter < N, counter + 1):
                    # The Coulomb pulse
                    play("bias" * amp(Coulomb_pk_to_pk), "gate_1")
                    play("bias" * amp(-Coulomb_pk_to_pk), "gate_1")

                # Play the qubit sequence in a separated loop to avoid gaps in the Coulomb drive
                with for_(counter2, 0, counter2 < N, counter2 + 1):
                    wait(16 * u.ns, "qubit")  # Wait before driving the qubit
                    play("cw", "qubit", duration=t_burst)  # Rabi pulse
                    # Wait to always play at the same point of the Coulomb pulse
                    # Because the for_ loop and the real-time pulse streching takes some time, need to manually adjust the gap (19ns here)
                    # To precisely adjust the timing, please use the simulator
                    wait((2 * bias_length) // 4 - (t_burst + 19), "qubit")

                # # Wait for the IV converter to reach its steady state and measure for a duration given by total_integration_time
                # wait(IV_buffer_len * u.ns, "TIA")

                with for_(n_ro, 0, n_ro < n_avg_ro, n_ro + 1):  # The inner averaging loop for I_on
                    measure('readout', 'charge_sensor_DC', None, integration.full('cos', I_on, 'out1'))
                    assign(I_on_avg, (I_on >> bit_shit_cte) + I_on_avg)
                save(I_on_avg, I_on_st)

                align()
                # Play the Coulomb pulse continuously for a time given by coulomb_drive_length without qubit drive
                with for_(counter, 0, counter < N, counter + 1):
                    # The Coulomb pulse
                    play("bias" * amp(Coulomb_pk_to_pk), "gate_1")
                    play("bias" * amp(-Coulomb_pk_to_pk), "gate_1")

                # # Wait for the IV converter to reach its steady state and measure for a duration given by total_integration_time
                # wait(IV_buffer_len * u.ns, "TIA")
                with for_(n_ro, 0, n_ro < n_avg_ro, n_ro + 1):  # The inner averaging loop for I_off
                    measure('readout', 'charge_sensor_DC', None, integration.full('cos', I_off, 'out1'))
                    assign(I_off_avg, (I_off >> bit_shit_cte) + I_off_avg)
                save(I_off_avg, I_off_st)

    with stream_processing():
        # Average and stream I_on, I_off, the difference and their sum
        I_on_st.buffer(len(burst_durations)).buffer(len(qubit_IFs)).average().save("I_on")
        I_off_st.buffer(len(burst_durations)).buffer(len(qubit_IFs)).average().save("I_off")
        (I_on_st - I_off_st).buffer(len(burst_durations)).buffer(len(qubit_IFs)).average().save("I_diff")
        (I_on_st + I_off_st).buffer(len(burst_durations)).buffer(len(qubit_IFs)).average().save("I_sum")
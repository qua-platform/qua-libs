
from quam_libs.macros import qua_declaration

from quam_libs.experiments.qubit_spectroscopy.node import get_optional_pulse_duration

from qualang_tools.loops import from_array

import numpy as np
from qm.qua import *


def define_program(node, qubits, machine, span, step, dfs, qubit_pulse_duration):
        
    # {QUA_program}
    # Qubit detuning sweep with respect to their resonance frequencies
    span = node.parameters.frequency_span_in_mhz * node.u.MHz
    step = node.parameters.frequency_step_in_mhz * node.u.MHz
    dfs = np.arange(-span // 2, +span // 2, step, dtype=np.int32)
    # Get the optional parameters
    qubit_pulse_duration = get_optional_pulse_duration(qubits, node.parameters)
    
    num_qubits = len(qubits)

    with program() as qua_prog:
        # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
        df = declare(int)  # QUA variable for the qubit frequency

        for i, qubit in enumerate(qubits):
            # Bring the active qubits to the desired frequency point
            machine.set_all_fluxes(flux_point=node.parameters.flux_point_joint_or_independent, target=qubit)

            with for_(n, 0, n < node.parameters.num_averages, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    # Update the qubit frequency
                    qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency)
                    qubit.align()
                    # Play the saturation pulse
                    qubit.xy.play(
                        node.parameters.operation,
                        amplitude_scale=node.parameters.operation_amplitude_factor,
                        duration=qubit_pulse_duration[qubit.name] * node.u.ns,
                    )
                    qubit.align()
                    # readout the resonator
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    # Wait for the qubit to decay to the ground state
                    qubit.resonator.wait(machine.depletion_time * node.u.ns)
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])

            # Measure sequentially
            if not node.parameters.multiplexed:
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")
    
    return qua_prog
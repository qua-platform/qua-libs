def gate_sequence_and_macros(model, basic_gates_macros=None):
    prep_fiducials, meas_fiducials, germs = model.prep_fiducials(), model.meas_fiducials(), model.germs()

    gate_sequence = list({k.str.split("@")[0] for k in prep_fiducials + germs + meas_fiducials})
    gate_sequence.remove("{}")
    gate_sequence.sort(key=len, reverse=True)
    if basic_gates_macros:
        gate_sequence_macros = [s.split("G") for s in gate_sequence]
        for i, s in enumerate(gate_sequence_macros):
            s = [basic_gates_macros[k] for k in s if basic_gates_macros.get(k) is not None]
            gate_sequence_macros[i] = sequence_macros(s)
        return gate_sequence, gate_sequence_macros
    else:
        return gate_sequence


def sequence_macros(macros):
    def foo():
        for m in macros:
            m()

    return foo


def encode_circuits(circuits, model):
    gate_sequence = gate_sequence_and_macros(model)
    gate_sequence_to_index = {k: i for i, k in enumerate(gate_sequence)}
    circ_list = []
    for circ in circuits:
        start_gate = -1
        end_gate = -1
        germ_gate = -1
        c = circ.rstrip()
        gates, qubit_measures = c.split("@")
        if gates.find("(") >= 0:
            germ_start_ind = gates.find("(")
            germ_end_ind = gates.find(")")
            germ = gates[germ_start_ind + 1:germ_end_ind]
            germ_gate = gate_sequence_to_index[germ]

            if gates.find("^") >= 0:
                germ_repeat = int(gates[gates.find("^") + 1])
                germ_end_ind += 3
            else:
                germ_repeat = 1
                germ_end_ind += 1

            prep_gates = gates[:germ_start_ind]
            meas_gates = gates[germ_end_ind:]
            if prep_gates:
                start_gate = gate_sequence_to_index[prep_gates]
            if meas_gates:
                end_gate = gate_sequence_to_index[meas_gates]

        else:
            germ_repeat = 0
            done = False
            for i, v in enumerate(map(gates.startswith, gate_sequence)):
                if v:
                    start_gate = i
                    if gates[len(gate_sequence[i]):]:
                        for j, k in enumerate(map(gates[len(gate_sequence[i]):].endswith, gate_sequence)):
                            if k:
                                end_gate = j
                                if gate_sequence[start_gate] + gate_sequence[end_gate] == gates:
                                    done = True
                                    break
                    else:
                        break

                    if done:
                        break

        gates = [start_gate, end_gate, germ_gate, germ_repeat]
        circ_list.append(gates)

    return circ_list
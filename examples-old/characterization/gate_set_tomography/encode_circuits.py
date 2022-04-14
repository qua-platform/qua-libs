def base_gate_sequence_and_macros(model, basic_gates_macros: dict = None):
    """
    Given a pyGSTi model generate a minimal generating set of gate sequences by taking a union of the fiducials and the germs.
    If basic gates macros given, generate a macro for each gate sequence as well.
    @param model: A pyGSTi model
    @param basic_gates_macros: A dictionary with basic gates as keys nad macros for each gate as value
    @return: List of strings of gate sequences, and list of gate sequences macros if given.
    """
    prep_fiducials, meas_fiducials, germs = (
        model.prep_fiducials(),
        model.meas_fiducials(),
        model.germs(),
    )

    # create minimal generating gate sequence
    base_gate_sequence = list(
        {k.str.split("@")[0] for k in prep_fiducials + germs + meas_fiducials}
    )
    base_gate_sequence.remove("{}")
    base_gate_sequence.sort(key=len, reverse=True)

    if basic_gates_macros:
        # create generating gate sequence macros
        base_gate_sequence_macros = [s.split("G") for s in base_gate_sequence]
        for i, s in enumerate(base_gate_sequence_macros):
            s = [
                basic_gates_macros[k]
                for k in s
                if basic_gates_macros.get(k) is not None
            ]
            base_gate_sequence_macros[i] = sequence_macros(s)
        return base_gate_sequence, base_gate_sequence_macros
    else:
        return base_gate_sequence


def sequence_macros(macros):
    """
    Generate a single macro from a list of macros
    @param macros: List of macros
    @return: macro
    """

    def foo():
        for m in macros:
            m()

    return foo


def encode_circuits(circuits, model):
    """
    Encode a list of circuits generated from a given model. Each circuit will be encoded using 4 integers:
        1. The index of the gate sequence before the germ, i.e. preparation fiducials
        2. The index of the gate sequence after the germ, i.e. measurement fiducials
        3. The index of the germ fate sequence
        4. The number of repetitions of the germs
    @param circuits: A list of circuits generates from a pyGSTi txt file
    @param model: pyGSTi model
    @return: list of encoded circuits, each circuit is a list of 4 integers
    """
    base_gate_sequence = base_gate_sequence_and_macros(model)
    base_gate_sequence_to_index = {k: i for i, k in enumerate(base_gate_sequence)}
    circ_list = []
    for circ in circuits:
        start_gate, end_gate, germ_gate, germ_repeat = -1, -1, -1, 0

        c = circ.rstrip()
        gates, qubit_measures = c.split("@")
        if gates == "{}":  # if empty sequence do nothing
            pass
        elif gates.find("(") >= 0:  # if found germ
            # extract germ indexes
            germ_start_ind = gates.find("(")
            germ_end_ind = gates.find(")")
            germ = gates[germ_start_ind + 1 : germ_end_ind]
            germ_gate = base_gate_sequence_to_index[germ]

            # find germ repetition
            if gates.find("^") >= 0:
                germ_repeat = int(gates[gates.find("^") + 1])
                germ_end_ind += 3
            else:
                germ_repeat = 1
                germ_end_ind += 1

            prep_gates = gates[:germ_start_ind]
            meas_gates = gates[germ_end_ind:]
            if prep_gates:
                start_gate = base_gate_sequence_to_index[prep_gates]
            if meas_gates:
                end_gate = base_gate_sequence_to_index[meas_gates]

        else:  # if no germ
            """
            Match the correct combination of gate sequences for the prep and meas fiducials
            """
            done = False
            for i, v in enumerate(map(gates.startswith, base_gate_sequence)):
                if v:
                    start_gate = i
                    if gates[len(base_gate_sequence[i]) :]:
                        for j, k in enumerate(
                            map(
                                gates[len(base_gate_sequence[i]) :].endswith,
                                base_gate_sequence,
                            )
                        ):
                            if k:
                                end_gate = j
                                if (
                                    base_gate_sequence[start_gate]
                                    + base_gate_sequence[end_gate]
                                    == gates
                                ):
                                    done = True
                                    break
                    else:
                        break

                    if done:
                        break

        gates = [start_gate, end_gate, germ_gate, germ_repeat]
        circ_list.append(gates)

    return circ_list

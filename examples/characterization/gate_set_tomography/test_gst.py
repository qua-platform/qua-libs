import pygsti
from pygsti.objects import Circuit, Model, DataSet, Label
from qm.qua import *
from pygsti.construction import make_lsgst_experiment_list
from pygsti.modelpacks import smq1Q_XYI

model = smq1Q_XYI

prep_fiducials, meas_fiducials, germs, basic_gates = model.prep_fiducials(), model.meas_fiducials(), model.germs(), model.gates

gate_sequences = list({k.str.split("@")[0] for k in prep_fiducials + germs + meas_fiducials})
gate_sequences.sort(key=len, reverse=True)
gate_sequence_to_index = {k: i for i, k in enumerate(gate_sequences)}

GST_sequence_file = 'Circuits_before_results.txt'
circ_list = []
with open(file=GST_sequence_file, mode='r') as f:
    circuits = f.readlines()
    for circ in circuits:
        start_gate = -1
        end_gate = -1
        germ_gate = -1
        gates_mapped = []
        qubit_indices = []
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
            for i, v in enumerate(map(gates.startswith, gate_sequences)):
                if v:
                    start_gate = i
                    if gates[len(gate_sequences[i]):]:
                        for j, k in enumerate(map(gates[len(gate_sequences[i]):].endswith, gate_sequences)):
                            if k:
                                end_gate = j
                                if gate_sequences[start_gate] + gate_sequences[end_gate] == gates:
                                    done = True
                                    break
                    else:
                        break

                    if done:
                        break

        gates = [start_gate, end_gate, germ_gate, germ_repeat]
        circ_list.append(gates)

with open(file=GST_sequence_file, mode='r') as f:
    circuits = f.readlines()
    for i, circ in enumerate(circuits):
        c = circ.rstrip()
        gates, qubit_measures = c.split("@")
        generated_gates = ""
        generated_gates += gate_sequences[circ_list[i][0]] if circ_list[i][0] >= 0 else ""
        generated_gates += "(" + gate_sequences[circ_list[i][2]] + ")" if circ_list[i][2] >= 0 else ""
        generated_gates += "^" + str(circ_list[i][3]) if circ_list[i][3] > 1 else ""
        generated_gates += gate_sequences[circ_list[i][1]] if circ_list[i][1] >= 0 else ""
        assert gates == generated_gates

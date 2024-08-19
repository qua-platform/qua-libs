from quam_libs.wiring.connectivity.connectivity import Connectivity
from quam_libs.wiring.visualizer.visualizer import visualize_chassis
from quam_libs.wiring.wirer import allocate_wiring

visualize = True


def test_5q_allocation(instruments_2lf_2mw):
    qubits = [1, 2, 3, 4, 5]
    qubit_pairs = [(1, 2), (2, 3), (3, 4), (4, 5)]

    connectivity = Connectivity()
    connectivity.add_resonator_line(qubits=qubits)
    connectivity.add_qubit_drive_lines(qubits=qubits, slot=7, con=1)
    connectivity.add_qubit_flux_lines(qubits=qubits)
    connectivity.add_qubit_pair_flux_lines(qubit_pairs=qubit_pairs)

    allocate_wiring(connectivity, instruments_2lf_2mw)

    if visualize:
        visualize_chassis(connectivity.elements)


def test_4rr_allocation(instruments_2lf_2mw):
    connectivity = Connectivity()
    connectivity.add_resonator_line(qubits=1)
    connectivity.add_qubit_drive_lines(qubits=list(range(7)))
    connectivity.add_resonator_line(qubits=2)
    connectivity.add_resonator_line(qubits=3)
    connectivity.add_resonator_line(qubits=4)

    allocate_wiring(connectivity, instruments_2lf_2mw)

    if visualize:
        visualize_chassis(connectivity.elements)

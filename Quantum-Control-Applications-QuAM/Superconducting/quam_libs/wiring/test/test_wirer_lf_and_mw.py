from quam_libs.wiring.connectivity.connectivity import Connectivity
from quam_libs.wiring.instruments import Instruments
from quam_libs.wiring.visualizer.visualizer import visualize_chassis
from quam_libs.wiring.wirer import allocate_wiring

visualize = True

# todo: add option to switch between external mixer, mw and baseband + octave
# todo: fix wirer exception message e.g. slot 6 and IO type
# todo: flexibility in spec specifying *all* ports.

def test_5q_allocation(instruments_2lf_2mw):
    qubits = [1, 2, 3, 4, 5]
    qubit_pairs = [(1, 2), (2, 3), (3, 4), (4, 5)]

    connectivity = Connectivity()
    connectivity.add_resonator_line(qubits=qubits, slot=7)
    connectivity.add_qubit_drive_lines(qubits=qubits, slot=7, con=1)
    # connectivity.add_qubit_drive_lines(qubits=qubits[0], slot=7, con=1, port=8)
    connectivity.add_qubit_flux_lines(qubits=qubits)
    connectivity.add_qubit_pair_flux_lines(qubit_pairs=qubit_pairs)

    allocate_wiring(connectivity, instruments_2lf_2mw)

    connectivity.elements
    print(connectivity.elements)

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


def test_6rr_6xy_6flux_allocation():
    instruments = Instruments()
    instruments.add_lf_fem(con=1, slots=1)
    instruments.add_mw_fem(con=1, slots=2)

    qubits = [1, 2, 3, 4, 5, 6]
    connectivity = Connectivity()
    connectivity.add_resonator_line(qubits=qubits)
    connectivity.add_qubit_drive_lines(qubits=qubits)
    connectivity.add_qubit_flux_lines(qubits=qubits)

    allocate_wiring(connectivity, instruments)

    visualize_chassis(connectivity.elements)
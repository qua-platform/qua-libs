from quam_libs.wiring.connectivity.connectivity import Connectivity
from quam_libs.wiring.instruments import Instruments
from quam_libs.wiring.visualizer.visualizer import visualize_chassis
from quam_libs.wiring.wirer import allocate_wiring
import pytest
from pprint import pprint

visualize = False


def test_rf_io_allocation(instruments_1octave):
    qubits = [1,2,3,4,5]

    connectivity = Connectivity()
    # connectivity.add_resonator_line(qubits=qubits)
    connectivity.add_qubit_drive_lines(qubits=qubits)

    allocate_wiring(connectivity, instruments_1octave)

    pprint(connectivity.elements)
    if visualize:
        visualize_chassis(connectivity.elements)

# todo: This fails but should work
def test_qw_soprano_allocation(instruments_qw_soprano):
    qubits = [1, 2, 3, 4, 5]

    connectivity = Connectivity()
    connectivity.add_resonator_line(qubits=qubits)
    connectivity.add_qubit_drive_lines(qubits=qubits)
    connectivity.add_qubit_flux_lines(qubits=qubits)

    allocate_wiring(connectivity, instruments_qw_soprano)

    pprint(connectivity.elements)

    if visualize:
        visualize_chassis(connectivity.elements)

from quam_libs.wiring.connectivity.connectivity import Connectivity
from quam_libs.wiring.instruments import Instruments
from quam_libs.wiring.visualizer.visualizer import visualize_chassis
from quam_libs.wiring.wirer import allocate_wiring
import pytest

visualize = False


def test_rf_io_allocation(instruments_1octave):
    qubits = [1]

    connectivity = Connectivity()
    connectivity.add_resonator_line(qubits=qubits)
    connectivity.add_qubit_drive_lines(qubits=qubits)

    allocate_wiring(connectivity, instruments_1octave)

    print(connectivity.elements)
    if visualize:
        visualize_chassis(connectivity.elements)

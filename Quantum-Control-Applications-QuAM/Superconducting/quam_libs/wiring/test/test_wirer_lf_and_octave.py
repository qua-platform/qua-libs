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

def test_qw_soprano_2qb_allocation(instruments_1OPX1Octave):
    active_qubits = [1, 2]

    connectivity = Connectivity()
    # TODO: is the port here the Octave port?
    connectivity.add_resonator_line(qubits=active_qubits, con=1, port=2)
    connectivity.add_qubit_drive_lines(qubits=[1], con=1, port=2)
    connectivity.add_qubit_drive_lines(qubits=[2], con=1, port=4)
    connectivity.add_qubit_flux_lines(qubits=active_qubits)

    allocate_wiring(connectivity, instruments_1OPX1Octave)

    pprint(connectivity.elements)

    if visualize:
        visualize_chassis(connectivity.elements)

def test_qw_soprano_2qb_among_5_allocation(instruments_1OPX1Octave):
    all_qubits = [1, 2, 3, 4, 5]
    active_qubits = [1, 2]
    other_qubits = list(set(all_qubits) - set(active_qubits))

    connectivity = Connectivity()
    # TODO: I want here to declare 2 qubits that I can address with my hardware (1OPX+ and 1 Octave)
    connectivity.add_resonator_line(qubits=active_qubits, con=1, port=1)
    connectivity.add_qubit_drive_lines(qubits=[1], con=1, port=2)
    connectivity.add_qubit_drive_lines(qubits=[2], con=1, port=4)
    connectivity.add_qubit_flux_lines(qubits=active_qubits)
    # TODO: I want to add here the remaining qubits so that the QuAM can be created for the entire chip.
    #  I thus connect the other qubits to the same ports as the active qubits.
    #  Can I have the same ports used in several qubits as it is done for the resonator line?
    connectivity.add_resonator_line(qubits=other_qubits, con=1, port=1)
    connectivity.add_qubit_drive_lines(qubits=other_qubits, con=1, port=2)
    connectivity.add_qubit_flux_lines(qubits=other_qubits, con=1, port=10)


    allocate_wiring(connectivity, instruments_1OPX1Octave)

    pprint(connectivity.elements)

    if visualize:
        visualize_chassis(connectivity.elements)

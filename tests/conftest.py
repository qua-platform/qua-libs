"""Pytest configuration and fixtures for cavity wiring tests."""
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from qualang_tools.wirer import Instruments, Connectivity, allocate_wiring
from qualang_tools.wirer.wirer.channel_specs import mw_fem_spec

# Add quam_config to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "qualibration_graphs" / "superconducting"))


@pytest.fixture
def mwfem_instruments():
    """MW-FEM instrument configuration for testing."""
    instruments = Instruments()
    instruments.add_mw_fem(controller=1, slots=[1])
    return instruments


@pytest.fixture
def transmon_cavity_connectivity(mwfem_instruments):
    """Connectivity for single transmon with cavity."""
    qubits = [1]
    
    # Define channel constraints
    rr_ch = mw_fem_spec(con=1, slot=1, in_port=1, out_port=1)
    xy_ch = mw_fem_spec(con=1, slot=1, out_port=2)
    cavity_ch = mw_fem_spec(con=1, slot=1, out_port=3)
    
    # Create connectivity
    connectivity = Connectivity()
    connectivity.add_resonator_line(qubits=qubits, constraints=rr_ch)
    connectivity.add_qubit_drive_lines(qubits=qubits, constraints=xy_ch)
    connectivity.add_cavity_lines(qubit=1, constraints=cavity_ch)
    
    # Allocate wiring
    allocate_wiring(connectivity, mwfem_instruments)
    
    return connectivity


@pytest.fixture
def multi_qubit_cavity_connectivity(mwfem_instruments):
    """Connectivity for multiple transmons with cavities."""
    qubits = [1, 2]
    
    # Define channel constraints for first qubit
    rr1_ch = mw_fem_spec(con=1, slot=1, in_port=1, out_port=1)
    xy1_ch = mw_fem_spec(con=1, slot=1, out_port=2)
    cavity1_ch = mw_fem_spec(con=1, slot=1, out_port=3)
    
    # Define channel constraints for second qubit
    rr2_ch = mw_fem_spec(con=1, slot=1, in_port=2, out_port=4)
    xy2_ch = mw_fem_spec(con=1, slot=1, out_port=5)
    cavity2_ch = mw_fem_spec(con=1, slot=1, out_port=6)
    
    # Create connectivity
    connectivity = Connectivity()
    connectivity.add_resonator_line(qubits=[1], constraints=rr1_ch)
    connectivity.add_qubit_drive_lines(qubits=[1], constraints=xy1_ch)
    connectivity.add_cavity_lines(qubit=1, constraints=cavity1_ch)
    
    connectivity.add_resonator_line(qubits=[2], constraints=rr2_ch)
    connectivity.add_qubit_drive_lines(qubits=[2], constraints=xy2_ch)
    connectivity.add_cavity_lines(qubit=2, constraints=cavity2_ch)
    
    # Allocate wiring
    allocate_wiring(connectivity, mwfem_instruments)
    
    return connectivity


@pytest.fixture
def output_dir(tmp_path):
    """Temporary directory for test output files."""
    output = tmp_path / "quam_state"
    output.mkdir(parents=True, exist_ok=True)
    return output


@pytest.fixture
def test_output_dir():
    """Persistent test output directory for manual inspection."""
    output = Path(__file__).parent / "output"
    output.mkdir(parents=True, exist_ok=True)
    return output


@pytest.fixture(autouse=True)
def mock_matplotlib():
    """Mock matplotlib to prevent plot windows in tests."""
    with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.figure"):
        yield


def get_band(freq):
    """Determine the MW fem DAC band corresponding to a given frequency.
    
    Args:
        freq (float): The frequency in Hz.
        
    Returns:
        int: The Nyquist band number.
            - 1 if 50 MHz <= freq < 5.5 GHz
            - 2 if 4.5 GHz <= freq < 7.5 GHz
            - 3 if 6.5 GHz <= freq <= 10.5 GHz
            
    Raises:
        ValueError: If the frequency is outside the MW fem bandwidth [50 MHz, 10.5 GHz].
    """
    if 50e6 <= freq < 5.5e9:
        return 1
    elif 4.5e9 <= freq < 7.5e9:
        return 2
    elif 6.5e9 <= freq <= 10.5e9:
        return 3
    else:
        raise ValueError(
            f"The specified frequency {freq} Hz is outside of the MW fem bandwidth [50 MHz, 10.5 GHz]"
        )


def closest_number(lst, target):
    """Find the closest number in a list to a target value."""
    return min(lst, key=lambda x: abs(x - target))


def get_full_scale_power_dBm_and_amplitude(desired_power: float, max_amplitude: float = 0.5) -> tuple[int, float]:
    """Get the full_scale_power_dbm and waveform amplitude for the MW FEM.
    
    Args:
        desired_power (float): Desired output power in dBm.
        max_amplitude (float, optional): Maximum allowed waveform amplitude in V. Default is 0.5V.
        
    Returns:
        tuple[int, float]: The full_scale_power_dBm and waveform amplitude.
    """
    allowed_powers = [-11, -8, -5, -2, 1, 4, 7, 10, 13, 16]
    resulting_power = desired_power - 20 * np.log10(max_amplitude)
    if resulting_power < 0:
        full_scale_power_dBm = closest_number(allowed_powers, max(resulting_power + 3, -11))
    else:
        full_scale_power_dBm = closest_number(allowed_powers, min(resulting_power + 3, 16))
    amplitude = 10 ** ((desired_power - full_scale_power_dBm) / 20)
    if -11 <= full_scale_power_dBm <= 16 and -1 <= amplitude <= 1:
        return full_scale_power_dBm, amplitude
    else:
        raise ValueError(
            f"The desired power is outside the specifications ([-11; +16]dBm, [-1; +1]), "
            f"got ({full_scale_power_dBm}; {amplitude})"
        )

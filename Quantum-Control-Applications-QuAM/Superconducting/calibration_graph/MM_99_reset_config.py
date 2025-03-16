"""
        RAMSEY WITH VIRTUAL Z ROTATIONS
The program consists in playing a Ramsey sequence (x90 - idle_time - x90 - measurement) for different idle times.
Instead of detuning the qubit gates, the frame of the second x90 pulse is rotated (de-phased) to mimic an accumulated
phase acquired for a given detuning after the idle time.
This method has the advantage of playing gates on resonance as opposed to the detuned Ramsey.

From the results, one can fit the Ramsey oscillations and precisely measure the qubit resonance frequency and T2*.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit spectroscopy, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Update the qubits frequency and T2_ramsey in the state.
    - Save the current state
"""

# %% {Imports}
from pathlib import Path
import shutil
from typing import Literal
from qualibrate import QualibrationNode, NodeParameters


# %% {Node_parameters}
class Parameters(NodeParameters):
    config_type: Literal["calibrated", "scrambled"] = "calibrated"


node = QualibrationNode(name="MM_99_reset_config", parameters=Parameters())


# %%
root_path = Path("/Users/serwan/Repositories/qua-libs/Quantum-Control-Applications-QuAM/Superconducting/configuration")
quam_state_path = root_path / f"quam_state_{node.parameters.config_type}"

if not quam_state_path.exists():
    raise FileNotFoundError(f"QuAM state file not found at {quam_state_path}")

target_path = root_path / "quam_state"

# Backup target directory if it exists
backup_path = root_path / "quam_state_backup"
if target_path.exists():
    # Remove old backup if it exists
    if backup_path.exists():
        shutil.rmtree(backup_path)
    # Create backup
    shutil.copytree(target_path, backup_path)
    # Remove target directory
    shutil.rmtree(target_path)

# Copy the source directory to the target location
shutil.copytree(quam_state_path, target_path)

raise Exception(f'Successfully set the quam to "{node.parameters.config_type}"')

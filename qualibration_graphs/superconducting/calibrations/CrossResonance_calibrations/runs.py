# %%
from pathlib import Path
import subprocess

scripts = [
    "../1Q_calibrations/00_close_other_qms.py",
    "30_CR_time_rabi_QST.py",
    "31a_CR_hamiltonian_tomography_vs_cr_drive_amp.py",
    "31b_CR_hamiltonian_tomography_vs_cr_drive_phase.py",
    "31c_CR_hamiltonian_tomography_vs_cr_cancel_phase.py",
    "31d_CR_hamiltonian_tomography_vs_cr_cancel_amp.py",
    "31e_CR_correction_phase.py",
    "50a_CNOT_zaxis_fidelity.py",
    "51a_2Q_confusion_matrix.py",
    "51b_Bell_state_tomography.py",
    # "60_two_qubit_randomized_benchmarking_interleaved.py",
]
path_config = Path.cwd()
for script in scripts:
    str_ = f"Running: {script}"
    space_ = "=" * 100
    print(f"{space_}\n{str_}\n{space_}")
    subprocess.run(["python", (path_config / script).resolve()], check=True)


# %%

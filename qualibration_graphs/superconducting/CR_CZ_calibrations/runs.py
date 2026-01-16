# %%
from pathlib import Path
import subprocess

scripts = [
    "../calibrations/00_close_other_qms.py",
    "20a_XY_crosstalk_amplitude.py",
    # "20b_XY_crosstalk_phase.py",
    # "30_CR_time_rabi_QST.py",
    "31a_CR_hamiltonian_tomography_vs_cr_drive_amp.py",
    # "31b_CR_hamiltonian_tomography_vs_cr_drive_phase.py",
    # "31c_CR_hamiltonian_tomography_vs_cr_cancel_phase.py",
    # "31d_CR_hamiltonian_tomography_vs_cr_cancel_amp.py",
    # "31e_CR_correction_phase.py",
    "40a_Stark_induced_ZZ_vs_duration.py",
    # "40b_Stark_induced_ZZ_vs_duration_and_frequency.py",
    # "40c_Stark_induced_ZZ_vs_duration_and_relative_phase.py",
    # "40d_Stark_induced_ZZ_vs_duration_and_amplitude.py",
    # "41a_Stark_induced_ZZ_R_vs_frequency_and_amplitude.py",
    # "42a_CZ_calib_cz_pulse_vs_correction_phase.py",
    # "42b_CZ_calib_cz_pulse_vs_amplitude.py",
    # "42c_CZ_calib_cz_pulse_vs_relative_phase.py",
    # "50a_CNOT_zaxis_fidelity.py",
    # "51a_2Q_confusion_matrix.py",
    # "51b_Bell_state_tomography.py",
    # "60_two_qubit_randomized_benchmarking_interleaved.py",
]
path_config = Path.cwd()
for script in scripts:
    print("=" * 100, f"\nRunning: {script}\n", "=" * 100)
    subprocess.run(["python", (path_config / script).resolve()], check=True)


# %%

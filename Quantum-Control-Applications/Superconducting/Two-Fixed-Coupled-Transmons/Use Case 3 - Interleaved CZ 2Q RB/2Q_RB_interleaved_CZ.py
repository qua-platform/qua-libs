"""
        TWO-QUBIT INTERLEAVED CZ RANDOMIZED BENCHMARKING
Magesan et al., PRL 109, 080505 (2012)

Protocol:
1. Standard 2Q RB: Random 2Q Cliffords -> inverse.  Decay: P(m) = A*alpha_std^m + B
2. Interleaved CZ RB: Random Clifford * CZ * ... -> inverse.  Decay: P(m) = A*alpha_int^m + B
   F_CZ = 1 - (d-1)/d * (1 - alpha_int/alpha_std),  d = 4

Implementation:
    - CZ-native decomposition via Qiskit `optimization_level=3`
    - Split-array encoding: gate_pattern[i], rz_ctrl[i], rz_tgt[i]
    - Coupler-mediated CZ via flux-pulsing tc12

Platform: OPX+ with external IQ mixers
    - Qubit XY drives via mixInputs (IQ upconversion, external mixers)
    - Coupler flux control via singleInput (DC offset for CZ gate)
    - Clifford decomposition via Qiskit transpiler (basis: cz, rz, sx, x)
    - Minimum pulse/wait: 16 ns (4 clock cycles)

Hardware mapping (OPX+ con1):
    q1_xy   — control qubit XY drive (ports 3-4, mixInputs)
    q2_xy   — target qubit XY drive (ports 7-8, mixInputs)
    rr1     — readout resonator for q1 (ports 5-6, multiplexed)
    rr2     — readout resonator for q2 (ports 5-6, multiplexed)
    tc12    — tunable coupler flux line (port 9, singleInput)

Prerequisites:
    - pip install qiskit (>= 1.0)
    - Calibrated x90, x180 on q1_xy and q2_xy
    - Calibrated CZ gate (tc12 flux pulse amplitude/duration set in macros.cz_gate)
    - Calibrated readout: rotation_angle_q1/q2 and ge_threshold_q1/q2 in configuration.py
    - (Optional) GZ phase calibration for xi_q1, xi_q2 values

Before proceeding:
    - Verify CZ gate is freshly calibrated
    - Verify state discrimination thresholds are set in configuration.py
    - Run verification section to confirm encoding correctness
"""

from pathlib import Path
import time

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.units import unit
from qualang_tools.results.data_handler import DataHandler
from macros import cz_gate, multiplexed_readout, active_reset

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Clifford, random_clifford, Operator
from qiskit.converters import circuit_to_dag

u = unit(coerce_to_integer=True)

#####################
#    Parameters     #
#####################
# Qubit element names in this config
qc = "q1_xy"   # control qubit (fixed-frequency transmon)
qt = "q2_xy"   # target qubit (flux-tunable transmon)
rr_c = "rr1"   # resonator for control qubit
rr_t = "rr2"   # resonator for target qubit

# RB parameters
num_of_sequences = 10    # Number of random Clifford sequences per depth
n_avg = 500              # Averages per sequence
depth_list = [0, 1, 2, 3, 5, 7, 10]  # Clifford depths

seed = 42    # None for random each run
method = "cooldown"  # "cooldown" or "active"

# GZ correction angles (turns) — calibrate these for your system
# TODO: Calibrate xi values. Set to 0 to disable GZ correction.
# xi_x90: phase correction per x90 pulse (calibrate with AllXY or fine Ramsey)
# xi_x180: phase correction per x180 pulse (may differ from x90 due to amplitude-dependent AC Stark)
xi_x90_qc = 0.0
xi_x90_qt = 0.0
xi_x180_qc = 0.0
xi_x180_qt = 0.0
_xi_x90 = {qc: xi_x90_qc, qt: xi_x90_qt}
_xi_x180 = {qc: xi_x180_qc, qt: xi_x180_qt}

# Coupler parking flux (returned to after CZ gate)
cz_parking_flux = coupling_off_flux

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "num_of_sequences": num_of_sequences,
    "depth_list": depth_list,
    "seed": seed,
    "config": config,
}

###############################################
#  Section 1: CZ-Native Sequence Generation  #
###############################################
# gate_pattern encoding:
#   0 = sx(ctrl) + sx(tgt)
#   1 = sx(ctrl) + x(tgt)
#   2 = sx(ctrl) only
#   3 = x(ctrl) + sx(tgt)
#   4 = x(ctrl) + x(tgt)
#   5 = x(ctrl) only
#   6 = sx(tgt) only
#   7 = x(tgt) only
#   8 = pure rz (no physical gates)
#   9 = CZ gate

PHYS = {"sx": 0, "x": 1, "idle": 2}
PATTERN_MAP = {
    (0, 0): 0,
    (0, 1): 1,
    (0, 2): 2,
    (1, 0): 3,
    (1, 1): 4,
    (1, 2): 5,
    (2, 0): 6,
    (2, 1): 7,
    (2, 2): 8,
}
CZ_PATTERN = 9


def _circuit_to_encoded(transpiled_qc):
    """Convert a transpiled circuit to (gate_patterns, rz_ctrl, rz_tgt) arrays."""
    dag = circuit_to_dag(transpiled_qc)
    patterns = []
    rz_c = []
    rz_t = []

    for layer in dag.layers():
        ctrl_phys = PHYS["idle"]
        tgt_phys = PHYS["idle"]
        ctrl_rz = 0.0
        tgt_rz = 0.0
        is_cz = False

        for node in layer["graph"].op_nodes():
            name = node.op.name
            qubits = [transpiled_qc.find_bit(q).index for q in node.qargs]

            if name == "cz":
                is_cz = True
            elif name == "sx":
                if qubits[0] == 0:
                    ctrl_phys = PHYS["sx"]
                else:
                    tgt_phys = PHYS["sx"]
            elif name == "x":
                if qubits[0] == 0:
                    ctrl_phys = PHYS["x"]
                else:
                    tgt_phys = PHYS["x"]
            elif name == "rz":
                angle_turns = node.op.params[0] / (2 * np.pi)
                if qubits[0] == 0:
                    ctrl_rz = angle_turns
                else:
                    tgt_rz = angle_turns
            elif name in ("barrier", "measure", "delay"):
                continue
            else:
                raise ValueError(f"Unexpected gate in transpiled circuit: {name}")

        if is_cz:
            patterns.append(CZ_PATTERN)
            rz_c.append(ctrl_rz)
            rz_t.append(tgt_rz)
        elif (
            ctrl_phys != PHYS["idle"]
            or tgt_phys != PHYS["idle"]
            or ctrl_rz != 0.0
            or tgt_rz != 0.0
        ):
            patterns.append(PATTERN_MAP[(ctrl_phys, tgt_phys)])
            rz_c.append(ctrl_rz)
            rz_t.append(tgt_rz)

    return patterns, rz_c, rz_t


def _transpile_clifford(cliff):
    """Transpile a 2Q Clifford into CZ-native basis (optimization_level=3)."""
    qc_circuit = cliff.to_circuit()
    transpiled = transpile(
        qc_circuit,
        basis_gates=["cz", "rz", "sx", "x"],
        optimization_level=3,
    )
    return _circuit_to_encoded(transpiled)


def generate_standard_rb_sequence(depth, rng):
    """Generate one standard 2Q RB sequence.

    Uses explicit numpy matrix multiplication to track the Clifford product,
    avoiding Qiskit version-dependent Clifford.compose() semantics.
    """
    if depth == 0:
        return [], [], []

    patterns, rz_c, rz_t = [], [], []
    product_mat = np.eye(4, dtype=complex)

    for _ in range(depth):
        cliff = random_clifford(2, seed=rng)
        cliff_mat = Operator(cliff).data
        product_mat = cliff_mat @ product_mat
        p, rc, rt = _transpile_clifford(cliff)
        patterns.extend(p)
        rz_c.extend(rc)
        rz_t.extend(rt)

    inverse_mat = product_mat.conj().T
    inverse_cliff = Clifford.from_operator(Operator(inverse_mat))
    p, rc, rt = _transpile_clifford(inverse_cliff)
    patterns.extend(p)
    rz_c.extend(rc)
    rz_t.extend(rt)

    return patterns, rz_c, rz_t


_cz_qc = QuantumCircuit(2)
_cz_qc.cz(0, 1)
CZ_CLIFF = Clifford(_cz_qc)
_cz_mat = Operator(CZ_CLIFF).data


def generate_interleaved_rb_sequence(depth, rng):
    """Generate one interleaved CZ RB sequence.

    Uses explicit numpy matrix multiplication to track the Clifford product.
    """
    if depth == 0:
        return [], [], []

    patterns, rz_c, rz_t = [], [], []
    product_mat = np.eye(4, dtype=complex)

    for _ in range(depth):
        cliff = random_clifford(2, seed=rng)
        cliff_mat = Operator(cliff).data
        product_mat = cliff_mat @ product_mat
        p, rc, rt = _transpile_clifford(cliff)
        patterns.extend(p)
        rz_c.extend(rc)
        rz_t.extend(rt)

        # Interleave CZ
        product_mat = _cz_mat @ product_mat
        patterns.append(CZ_PATTERN)
        rz_c.append(0.0)
        rz_t.append(0.0)

    inverse_mat = product_mat.conj().T
    inverse_cliff = Clifford.from_operator(Operator(inverse_mat))
    p, rc, rt = _transpile_clifford(inverse_cliff)
    patterns.extend(p)
    rz_c.extend(rc)
    rz_t.extend(rt)

    return patterns, rz_c, rz_t


def pre_generate_sequences(num_sequences, depth_list, interleaved=False, seed=None):
    """Pre-generate all RB sequences. Returns (patterns, rz_ctrl, rz_tgt, len_list)."""
    rng = np.random.default_rng(seed)
    all_patterns = []
    all_rz_c = []
    all_rz_t = []
    len_list = []

    gen_func = generate_interleaved_rb_sequence if interleaved else generate_standard_rb_sequence

    for seq_idx in range(num_sequences):
        for d in depth_list:
            p, rc, rt = gen_func(d, rng)
            all_patterns.extend(p)
            all_rz_c.extend(rc)
            all_rz_t.extend(rt)
            len_list.append(len(p))
        if (seq_idx + 1) % 5 == 0:
            print(f"  ... {seq_idx + 1}/{num_sequences} sequences done")

    return all_patterns, all_rz_c, all_rz_t, len_list


###############################################
#  Section 2: Verification (run once)        #
###############################################

I2 = np.eye(2, dtype=complex)
SX_MAT = np.array([[1, -1j], [-1j, 1]], dtype=complex) / np.sqrt(2)
X_MAT = np.array([[0, 1], [1, 0]], dtype=complex)
CZ_MAT = np.diag([1, 1, 1, -1]).astype(complex)


def _rz_mat(turns):
    theta = 2 * np.pi * turns
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])


def _reconstruct_unitary(patterns, rz_c, rz_t):
    """Build the 4x4 unitary from encoded sequence (Qiskit qubit ordering)."""
    U = np.eye(4, dtype=complex)
    for p, rc, rt in zip(patterns, rz_c, rz_t):
        rz_layer = np.kron(_rz_mat(rt), _rz_mat(rc))
        if p == CZ_PATTERN:
            gate_layer = CZ_MAT
        else:
            ctrl_phys_type = p // 3
            tgt_phys_type = p % 3
            ctrl_mat = [SX_MAT, X_MAT, I2][ctrl_phys_type]
            tgt_mat = [SX_MAT, X_MAT, I2][tgt_phys_type]
            gate_layer = np.kron(tgt_mat, ctrl_mat)
        U = gate_layer @ rz_layer @ U
    return U


def _unitaries_match(U1, U2, atol=1e-6):
    """Check if two unitaries are equal up to global phase."""
    if np.allclose(U1, 0) or np.allclose(U2, 0):
        return np.allclose(U1, U2, atol=atol)
    phase = np.vdot(U2.ravel(), U1.ravel())
    phase /= abs(phase)
    return np.allclose(U1, phase * U2, atol=atol)


def run_verification():
    """Run all verification tests on the encoding pipeline."""
    print("=" * 60)
    print("  VERIFICATION: Encoding pipeline correctness")
    print("=" * 60)

    # Test 1: Per-Clifford encoding
    print("\nTest 1: Verifying encoding on 200 random Cliffords...")
    rng_verify = np.random.default_rng(12345)
    n_pass = 0
    for k in range(200):
        cliff = random_clifford(2, seed=rng_verify)
        U_ref = Operator(cliff).data
        pats, rc, rt = _transpile_clifford(cliff)
        U_enc = _reconstruct_unitary(pats, rc, rt)
        if _unitaries_match(U_ref, U_enc):
            n_pass += 1
        else:
            print(f"  MISMATCH on Clifford {k}!")
    print(f"  Result: {n_pass}/200 passed.")
    assert n_pass == 200, "Per-Clifford encoding has errors!"

    # Test 2: Full-sequence (forward + inverse = Identity)
    print("\nTest 2: Full-sequence verification (depth 1, 2, 3)...")
    rng_seq = np.random.default_rng(54321)
    I4 = np.eye(4, dtype=complex)
    seq_fail = 0
    for d in [1, 2, 3]:
        for trial in range(20):
            pats, rc, rt = generate_standard_rb_sequence(d, rng_seq)
            if len(pats) == 0:
                continue
            U_total = _reconstruct_unitary(pats, rc, rt)
            if not _unitaries_match(U_total, I4, atol=1e-4):
                seq_fail += 1
    assert seq_fail == 0, f"{seq_fail} standard sequences FAILED!"
    print("  All depth-1/2/3 sequences: forward + inverse = Identity.")

    # Test 3: Interleaved sequences
    print("\nTest 3: Interleaved CZ sequence verification...")
    rng_int = np.random.default_rng(99999)
    int_fail = 0
    for d in [1, 2, 3]:
        for trial in range(20):
            pats, rc, rt = generate_interleaved_rb_sequence(d, rng_int)
            if len(pats) == 0:
                continue
            U_total = _reconstruct_unitary(pats, rc, rt)
            if not _unitaries_match(U_total, I4, atol=1e-4):
                int_fail += 1
    assert int_fail == 0, f"{int_fail} interleaved sequences FAILED!"
    print("  All interleaved sequences verified.")

    # Test 4: Qubit layout
    print("\nTest 4: Checking transpiler preserves qubit ordering...")
    test_circ = QuantumCircuit(2)
    test_circ.sx(0)
    test_tr = transpile(test_circ, basis_gates=["cz", "rz", "sx", "x"], optimization_level=3)
    for inst in test_tr:
        if inst.operation.name == "sx":
            qidx = test_tr.find_bit(inst.qubits[0]).index
            assert qidx == 0, f"Layout permutation detected! SX(q0) → q{qidx}"
            print("  Qubit layout OK: SX(q0) stays on q0.")
            break

    print("\n  ALL VERIFICATION TESTS PASSED.")
    print("=" * 60)


###############################################
#  Section 3: QUA Macros for RB              #
###############################################


def _sx_qua(element):
    """GZ-corrected sx gate (pi/2 rotation) using native x90 pulse."""
    frame_rotation_2pi(_xi_x90[element], element)
    play("x90", element)
    frame_rotation_2pi(_xi_x90[element], element)


def _x_qua(element):
    """GZ-corrected x gate (pi rotation) using native x180 pulse."""
    frame_rotation_2pi(_xi_x180[element], element)
    play("x180", element)
    frame_rotation_2pi(_xi_x180[element], element)


def reset_qubits_qua():
    """Reset both qubits based on configured method."""
    if method == "cooldown":
        wait(thermalization_time * u.ns)
    elif method == "active":
        # TODO: Set ge_threshold_q1/q2 to calibrated values in configuration.py
        active_reset(ge_threshold_q1, qc, rr_c, max_tries=20)
        active_reset(ge_threshold_q2, qt, rr_t, max_tries=20)


def play_sequence_qua(gate_pat, rz_c_arr, rz_t_arr, start, length):
    """Play an encoded gate sequence from three parallel arrays."""
    i = declare(int)
    with for_(i, start, i < start + length, i + 1):
        # Only align qubit XY elements (same MW FEM) to avoid cross-FEM overhead.
        # The CZ case does its own align() including tc12 (LF FEM) when needed.
        align(qc, qt)
        frame_rotation_2pi(rz_c_arr[i], qc)
        frame_rotation_2pi(rz_t_arr[i], qt)
        with switch_(gate_pat[i], unsafe=False):
            with case_(0):  # sx(ctrl) + sx(tgt)
                _sx_qua(qc)
                _sx_qua(qt)
            with case_(1):  # sx(ctrl) + x(tgt)
                _sx_qua(qc)
                _x_qua(qt)
            with case_(2):  # sx(ctrl) only
                _sx_qua(qc)
            with case_(3):  # x(ctrl) + sx(tgt)
                _x_qua(qc)
                _sx_qua(qt)
            with case_(4):  # x(ctrl) + x(tgt)
                _x_qua(qc)
                _x_qua(qt)
            with case_(5):  # x(ctrl) only
                _x_qua(qc)
            with case_(6):  # sx(tgt) only
                _sx_qua(qt)
            with case_(7):  # x(tgt) only
                _x_qua(qt)
            with case_(8):  # pure rz (already applied above)
                wait(4, qc, qt)
            with case_(9):  # CZ gate via coupler
                # Cross-FEM align: syncs MW FEM qubits with LF FEM coupler
                align(qc, qt, "tc12")
                cz_gate(cz_parking_flux)
                # cz_gate() internally aligns all elements and settles flux.
                # Touch qubit timelines so frame_rotation works in next iteration.
                wait(4, qc)
                wait(4, qt)


def two_qubit_state_measurement():
    """Measure both qubits, threshold, and return state_gg (bool).

    Uses rotated integration weights and ge_threshold for state discrimination.
    Returns: (state_gg, I_c, I_t) where state_gg = both qubits in ground state.
    """
    I_c = declare(fixed)
    I_t = declare(fixed)
    state_gg = declare(bool)

    align(qc, qt, rr_c, rr_t)
    measure(
        "readout",
        rr_c,
        dual_demod.full("rotated_cos", "rotated_sin", I_c),
    )
    measure(
        "readout",
        rr_t,
        dual_demod.full("rotated_cos", "rotated_sin", I_t),
    )
    # state_gg = True when both qubits are in ground state (I < threshold)
    assign(state_gg, (I_c < ge_threshold_q1) & (I_t < ge_threshold_q2))
    return state_gg


###############################################
#  Section 4: Build and Run QUA Programs     #
###############################################


def build_rb_program(patterns, rz_c_data, rz_t_data, len_list_data, tag_suffix="std"):
    """Build a QUA program for standard or interleaved RB."""
    with program() as rb_prog:
        depth_len = declare(int, value=len(depth_list))
        n_avg_ = declare(int, value=n_avg)
        m = declare(int)
        depth_idx = declare(int)
        n = declare(int)
        state_gg = declare(bool)

        m_st = declare_stream()
        state_st_gg = declare_stream()

        gate_pat_qua = declare(int, value=patterns)
        rz_c_qua = declare(fixed, value=rz_c_data)
        rz_t_qua = declare(fixed, value=rz_t_data)
        len_qua = declare(int, value=len_list_data)

        start = declare(int, value=0)
        run_idx = declare(int, value=0)
        saved_start = declare(int)

        with for_(m, 0, m < num_of_sequences, m + 1):
            save(m, m_st)
            with for_(depth_idx, 0, depth_idx < depth_len, depth_idx + 1):
                assign(saved_start, start)
                with for_(n, 0, n < n_avg_, n + 1):
                    align()
                    reset_qubits_qua()
                    align()
                    reset_frame(qc, qt)
                    align()
                    play_sequence_qua(gate_pat_qua, rz_c_qua, rz_t_qua, saved_start, len_qua[run_idx])
                    align()
                    state_gg = two_qubit_state_measurement()
                    save(state_gg, state_st_gg)
                assign(start, start + len_qua[run_idx])
                assign(run_idx, run_idx + 1)

        with stream_processing():
            m_st.save("iteration")
            state_st_gg.boolean_to_int().buffer(n_avg).map(FUNCTIONS.average()).buffer(
                len(depth_list)
            ).save_all(f"state_gg_{tag_suffix}")
            state_st_gg.boolean_to_int().buffer(n_avg).map(FUNCTIONS.average()).buffer(
                len(depth_list)
            ).average().save(f"state_gg_{tag_suffix}_avg")

    return rb_prog


def run_rb_program(qmm, rb_prog, tag_suffix="std"):
    """Execute an RB program and fetch results with live progress."""
    qm = qmm.open_qm(config)
    job = qm.execute(rb_prog)
    results = fetching_tool(
        job,
        data_list=["iteration", f"state_gg_{tag_suffix}_avg", f"state_gg_{tag_suffix}"],
        mode="live",
    )
    while results.is_processing():
        iteration, avg_data, full_data = results.fetch_all()
        progress_counter(iteration, num_of_sequences, start_time=results.start_time)

    iteration, avg_data, full_data = results.fetch_all()
    qm.close()
    return avg_data, full_data


###############################################
#  Section 5: Analysis                       #
###############################################


def rb_decay(m, A, alpha, B):
    """RB decay model: P(m) = A * alpha^m + B."""
    return A * alpha**m + B


def analyze_results(y_std, y_int, depth_list):
    """Fit RB decay curves and extract CZ gate fidelity."""
    d = 4  # dimension for 2-qubit system
    x = np.array(depth_list)
    results = {}

    try:
        popt_std, pcov_std = curve_fit(
            rb_decay, x, y_std, p0=[0.75, 0.99, 0.25], bounds=([0, 0, 0], [1, 1, 1])
        )
        A_std, alpha_std, B_std = popt_std
        alpha_std_err = np.sqrt(np.diag(pcov_std))[1]
        results["fit_std"] = True
        results["popt_std"] = popt_std
        results["alpha_std"] = alpha_std
        results["alpha_std_err"] = alpha_std_err
        results["EPC_std"] = (d - 1) / d * (1 - alpha_std)
    except Exception as e:
        print(f"Standard RB fit failed: {e}")
        results["fit_std"] = False

    try:
        popt_int, pcov_int = curve_fit(
            rb_decay, x, y_int, p0=[0.75, 0.98, 0.25], bounds=([0, 0, 0], [1, 1, 1])
        )
        A_int, alpha_int, B_int = popt_int
        alpha_int_err = np.sqrt(np.diag(pcov_int))[1]
        results["fit_int"] = True
        results["popt_int"] = popt_int
        results["alpha_int"] = alpha_int
        results["alpha_int_err"] = alpha_int_err
        results["EPC_int"] = (d - 1) / d * (1 - alpha_int)
    except Exception as e:
        print(f"Interleaved RB fit failed: {e}")
        results["fit_int"] = False

    if results.get("fit_std") and results.get("fit_int"):
        alpha_ratio = alpha_int / alpha_std
        F_CZ = 1 - (d - 1) / d * (1 - alpha_ratio)
        CZ_error = 1 - F_CZ
        dF_dalpha_int = (d - 1) / (d * alpha_std)
        dF_dalpha_std = (d - 1) / d * alpha_int / alpha_std**2
        F_CZ_err = np.sqrt(
            (dF_dalpha_int * alpha_int_err) ** 2 + (dF_dalpha_std * alpha_std_err) ** 2
        )
        results["F_CZ"] = F_CZ
        results["F_CZ_err"] = F_CZ_err
        results["CZ_error"] = CZ_error
        results["alpha_ratio"] = alpha_ratio

    return results


def print_results(results):
    """Print formatted results summary."""
    print("=" * 65)
    print("  INTERLEAVED CZ RANDOMIZED BENCHMARKING RESULTS")
    print("=" * 65)
    print(f"  Qubits: {qc} (ctrl) — {qt} (target)")
    print(f"  Sequences: {num_of_sequences}, Shots: {n_avg}")

    if results.get("fit_std"):
        print(f"\n  Standard RB:")
        print(f"    alpha_std = {results['alpha_std']:.6f} +/- {results['alpha_std_err']:.6f}")
        print(f"    EPC_std   = {results['EPC_std']:.6f} ({results['EPC_std']*100:.2f}%)")

    if results.get("fit_int"):
        print(f"\n  Interleaved RB:")
        print(f"    alpha_int = {results['alpha_int']:.6f} +/- {results['alpha_int_err']:.6f}")
        print(f"    EPC_int   = {results['EPC_int']:.6f} ({results['EPC_int']*100:.2f}%)")

    if results.get("F_CZ") is not None:
        print(f"\n  CZ Gate Fidelity:")
        print(f"    F_CZ     = {results['F_CZ']:.6f} +/- {results['F_CZ_err']:.6f}")
        print(f"    CZ error = {results['CZ_error']:.6f} ({results['CZ_error']*100:.2f}%)")
        print(f"    alpha_int/alpha_std = {results['alpha_ratio']:.6f}")
    print("=" * 65)


def plot_results(y_std, y_int, results, depth_list):
    """Plot RB decay curves with fits."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    x = np.array(depth_list)
    x_fit = np.linspace(0, max(depth_list), 200)

    ax.plot(x, y_std, "o", color="steelblue", markersize=8, label="Standard 2Q RB", zorder=3)
    if results.get("fit_std"):
        ax.plot(
            x_fit,
            rb_decay(x_fit, *results["popt_std"]),
            "-",
            color="steelblue",
            linewidth=2,
            label=rf"Fit: $\alpha_{{\rm std}}$ = {results['alpha_std']:.4f}",
            alpha=0.8,
        )

    ax.plot(x, y_int, "s", color="crimson", markersize=8, label="Interleaved CZ RB", zorder=3)
    if results.get("fit_int"):
        ax.plot(
            x_fit,
            rb_decay(x_fit, *results["popt_int"]),
            "-",
            color="crimson",
            linewidth=2,
            label=rf"Fit: $\alpha_{{\rm int}}$ = {results['alpha_int']:.4f}",
            alpha=0.8,
        )

    ax.set_xlabel("Number of Cliffords (m)", fontsize=12)
    ax.set_ylabel(r"Sequence Fidelity $P(|00\rangle)$", fontsize=12)
    ax.set_title(f"Interleaved CZ RB — {qc} / {qt}", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.25, ls="--", color="gray", alpha=0.5)
    ax.grid(True, alpha=0.3)

    if results.get("F_CZ") is not None:
        ax.text(
            0.5,
            0.92,
            rf"$F_{{\rm CZ}}$ = {results['F_CZ']:.4f} ± {results['F_CZ_err']:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            ha="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    return fig


###############################################
#  Main entry point                          #
###############################################


def main():
    # --- Step 0: Verify encoding pipeline ---
    run_verification()

    # --- Step 1: Generate sequences ---
    print("\nGenerating standard RB sequences (CZ-native, opt_level=3)...")
    print("  (This takes a few minutes — Qiskit is optimizing each Clifford)")
    t0 = time.time()
    pat_std, rzc_std, rzt_std, len_list_std = pre_generate_sequences(
        num_of_sequences, depth_list, interleaved=False, seed=seed
    )
    print(f"  Standard: {len(pat_std)} layers total ({time.time() - t0:.1f}s)")

    print("\nGenerating interleaved CZ RB sequences...")
    t0 = time.time()
    pat_int, rzc_int, rzt_int, len_list_int = pre_generate_sequences(
        num_of_sequences, depth_list, interleaved=True, seed=seed + 1000 if seed else None
    )
    print(f"  Interleaved: {len(pat_int)} layers total ({time.time() - t0:.1f}s)")

    # Memory check
    OPX_LIMIT = 65000
    max_arr = max(len(pat_std), len(pat_int))
    print(f"\n--- Memory: std={len(pat_std)}, int={len(pat_int)} (limit ~{OPX_LIMIT}) ---")
    if max_arr > OPX_LIMIT:
        raise RuntimeError(
            f"Array size {max_arr} exceeds OPX limit {OPX_LIMIT}! "
            "Reduce num_of_sequences or depth_list."
        )
    print("OK — fits in OPX memory.")

    # --- Step 2: Build QUA programs ---
    print("\nBuilding QUA programs...")
    rb_standard = build_rb_program(pat_std, rzc_std, rzt_std, len_list_std, tag_suffix="std")
    rb_interleaved = build_rb_program(pat_int, rzc_int, rzt_int, len_list_int, tag_suffix="int")

    # --- Step 3: Connect and execute ---
    qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

    print("\nRunning standard 2Q RB...")
    state_gg_std_avg, state_gg_std = run_rb_program(qmm, rb_standard, tag_suffix="std")
    print(f"  P(|00>) vs depth: {np.round(state_gg_std_avg, 4)}")

    print("\nRunning interleaved CZ RB...")
    state_gg_int_avg, state_gg_int = run_rb_program(qmm, rb_interleaved, tag_suffix="int")
    print(f"  P(|00>) vs depth: {np.round(state_gg_int_avg, 4)}")

    # --- Step 4: Analysis ---
    results = analyze_results(state_gg_std_avg, state_gg_int_avg, depth_list)
    print_results(results)

    # --- Step 5: Plot ---
    fig = plot_results(state_gg_std_avg, state_gg_int_avg, results, depth_list)
    plt.savefig(
        str(save_dir / "2Q_RB_interleaved_CZ.png"),
        dpi=150,
        bbox_inches="tight",
    )

    # --- Step 6: Save data ---
    save_data_dict.update(
        {
            "depth_list": np.array(depth_list),
            "state_gg_std_avg": state_gg_std_avg,
            "state_gg_std": state_gg_std,
            "state_gg_int_avg": state_gg_int_avg,
            "state_gg_int": state_gg_int,
            "fig": fig,
        }
    )
    if results.get("F_CZ") is not None:
        save_data_dict.update(
            {
                "alpha_std": results["alpha_std"],
                "alpha_std_err": results["alpha_std_err"],
                "alpha_int": results["alpha_int"],
                "alpha_int_err": results["alpha_int_err"],
                "F_CZ": results["F_CZ"],
                "F_CZ_err": results["F_CZ_err"],
            }
        )
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    data_handler.additional_files = {str(Path(__file__).resolve()): script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="2Q_RB_interleaved_CZ")

    plt.show(block=True)


if __name__ == "__main__":
    main()

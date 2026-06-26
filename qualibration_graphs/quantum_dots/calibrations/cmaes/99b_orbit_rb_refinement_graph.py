# %%
"""Two-stage ORBIT gate optimisation with RB verification.

Pipeline (per qubit pair)
-------------------------
1. ``cmaes_orbit_99``  — CMA-ES ORBIT tune-up targeting ~99 % single-qubit
   Clifford fidelity.  Uses the depth matched to that fidelity
   (``orbit_depth = 40 ≈ 0.8 / (2·(1 − 0.99))``) and the wide default search
   bounds.  On success its ``update_state`` writes the optimal x90 amplitude,
   duration, and larmor frequency back into the QuAM state for both qubits.

2. ``rb_after_99``     — Single-qubit randomized benchmarking to independently
   verify the fidelity reached after stage 1.

3. ``cmaes_orbit_999`` — CMA-ES ORBIT tune-up targeting ~99.9 % fidelity.
   Because stage 1 already wrote its optimum into the QuAM state, this stage
   starts from that optimum as its baseline (amplitude_scale = 1.0,
   duration_offset = 0, freq_detuning = 0 all map to the stage-1 result).  We
   therefore (a) deepen the ORBIT sequence and (b) tighten the search bounds to
   a narrow window around that optimum so CMA-ES refines rather than
   re-explores.

   Depth choice (``orbit_depth = 160``).  The Fisher-optimal depth grows as
   m* ≈ 0.8 / (2·(1 − F)): 40 at 99 %, 400 at 99.9 %.  But a depth matched to
   the *target* (400) is signal-dead at the *start* — at 99 % the separation
   p^400 = 0.98^400 ≈ 3e-4 is below the shot-noise floor, so CMA-ES would have
   no gradient until fidelity already improved.  160 is the compromise: it is
   Fisher-optimal near 99.75 % (the midpoint of the 99 % → 99.9 % push), still
   leaves a measurable start signal (0.98^160 ≈ 0.04 ≈ 7σ at 600 shots × 30
   circuits) and a strong end signal (0.998^160 ≈ 0.73).  (200 also works but
   the start signal drops to ≈3σ — borderline.)

4. ``rb_after_999``    — Single-qubit randomized benchmarking to verify the
   refined fidelity, on a deeper depth range for the higher-fidelity gate.

Targeting note
--------------
Both the ORBIT nodes and the pair-RB verification nodes target ``qubit_pairs``
(the pair-RB node, ``03c_pair_single_qubit_rb``, runs RB on qubit_target then
qubit_control of each pair), so the single graph target list serves every node
and no per-node target opt-out is needed.
"""

from typing import ClassVar, List, Optional

from qualibrate.core.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.core.parameters import GraphParameters
from qualibrate.core.qualibration_graph import QualibrationGraph
from qualibrate.core.qualibration_library import QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    targets_name: ClassVar[str | None] = "qubit_pairs"
    qubit_pairs: Optional[List[str]] = ["q1_q2"]


# RB verification settings shared by both pair-RB stages.  These are set
# explicitly (rather than left to the node defaults) so the benchmark is
# reproducible and tuned for the high-fidelity regime:
#   num_circuits_per_length=40, num_shots=400  → σ_F ≈ 1/√(40·400) per depth,
#   fixed seed for deterministic circuits.
# log_scale=False ⇒ linearly spaced depths by `delta_clifford` (set per stage),
# giving dense sampling of the decay rather than powers of two.
_RB_COMMON = dict(
    num_circuits_per_length=40,
    num_shots=400,
    log_scale=False,
    seed=42,
)

# --- Stage 1: ORBIT targeting ~99 % (wide bounds, shallow depth) ------------
cmaes_orbit_99 = library.nodes["03b_cmaes_orbit"].copy(
    name="cmaes_orbit_99",
    orbit_depth=40,
    success_threshold=0.3,
    population_size=12,
    max_generations=100,  # easy stage, high SNR — converges fast
    num_circuits=30,
    num_shots=200,  # N=6000 → ~49σ at depth 40, plenty of headroom
    # Wide search around the current calibration.
    amplitude_scale_min=0.8,
    amplitude_scale_max=1.2,
    duration_offset_min=-200.0,
    duration_offset_max=200.0,
    freq_detuning_min=-100e3,
    freq_detuning_max=100e3,
)

# --- Stage 2: pair RB verification after the 99 % stage ---------------------
# Linear depths 1, 4, 8, …, 256 (delta_clifford=4) fully resolve a ~99 % decay
# (0.98^256 ≈ 6e-3).  256 / 4 = 64 (integer, required by get_depths).
rb_after_99 = library.nodes["03c_pair_single_qubit_rb"].copy(
    name="rb_after_99",
    max_circuit_depth=256,
    delta_clifford=4,
    **_RB_COMMON,
)

# --- Stage 3: ORBIT targeting ~99.9 % (tight bounds, deeper depth) ----------
# Starts from the stage-1 optimum (already in the QuAM state), so 1.0 / 0 / 0
# now mean "the stage-1 result"; we refine in a narrow window around it.
cmaes_orbit_999 = library.nodes["03b_cmaes_orbit"].copy(
    name="cmaes_orbit_999",
    orbit_depth=160,  # see module docstring for the depth rationale
    success_threshold=0.5,  # p^160 = 0.5 ⇒ F ≈ 99.78 %: a meaningful floor
    population_size=12,
    max_generations=140,  # the precision stage — gets the larger gen budget
    num_circuits=30,
    num_shots=300,  # N=9000 → ~5.3σ start signal at depth 160 (grows as F→99.75%)
    # Tight search around the stage-1 optimum.
    amplitude_scale_min=0.95,
    amplitude_scale_max=1.05,
    duration_offset_min=-40.0,
    duration_offset_max=40.0,
    freq_detuning_min=-20e3,
    freq_detuning_max=20e3,
)

# --- Stage 4: pair RB verification after the 99.9 % stage -------------------
# Deeper range (linear 1, 8, 16, …, 512 with delta_clifford=8) so the slow
# 99.9 % decay (0.998^512 ≈ 0.36) is resolved with good dynamic range for the
# fit.  512 / 8 = 64 (integer, required by get_depths).
rb_after_999 = library.nodes["03c_pair_single_qubit_rb"].copy(
    name="rb_after_999",
    max_circuit_depth=512,
    delta_clifford=8,
    **_RB_COMMON,
)


g = QualibrationGraph(
    name="OrbitRBRefinement",
    parameters=Parameters(),
    nodes={
        "cmaes_orbit_99": cmaes_orbit_99,
        "rb_after_99": rb_after_99,
        "cmaes_orbit_999": cmaes_orbit_999,
        "rb_after_999": rb_after_999,
    },
    connectivity=[
        ("cmaes_orbit_99", "rb_after_99"),
        ("rb_after_99", "cmaes_orbit_999"),
        ("cmaes_orbit_999", "rb_after_999"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run()
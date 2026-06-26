"""Black-box optimisation of a voltage-balanced CZ gate.

Overview
--------
This node optimises the parameters of a BalancedCz2QMacro (ramp_duration,
wait_duration, and barrier gate voltage) using a batched black-box
optimizer.  The default optimizer is CMA-ES, but the architecture is
designed so that swapping to a different optimizer requires changing only
a single import and function call.

The node runs 3 (or optionally 5) quantum circuits per candidate:

    Circuit 1: Init |↓↑⟩ → X90(target) → CZ → X90(target) → pair measure
    Circuit 2: Init |↓↑⟩ → X90(control) → CZ → X90(control) → pair measure
    Circuit 3: Init |↓↑⟩ → CZ → control single-qubit measure via ancilla qubit
    Circuit 4: Init |↓↑⟩ → Y90(target) → CZ → X90(target) → pair measure  [optional]
    Circuit 5: Init |↓↑⟩ → Y90(control) → CZ → X90(control) → pair measure [optional]

A pluggable scoring function converts the raw circuit probabilities into
a scalar objective for the optimizer.

Search space (3D)
-----------------
    barrier_voltage  — barrier gate voltage (V)
    wait_duration    — hold time at exchange point (ns, ×4)
    ramp_duration    — ramp between voltage levels (ns, ×4)

Architecture
------------
A single QUA program is compiled once.  Parameters are streamed via
input_stream for each candidate in a generation, avoiding recompilation.

Prerequisites:
    - Calibrated single-qubit gates (X90, Y90, X180) for both qubits.
    - Calibrated initialization (pair) and measurement (PSB pair + single-qubit).
    - BalancedCz2QMacro registered on qubit pair.

State update:
    - Updates BalancedCz2QMacro wait_duration and ramp_duration.
    - Updates the barrier gate voltage in the exchange point.
"""

# %% {Imports}
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import mlflow
import sys
import time

from qm.qua import *

from qualang_tools.multi_user import qm_session

from qualibrate.core import QualibrationNode
from qualibrate.core.models.outcome import Outcome
from quam.components.quantum_components import qubit
from quam_config import Quam

# ── Optimizer (swap this import to use a different optimizer) ──────────
from calibration_utils.cmaes import (
    OptimizationResult,
    run_cmaes_optimization,
    analyse_optimization,
    log_optimization_results,
    plot_parameter_evolution,
    plot_score_convergence_on_ax,
)
from calibration_utils.cmaes.cz_opt_parameters import CZOptParameters
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.common_utils.experiment import get_qubit_pairs
from calibration_utils.cz_optimization import (
    plot_circuit_probabilities_on_ax,
    plot_individual_scores_on_ax,
    summarise_cz_optimization,
)
from qualibration_libs.runtime import simulate_and_plot

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import HumanOutputFormat, Logger, KVWriter

from mlflow.tracking import MlflowClient
from mlflow.entities import Metric

from typing import Any, Dict

from gymnasium.spaces import Box


class QMCPhaseEnv(gym.Env):
    def __init__(self, lo, hi, job, include_quadrature):
        super(QMCPhaseEnv, self).__init__()

        self.lo = lo
        self.hi = hi
        self.job = job
        self.include_quadrature = include_quadrature

        self.circuit_keys = ["circuit_1", "circuit_2", "circuit_3"]
        if include_quadrature:
            self.circuit_keys += ["circuit_4", "circuit_5"]

        self.handles = {
            key: job.result_handles.get(key) for key in self.circuit_keys
        }

        if include_quadrature:
            self.observation_space = Box(low=0, high=1, shape=(5,), dtype=np.float32)
        else:
            self.observation_space = Box(low=0, high=1, shape=(3,), dtype=np.float32)

        self.action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        self.generation_counter = [0]

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.
        """

        return np.zeros(self.observation_space.shape, dtype=np.float32), {}
    
    def step(self, action):
        """
        Passes the action to the qua program and returns the next observation, reward, done, and info.
        """
        scaled_action = (action + 1) / 2 * (self.hi - self.lo) + self.lo
        results = run_cz_circuits(
            self.job, scaled_action.reshape(1, 3), self.handles, self.generation_counter, self.include_quadrature
        )

        reward, info = self._calculate_reward(results)

        return np.array([results[key][0] for key in self.circuit_keys], dtype=np.float32), reward, True, False, info

    def _calculate_reward(self, results):
        """Calculates the reward based on the results of the circuits.
        """

        P_x_0 = results["circuit_1"][0]
        P_x_1 = results["circuit_2"][0]
        cz = results["circuit_3"][0]

        R_swap = cz

        if self.include_quadrature:
            P_y_0 = results["circuit_4"][0]
            P_y_1 = results["circuit_5"][0]

            theta_0 = np.arctan2(1/2 - P_y_0, P_x_0 - 1/2)
            theta_1 = np.arctan2(P_y_1 - 1/2, 1/2 - P_x_1)
        else:
            theta_0 = np.arccos(2 * P_x_0 - 1)
            theta_1 = np.arccos(1 - 2 * P_x_1)

        R_phase = (1 - np.cos(theta_0 + theta_1))/2

        reward = R_swap * R_phase

        return reward, {
            "R_swap": R_swap,
            "R_phase": R_phase,
            "theta_0": theta_0,
            "theta_1": theta_1
        }
    
class BatchedMLflowLogger(KVWriter):
    """
    Batched MLflow logger for Stable Baselines3.
    Logs metrics to MLflow in batches to reduce the number of API calls.
    """
    def __init__(self, log_interval=100):
        """
        Initialize the BatchedMLflowLogger.
        Args:
            log_interval (int): The number of metrics to accumulate before logging to MLflow.
                Defaults to 100.
        """
        self.log_interval = log_interval
        self.metrics_buffer = []
        self.client = MlflowClient()
        
    def write(self, key_values: Dict[str, Any], key_excluded:dict[str, tuple[str, ...]], step: int) -> None:
        """
        Write metrics to the logger.
        Args:
            key_values (Dict[str, Any]): Dictionary of metric names and values.
            key_excluded (Dict[str, Tuple[str, ...]]): Dictionary of metric names and excluded loggers.
            step (int): The current step or iteration.
        """
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):

                    metric = Metric(
                        key=key,
                        value=float(value),  # Ensure it's a float
                        timestamp=int(time.time() * 1000),
                        step=step
                    )
                    self.metrics_buffer.append(metric)
        
        if len(self.metrics_buffer) >= self.log_interval:
            self.flush()

    def flush(self) -> None:
        """
        Flush the buffered metrics to MLflow.
        """
        if self.metrics_buffer:
            self.client.log_batch(
                run_id=mlflow.active_run().info.run_id,
                metrics=self.metrics_buffer
            )
            self.metrics_buffer = []

    def close(self) -> None:
        """
        Ensure all remaining metrics are flushed when logger is closed.
        This method must be called directly after learning to ensure all metrics are logged to MLflow.
        """
        self.flush()

# %% {Node initialisation}
description = """
        CZ GATE OPTIMISATION — BLACK-BOX (VOLTAGE-BALANCED CZ)
Optimises the voltage-balanced CZ gate parameters (barrier voltage,
wait duration, ramp duration) using a batched black-box optimizer.

The default optimizer is CMA-ES but can be trivially swapped by changing
a single import.  The experiment function (run_cz_circuits) and scoring
function (compute_score) are optimizer-agnostic and fully documented for
reuse with alternative optimizers.

Search space (3D):
    [barrier_voltage, wait_duration, ramp_duration]

Circuits measure P(even parity) under X-Ramsey and bare-CZ configurations
to characterise the conditional phase and leakage of the gate.

Prerequisites:
    - Calibrated single-qubit gates (X90, Y90) for both qubits.
    - Calibrated pair initialization and PSB parity measurement.
    - BalancedCz2QMacro registered on qubit pair.

State update:
    - Updates BalancedCz2QMacro wait_duration and ramp_duration.
    - Updates the barrier gate voltage in the exchange voltage point.
"""

node = QualibrationNode[CZOptParameters, Quam](
    name="04_T_cz_optimization",
    description=description,
    parameters=CZOptParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[CZOptParameters, Quam]):
    """Debug-only parameter overrides; skipped when run externally."""
    node.parameters.max_generations = 30
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.num_shots = 10
    node.parameters.population_size = 2
    # node.parameters.include_quadrature_circuits = True
    # node.parameters.simulate = True


node.machine = Quam.load()


# ── Helpers ──────────────────────────────────────────────────────────────

_PARAM_NAMES = ["barrier_voltage", "wait_duration", "ramp_duration"]


def _quantize_duration(val: float, min_val: int = 16) -> int:
    """Quantize a continuous duration to a multiple of 4 ns, clamped."""
    return max(min_val, int(round(val / 4.0)) * 4)


# ── Scoring function (pluggable) ─────────────────────────────────────────

def compute_score(measurements: dict[str, np.ndarray]) -> np.ndarray:
    """Compute a scalar score per candidate from circuit measurements.

    Parameters
    ----------
    measurements : dict[str, np.ndarray]
        Output of run_cz_circuits(). Keys and semantics:
            "circuit_1": P(even parity) after X(target)/2 - CZ - X(target)/2
            "circuit_2": P(even parity) after X(control)/2 - CZ - X(control)/2
            "circuit_3": P(spin-up control) after bare CZ
            "circuit_4": P(even parity) after Y(target)/2 - CZ - X(target)/2  [optional]
            "circuit_5": P(even parity) after Y(control)/2 - CZ - X(control)/2 [optional]
        Each np.ndarray has shape (pop_size,) with values in [0, 1].

    Returns
    -------
    np.ndarray, shape (pop_size,)
        Score per candidate. Higher is better.

    Notes
    -----
    This is a placeholder implementation. Replace with a physics-informed
    cost function once ideal target values are determined experimentally.
    """
    score = np.zeros(len(measurements["circuit_1"]))
    for v in measurements.values():
        score += v
    return score


# ── Experiment function (optimizer-agnostic) ──────────────────────────────

def run_cz_circuits(
    job,
    candidates_phys: np.ndarray,
    handles: dict,
    generation_counter: list,
    include_quadrature: bool,
) -> dict[str, np.ndarray]:
    """Run one generation of CZ circuits on hardware.

    Pushes candidate parameters via input streams and waits for results.
    This function is optimizer-agnostic: any optimizer that produces a
    batch of candidate parameter vectors can call it.

    Parameters
    ----------
    job : QmJob
        Running QUA job with input streams.
    candidates_phys : np.ndarray, shape (pop_size, 3)
        Physical parameter values per candidate. Columns:
            0: barrier_voltage (V) — barrier gate voltage
            1: wait_duration (ns) — hold time at exchange point
            2: ramp_duration (ns) — ramp time between voltage levels
    handles : dict
        Result handles keyed by "circuit_1" through "circuit_5".
    generation_counter : list[int]
        Mutable single-element list holding the current generation index.
        Incremented after each successful fetch.
    include_quadrature : bool
        Whether circuits 4 and 5 are present in the compiled program.

    Returns
    -------
    dict[str, np.ndarray]
        Keys "circuit_1" through "circuit_3" (or "circuit_5" if quadrature
        is enabled). Each value is np.ndarray of shape (pop_size,) containing
        the average measurement probability for that circuit.
    """
    import time as _time

    for c in candidates_phys:
        job.push_to_input_stream("barrier_voltage", float(c[0]))
        job.push_to_input_stream("wait_duration", _quantize_duration(c[1]))
        job.push_to_input_stream("ramp_duration", _quantize_duration(c[2]))

    target_count = generation_counter[0] + 1

    circuit_keys = ["circuit_1", "circuit_2", "circuit_3"]
    if include_quadrature:
        circuit_keys += ["circuit_4", "circuit_5"]

    for key in circuit_keys:
        while handles[key].count_so_far() < target_count:
            _time.sleep(0.001)

    measurements = {}
    for key in circuit_keys:
        measurements[key] = np.asarray(
            handles[key].fetch(generation_counter[0], flat_struct=True),
            dtype=np.float64,
        )

    generation_counter[0] += 1
    return measurements


# %% {Create_QUA_program}
def _build_qua_program(node, qubit_pair, include_quadrature: bool):
    """Build a QUA program for CZ characterisation circuits.

    The program streams barrier_voltage, wait_duration, and ramp_duration
    per candidate and runs 3 (or 5) circuits for each.

    The balanced CZ voltage sequence is inlined with pre-assigned QUA
    variables to avoid nested expressions that can trip the OPX compiler.
    """
    qubit_target = qubit_pair.qubit_target
    qubit_control = qubit_pair.qubit_control

    num_shots = node.parameters.num_shots
    pop_size = node.parameters.population_size

    barrier_gate_id = qubit_pair.quantum_dot_pair.barrier_gate.id
    vs = qubit_pair.voltage_sequence
    vs.limit_play_commands = True
    gates = list(vs.gate_set.channels.keys())

    with program() as qua_prog:
        barrier_v_in = declare_input_stream(
            "client", stream_id="barrier_voltage", dtype=fixed
        )
        wait_dur_in = declare_input_stream(
            "client", stream_id="wait_duration", dtype=int
        )
        ramp_dur_in = declare_input_stream(
            "client", stream_id="ramp_duration", dtype=int
        )

        barrier_v = declare(fixed)
        neg_barrier_v = declare(fixed)
        wait_dur = declare(int)
        ramp_dur = declare(int)
        double_ramp_dur = declare(int)

        candidate_idx = declare(int)
        shot_idx = declare(int)

        state_c1 = declare(int)
        state_c2 = declare(int)
        state_c3 = declare(int)
        state_c1_st = declare_output_stream()
        state_c2_st = declare_output_stream()
        state_c3_st = declare_output_stream()

        if include_quadrature:
            state_c4 = declare(int)
            state_c5 = declare(int)
            state_c4_st = declare_output_stream()
            state_c5_st = declare_output_stream()

        with infinite_loop_():
            with for_(candidate_idx, 0, candidate_idx < pop_size, candidate_idx + 1):
                advance_input_stream(barrier_v_in)
                advance_input_stream(wait_dur_in)
                advance_input_stream(ramp_dur_in)
                assign(barrier_v, barrier_v_in)
                assign(neg_barrier_v, -barrier_v_in)
                assign(wait_dur, wait_dur_in)
                assign(ramp_dur, ramp_dur_in)
                assign(double_ramp_dur, ramp_dur_in << 1)

                with for_(shot_idx, 0, shot_idx < num_shots, shot_idx + 1):
                    # ─── Circuit 1: X90(target) → CZ → X90(target) → pair measure ───
                    qubit_pair.initialize()
                    align(qubit_target.xy.name, qubit_target.physical_channel.id)
                    qubit_target.x90()
                    align(qubit_target.xy.name, qubit_pair.quantum_dot_pair.barrier_gate.physical_channel.id)
                    vs.ramp_to_voltages({barrier_gate_id: neg_barrier_v}, duration=wait_dur, ramp_duration=ramp_dur, ensure_align=False)
                    vs.ramp_to_voltages({barrier_gate_id: barrier_v}, duration=wait_dur, ramp_duration=double_ramp_dur, ensure_align=False)
                    vs.ramp_to_voltages({barrier_gate_id: 0.0}, duration=16, ramp_duration=ramp_dur, ensure_align=False)
                    align(qubit_target.xy.name, qubit_pair.quantum_dot_pair.barrier_gate.physical_channel.id)
                    qubit_target.x90()
                    align(qubit_target.xy.name, qubit_target.physical_channel.id)
                    p = qubit_pair.measure()

                    assign(state_c1, Cast.to_int(p))
                    save(state_c1, state_c1_st)

                    # ─── Circuit 2: X90(control) → CZ → X90(control) → pair measure ───
                    qubit_pair.initialize()
                    align(qubit_target.xy.name, qubit_target.physical_channel.id)
                    qubit_target.x180()
                    align(qubit_target.xy.name, qubit_control.xy.name)
                    qubit_control.x90()
                    align(qubit_control.xy.name, qubit_pair.quantum_dot_pair.barrier_gate.physical_channel.id)
                    vs.ramp_to_voltages({barrier_gate_id: neg_barrier_v}, duration=wait_dur, ramp_duration=ramp_dur, ensure_align=False)
                    vs.ramp_to_voltages({barrier_gate_id: barrier_v}, duration=wait_dur, ramp_duration=double_ramp_dur, ensure_align=False)
                    vs.ramp_to_voltages({barrier_gate_id: 0.0}, duration=16, ramp_duration=ramp_dur, ensure_align=False)
                    align(qubit_control.xy.name, qubit_pair.quantum_dot_pair.barrier_gate.physical_channel.id)
                    qubit_control.x90()
                    align(qubit_control.xy.name, qubit_target.physical_channel.id)
                    p = qubit_pair.measure()

                    assign(state_c2, Cast.to_int(p))
                    save(state_c2, state_c2_st)

                    # ─── Circuit 3: CZ → single-qubit measure on control ───
                    qubit_pair.initialize()
                    align(qubit_pair.quantum_dot_pair.barrier_gate.physical_channel.id, qubit_target.physical_channel.id)
                    vs.ramp_to_voltages({barrier_gate_id: neg_barrier_v}, duration=wait_dur, ramp_duration=ramp_dur, ensure_align=False)
                    vs.ramp_to_voltages({barrier_gate_id: barrier_v}, duration=wait_dur, ramp_duration=double_ramp_dur, ensure_align=False)
                    vs.ramp_to_voltages({barrier_gate_id: 0.0}, duration=16, ramp_duration=ramp_dur, ensure_align=False)
                    align(qubit_pair.quantum_dot_pair.barrier_gate.physical_channel.id, qubit_target.physical_channel.id)
                    p = qubit_control.measure()

                    assign(state_c3, Cast.to_int(p))
                    save(state_c3, state_c3_st)

                    if include_quadrature:
                        # ─── Circuit 4: Y90(target) → CZ → X90(target) ───
                        qubit_pair.initialize()
                        align(qubit_target, qubit_target.physical_channel.id)
                        qubit_target.y90()
                        align(qubit_target.xy.name, qubit_pair.quantum_dot_pair.barrier_gate.physical_channel.id)
                        vs.ramp_to_voltages({barrier_gate_id: neg_barrier_v}, duration=wait_dur, ramp_duration=ramp_dur, ensure_align=False)
                        vs.ramp_to_voltages({barrier_gate_id: barrier_v}, duration=wait_dur, ramp_duration=double_ramp_dur, ensure_align=False)
                        vs.ramp_to_voltages({barrier_gate_id: 0.0}, duration=16, ramp_duration=ramp_dur, ensure_align=False)
                        align(qubit_pair.quantum_dot_pair.barrier_gate.physical_channel.id, qubit_target.xy.name)
                        qubit_target.x90()
                        align(qubit_target.xy.name, qubit_target.physical_channel.id)
                        p = qubit_pair.measure()

                        assign(state_c4, Cast.to_int(p))
                        save(state_c4, state_c4_st)

                        # ─── Circuit 5: Y90(control) → CZ → X90(control) ───
                        qubit_pair.initialize()
                        align(qubit_target.physical_channel.id, qubit_target.xy.name)
                        qubit_target.x180()
                        align(qubit_target.xy.name, qubit_control.xy.name)
                        qubit_control.y90()
                        align(qubit_control.xy.name, qubit_pair.quantum_dot_pair.barrier_gate.physical_channel.id)
                        vs.ramp_to_voltages({barrier_gate_id: neg_barrier_v}, duration=wait_dur, ramp_duration=ramp_dur, ensure_align=False)
                        vs.ramp_to_voltages({barrier_gate_id: barrier_v}, duration=wait_dur, ramp_duration=double_ramp_dur, ensure_align=False)
                        vs.ramp_to_voltages({barrier_gate_id: 0.0}, duration=16, ramp_duration=ramp_dur, ensure_align=False)
                        align(qubit_pair.quantum_dot_pair.barrier_gate.physical_channel.id, qubit_control.xy.name)
                        qubit_control.x90()
                        align(qubit_control.xy.name, qubit_target.physical_channel.id)
                        p = qubit_pair.measure()

                        assign(state_c5, Cast.to_int(p))
                        save(state_c5, state_c5_st)
        with stream_processing():
            (
                state_c1_st
                .buffer(num_shots)
                .map(FUNCTIONS.average())
                .buffer(pop_size)
                .save_all("circuit_1")
            )
            (
                state_c2_st
                .buffer(num_shots)
                .map(FUNCTIONS.average())
                .buffer(pop_size)
                .save_all("circuit_2")
            )
            (
                state_c3_st
                .buffer(num_shots)
                .map(FUNCTIONS.average())
                .buffer(pop_size)
                .save_all("circuit_3")
            )
            if include_quadrature:
                (
                    state_c4_st
                    .buffer(num_shots)
                    .map(FUNCTIONS.average())
                    .buffer(pop_size)
                    .save_all("circuit_4")
                )
                (
                    state_c5_st
                    .buffer(num_shots)
                    .map(FUNCTIONS.average())
                    .buffer(pop_size)
                    .save_all("circuit_5")
                )

    return qua_prog


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[CZOptParameters, Quam]):
    """Validate parameters and compile the QUA program."""
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    if not qubit_pairs:
        raise ValueError(
            "No qubit pairs resolved — check qubit_pairs parameter or machine config."
        )

    include_quadrature = node.parameters.include_quadrature_circuits
    node.namespace["include_quadrature"] = include_quadrature

    qp = qubit_pairs[0]
    node.namespace["qua_program"] = _build_qua_program(node, qp, include_quadrature)


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[CZOptParameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Run_optimization_loop}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def run_optimization_loop(node: QualibrationNode[CZOptParameters, Quam]):
    """Execute the black-box optimisation for each qubit pair.

    The optimizer (CMA-ES by default) calls run_cz_circuits() to evaluate
    candidates and compute_score() to convert measurements to a scalar.
    To swap the optimizer, replace the run_cmaes_optimization import and call.
    """
    qubit_pairs = node.namespace["qubit_pairs"]
    include_quadrature = node.namespace["include_quadrature"]

    qmm = node.machine.connect(timeout=node.parameters.compilation_timeout)
    config = node.machine.generate_config()

    optimization_results = {}
    measurement_streams = {}

    for qp in qubit_pairs:
        node.log(f"  Starting CZ optimisation for pair {qp.name}...")

        lo = np.array([
            node.parameters.barrier_voltage_min,
            float(node.parameters.wait_duration_min),
            float(node.parameters.ramp_duration_min),
        ])
        hi = np.array([
            node.parameters.barrier_voltage_max,
            float(node.parameters.wait_duration_max),
            float(node.parameters.ramp_duration_max),
        ])

        qua_prog = _build_qua_program(node, qp, include_quadrature)

        circuit_history = {
            "circuit_1": [], "circuit_2": [], "circuit_3": [],
            "score": [],
            "all_scores": [],
            "running_best_score": [],
            "running_best_measurements": [],
        }
        if include_quadrature:
            circuit_history["circuit_4"] = []
            circuit_history["circuit_5"] = []

        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(qua_prog)

            print('Creating Environment...')
            env = Monitor(QMCPhaseEnv(lo, hi, job, include_quadrature))

            action_noise = NormalActionNoise(mean=np.zeros(3), sigma=node.parameters.action_sigma * np.ones(3))
            print('Intializing Model...')
            model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=1, learning_starts=5)
            print('Begining training...')
            model.learn(total_timesteps=node.parameters.num_steps, log_interval=1)

            obs, _ = env.reset()
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            print(f'Final Obs: {obs}')
            print(f'Final Reward: {reward}')
            print(f'Final Info: {info}')
            
            job.cancel()
    node.namespace["optimization_results"] = optimization_results
    node.namespace["measurement_streams"] = measurement_streams
    node.results["optimization_results"] = {
        name: result.to_dict() for name, result in optimization_results.items()
    }

    def _serialize_value(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {kk: _serialize_value(vv) for kk, vv in v.items()}
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
            return [arr.tolist() for arr in v]
        return v

    node.results["measurement_streams"] = {
        name: _serialize_value(streams)
        for name, streams in measurement_streams.items()
    }


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[CZOptParameters, Quam]):
    """Load a previously saved optimisation result."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)
    node.namespace["include_quadrature"] = node.parameters.include_quadrature_circuits
    node.namespace["optimization_results"] = {
        name: OptimizationResult.from_dict(d)
        for name, d in node.results["optimization_results"].items()
    }

    def _deserialize_value(v):
        if isinstance(v, dict):
            return {kk: _deserialize_value(vv) for kk, vv in v.items()}
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            return [np.array(arr) for arr in v]
        return v

    raw_streams = node.results.get("measurement_streams", {})
    node.namespace["measurement_streams"] = {
        name: _deserialize_value(streams)
        for name, streams in raw_streams.items()
    }


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[CZOptParameters, Quam]):
    """Summarise the optimisation outcome per qubit pair."""
    opt_results = node.namespace["optimization_results"]
    measurement_streams = node.namespace.get("measurement_streams", {})

    fit_results = analyse_optimization(
        opt_results, success_threshold=node.parameters.success_threshold
    )
    fit_results = summarise_cz_optimization(fit_results, measurement_streams)

    node.results["fit_results"] = fit_results

    log_optimization_results(opt_results, log_callable=node.log)

    for qp in node.namespace["qubit_pairs"]:
        summary = fit_results.get(qp.name)
        if summary is None:
            continue
        best = summary.get("best_params", {})
        node.log(
            f"  [{qp.name}] Best CZ params: "
            f"barrier_V={best.get('barrier_voltage', 0):.4f} V, "
            f"wait_dur={best.get('wait_duration', 0):.0f} ns, "
            f"ramp_dur={best.get('ramp_duration', 0):.0f} ns, "
            f"score={summary.get('best_score', 0):.4f}"
        )

    node.outcomes = {
        qp_name: (Outcome.SUCCESSFUL if summary["success"] else Outcome.FAILED)
        for qp_name, summary in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[CZOptParameters, Quam]):
    """Generate convergence and circuit probability plots (CMA-ES example)."""
    opt_results = node.namespace["optimization_results"]
    measurement_streams = node.namespace.get("measurement_streams", {})
    include_quadrature = node.namespace["include_quadrature"]

    pair_names = list(opt_results.keys())
    n_pairs = max(len(pair_names), 1)

    fig_combined, axes = plt.subplots(
        3, n_pairs, figsize=(9 * n_pairs, 14), squeeze=False,
    )

    for col, pname in enumerate(pair_names):
        streams = measurement_streams.get(pname, {})
        circuit_hist = streams.get("circuit_history", {})

        plot_circuit_probabilities_on_ax(
            axes[0, col], circuit_hist, include_quadrature, pname,
        )
        plot_score_convergence_on_ax(axes[1, col], opt_results[pname], pname)
        plot_individual_scores_on_ax(axes[2, col], circuit_hist, pname)

    fig_combined.tight_layout()

    fig_params = plot_parameter_evolution(opt_results)
    plt.show()

    node.results["figures"] = {
        "circuits_and_convergence": fig_combined,
        "parameter_evolution": fig_params,
    }
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[CZOptParameters, Quam]):
    """Apply optimal CZ parameters to the BalancedCz2QMacro."""
    fit_results = node.results["fit_results"]

    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            pair_summary = fit_results.get(qp.name)
            if pair_summary is None:
                continue
            if not pair_summary["success"]:
                node.log(
                    f"  {qp.name}: optimisation did not succeed — skipping."
                )
                continue

            best = pair_summary["best_params"]
            opt_barrier_v = best["barrier_voltage"]
            opt_wait_dur = _quantize_duration(best["wait_duration"])
            opt_ramp_dur = _quantize_duration(best["ramp_duration"])

            cz_macro = qp.macros["cz"]
            old_wait = cz_macro.wait_duration
            old_ramp = cz_macro.ramp_duration

            cz_macro.update(
                wait_duration=opt_wait_dur,
                ramp_duration=opt_ramp_dur,
            )

            barrier_gate = qp.quantum_dot_pair.barrier_gate
            point_name = cz_macro.point
            point_obj = qp.voltage_sequence.gate_set.macros.get(point_name)
            if point_obj is not None:
                old_barrier_v = point_obj.voltages.get(barrier_gate.id, 0.0)
                point_obj.voltages[barrier_gate.id] = opt_barrier_v
            else:
                old_barrier_v = 0.0

            node.log(
                f"  {qp.name}: CZ params updated — "
                f"barrier_V: {old_barrier_v:.4f} → {opt_barrier_v:.4f} V, "
                f"wait_dur: {old_wait} → {opt_wait_dur} ns, "
                f"ramp_dur: {old_ramp} → {opt_ramp_dur} ns, "
                f"score={pair_summary['best_score']:.4f}"
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[CZOptParameters, Quam]):
    """Persist all results, figures, and parameters to disk."""
    node.save()

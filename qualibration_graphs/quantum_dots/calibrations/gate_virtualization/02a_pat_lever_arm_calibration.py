# %% {Imports}
from __future__ import annotations

from typing import Dict

from qualibrate import QualibrationNode
from quam_config import Quam

from calibration_utils.gate_virtualization import PATLeverArmParameters

# %% {Node initialisation}
description = """
        PAT LEVER-ARM CALIBRATION (DOT-PAIR SCALE FACTORS)
This node is a placeholder for PAT-based extraction of detuning lever arms
for inter-dot transitions used by barrier virtualization.

Current scope:
    - Accept PAT lever-arm values via parameters (`pat_lever_arm_mapping`).
    - Expand dot-pair lever arms into barrier-keyed lever arms using
      `barrier_dot_pair_mapping`.
    - Persist mapping in node results for downstream manual/graph use.

TODO:
    - Implement hardware PAT acquisition and fitting in this node.
    - Persist lever-arm calibration in machine state once QuAM schema supports it.
"""


node = QualibrationNode[PATLeverArmParameters, Quam](
    name="02a_pat_lever_arm_calibration",
    description=description,
    parameters=PATLeverArmParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[PATLeverArmParameters, Quam]):
    """Allow local debug parameter overrides."""
    # node.parameters.pat_lever_arm_mapping = {"qd_pair_1_2": 175.0, "qd_pair_2_3": 168.0}
    # node.parameters.barrier_dot_pair_mapping = {"barrier_12": "qd_pair_1_2", "barrier_23": "qd_pair_2_3"}
    pass


node.machine = Quam.load()


def _build_dot_pair_mapping(node: QualibrationNode[PATLeverArmParameters, Quam]) -> Dict[str, float]:
    """Return dot-pair lever arms, with conservative fallback."""
    supplied = node.parameters.pat_lever_arm_mapping or {}
    default = float(node.parameters.default_lever_arm)
    if supplied:
        return {str(k): float(v) for k, v in supplied.items()}
    return {}


def _build_barrier_mapping(
    dot_pair_lever_arms: Dict[str, float],
    barrier_dot_pair_mapping: Dict[str, str],
    default: float,
) -> Dict[str, float]:
    """Map barrier names to lever arms using target-barrier -> dot-pair mapping."""
    out: Dict[str, float] = {}
    for barrier_name, dot_pair_name in barrier_dot_pair_mapping.items():
        out[str(barrier_name)] = float(dot_pair_lever_arms.get(str(dot_pair_name), default))
    return out


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[PATLeverArmParameters, Quam]):
    """No hardware QUA program yet; prepare mapping inputs."""
    dot_pair_lever_arms = _build_dot_pair_mapping(node)
    barrier_dot_pair_mapping = node.parameters.barrier_dot_pair_mapping or {}
    node.namespace["dot_pair_lever_arms"] = dot_pair_lever_arms
    node.namespace["barrier_dot_pair_mapping"] = {str(k): str(v) for k, v in barrier_dot_pair_mapping.items()}


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[PATLeverArmParameters, Quam]):
    """Placeholder simulation action for PAT node."""
    node.log("PAT simulation is not implemented yet; using provided mapping/fallback values.")


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[PATLeverArmParameters, Quam]):
    """Placeholder execute action for PAT node."""
    node.log("PAT execute is not implemented yet; using provided mapping/fallback values.")


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[PATLeverArmParameters, Quam]):
    """Load previously saved PAT node results."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id


# %% {Analyse_data}
@node.run_action(skip_if=False)
def analyse_data(node: QualibrationNode[PATLeverArmParameters, Quam]):
    """Build and store dot-pair and barrier lever-arm mappings."""
    dot_pair_lever_arms = node.namespace.get("dot_pair_lever_arms") or _build_dot_pair_mapping(node)
    barrier_dot_pair_mapping = node.namespace.get("barrier_dot_pair_mapping") or (
        node.parameters.barrier_dot_pair_mapping or {}
    )
    default = float(node.parameters.default_lever_arm)

    barrier_lever_arms = _build_barrier_mapping(
        dot_pair_lever_arms=dot_pair_lever_arms,
        barrier_dot_pair_mapping=barrier_dot_pair_mapping,
        default=default,
    )

    node.results["lever_arm_mapping_dot_pair"] = dot_pair_lever_arms
    node.results["lever_arm_mapping_barrier"] = barrier_lever_arms
    node.results["default_lever_arm"] = default
    node.results["todo"] = (
        "Machine/state persistence of PAT lever arms is pending QuAM schema support. "
        "For now, pass mapping into barrier node parameters."
    )


# %% {Plot_data}
@node.run_action(skip_if=False)
def plot_data(node: QualibrationNode[PATLeverArmParameters, Quam]):
    """No plotting for placeholder PAT node."""
    node.log("PAT plotting is not implemented for placeholder mode.")


# %% {Update_state}
@node.run_action(skip_if=False)
def update_state(node: QualibrationNode[PATLeverArmParameters, Quam]):
    """No machine-state update yet (TODO)."""
    node.log(
        "TODO: persist PAT lever arms in machine state once QuAM lever-arm support exists. "
        "Current node stores mappings in node.results only."
    )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[PATLeverArmParameters, Quam]):
    """Persist node outputs."""
    node.save()

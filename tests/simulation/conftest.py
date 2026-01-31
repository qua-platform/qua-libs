"""Fixtures and helpers for QUA program simulation tests.

This module provides:
- Custom macros (X180Macro, MeasureMacro) for programmatic QuAM construction
- minimal_quam_factory: Factory fixture for creating LossDiVincenzoQuam machines
- node_loader: Fixture for dynamically loading calibration node modules
- qm_saas_credentials: Fixture for loading QM SaaS credentials
- markdown_generator: Fixture for generating README.md documentation
- save_simulation_plot: Fixture for saving simulation plots
- simulation_test_context: Combined fixture providing all test context
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

# Ensure matplotlib/qualibrate can write caches/logs under repo.
_cache_base = Path(__file__).resolve().parent / ".pytest_cache"
_cache_base.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_base / "matplotlib"))
os.environ.setdefault("QUALIBRATE_LOG_DIR", str(_cache_base / "qualibrate"))

import matplotlib

matplotlib.use("Agg")  # Headless backend for CI/local runs
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt

from qm.qua import assign, declare, fixed

from quam.components import pulses
from quam.components.channels import StickyChannelAddon
from quam.components.hardware import FrequencyConverter, LocalOscillator
from quam.components.ports import (
    LFFEMAnalogInputPort,
    LFFEMAnalogOutputPort,
    MWFEMAnalogOutputPort,
)
from quam.core import quam_dataclass
from quam.core.macro.quam_macro import QuamMacro
from quam.utils.qua_types import QuaVariableBool

from quam_builder.architecture.quantum_dots.components import VoltageGate, XYDrive
from quam_builder.architecture.quantum_dots.components.readout_resonator import (
    ReadoutResonatorIQ,
)
from quam_builder.architecture.quantum_dots.qpu import LossDiVincenzoQuam
from quam_builder.architecture.quantum_dots.qubit import LDQubit

# Compatibility shim for quam-builder feat/quantum_dots: ReadoutResonatorIQ.__post_init__
# expects opx_output, but InOutIQChannel defines opx_output_I/Q only.
if not hasattr(ReadoutResonatorIQ, "opx_output"):
    ReadoutResonatorIQ.opx_output = property(lambda self: self.opx_output_I)

# pylint: enable=wrong-import-position

# =============================================================================
# SECTION 1: Test Infrastructure Paths
# =============================================================================

TEST_ROOT = Path(__file__).resolve().parent
REPO_ROOT = None
for parent in [TEST_ROOT, *TEST_ROOT.parents]:
    if (parent / "qualibration_graphs").is_dir() and (parent / "tests").is_dir():
        REPO_ROOT = parent
        break
if REPO_ROOT is None:
    REPO_ROOT = TEST_ROOT.parents[0]

CREDENTIALS_PATH = Path(
    os.environ.get("QM_SAAS_CREDENTIALS_PATH", REPO_ROOT / "tests" / ".qm_saas_credentials.json")
)
ARTIFACTS_BASE = TEST_ROOT / "artifacts"


# =============================================================================
# SECTION 2: Custom Macros (copied from rabi_chevron_batched.py)
# =============================================================================


@quam_dataclass
class X180Macro(QuamMacro):  # pylint: disable=too-few-public-methods
    """Macro for X180 gate: step to operate point and apply pi pulse.

    This macro:
    1. Steps to the 'operate' voltage point (manipulation sweet spot)
    2. Applies the X180 (pi) pulse with variable duration

    Attributes:
        pulse_name: Name of the pulse operation to play (default: "X180")
        amplitude_scale: Optional amplitude scaling factor
    """

    pulse_name: str = "X180"
    amplitude_scale: float = None
    duration: int = None

    def _validate(self, xy_channel, duration, amplitude_scale) -> None:
        """Validate parameters for X180 gate execution.

        Raises:
            ValueError: If xy_channel is None or required parameters are missing.
        """
        if xy_channel is None:
            raise ValueError(
                "Cannot apply X180 gate: xy_channel is not configured on parent qubit."
            )

        missing = []
        if duration is None:
            missing.append("duration")
        if amplitude_scale is None:
            missing.append("amplitude_scale")
        if missing:
            raise ValueError(
                f"Missing required parameter(s): {', '.join(missing)}. "
                "Provide via kwargs or set as class attributes."
            )

    def apply(self, duration: int = None, **kwargs) -> None:
        """Execute X180 gate sequence.

        Args:
            duration: Pulse duration in clock cycles (4ns each)
        """
        parent_qubit = self.parent.parent
        amp_scale = kwargs.get("amplitude_scale", self.amplitude_scale)
        # Use positional arg if provided, otherwise check kwargs, then fall back to default
        if duration is None:
            duration = kwargs.get("duration", self.duration)

        self._validate(parent_qubit.xy_channel, duration, amp_scale)

        parent_qubit.xy_channel.play(
            self.pulse_name,
            amplitude_scale=amp_scale,
            duration=duration,
        )


@quam_dataclass
class MeasureMacro(QuamMacro):  # pylint: disable=too-few-public-methods
    """Macro for measurement with integrated voltage point navigation and thresholding.

    This macro:
    1. Steps to the 'readout' voltage point (PSB configuration)
    2. Performs demodulated measurement (I, Q)
    3. Thresholds the I component: state = I > threshold

    The threshold is retrieved from the sensor_dot's readout_thresholds
    dictionary, keyed by the quantum_dot_pair_id.

    Attributes:
        pulse_name: Name of the readout pulse operation (default: "readout")
        readout_duration: Hold duration at readout point (ns)
    """

    pulse_name: str = "readout"
    readout_duration: int = 2000

    def _validate(self, parent_qubit) -> None:
        """Validate that the qubit is properly configured for measurement.

        Raises:
            ValueError: If required components are missing or misconfigured.
        """
        if not parent_qubit.sensor_dots:
            raise ValueError("Cannot measure: no sensor_dots configured on parent qubit.")

        sensor_dot = parent_qubit.sensor_dots[0]

        if sensor_dot.readout_resonator is None:
            raise ValueError("Cannot measure: readout_resonator is not configured on sensor_dot.")

        if parent_qubit.quantum_dot is None:
            raise ValueError("Cannot measure: quantum_dot is not configured on parent qubit.")

        if parent_qubit.preferred_readout_quantum_dot is None:
            raise ValueError(
                "Cannot measure: preferred_readout_quantum_dot is not set on parent qubit."
            )

    def apply(self, **kwargs) -> QuaVariableBool:
        """Execute measurement sequence and return qubit state (parity).

        The measurement thresholds the I quadrature component to determine state.

        Returns:
            Boolean QUA variable indicating qubit state (True = I > threshold)
        """
        pulse = kwargs.get("pulse_name", self.pulse_name)

        # Navigate to qubit
        parent_qubit = self.parent.parent

        self._validate(parent_qubit)

        # Step to readout point (PSB configuration) - integrated into measure
        parent_qubit.step_to_point("measure", duration=self.readout_duration)

        # Get the associated sensor dot and quantum dot pair info
        sensor_dot = parent_qubit.sensor_dots[0]

        # Get the quantum_dot_pair_id for looking up threshold/projector
        qd_pair_id = parent_qubit.machine.find_quantum_dot_pair(
            parent_qubit.quantum_dot.id, parent_qubit.preferred_readout_quantum_dot
        )

        # Declare QUA variables for I and Q quadratures
        I = declare(fixed)
        Q = declare(fixed)

        # Wait for transients, then perform measurement
        sensor_dot.readout_resonator.wait(64)
        sensor_dot.readout_resonator.measure(
            pulse,
            qua_vars=(I, Q),
        )

        # Get threshold from sensor_dot (default to 0.0)
        threshold = sensor_dot.readout_thresholds.get(qd_pair_id, 0.0)

        # Threshold I component to get state
        state = declare(bool)
        assign(state, I > threshold)

        return state


# =============================================================================
# SECTION 3: Minimal QuAM Factory
# =============================================================================


def _create_minimal_machine() -> LossDiVincenzoQuam:
    """Create a machine configuration with 4 qubits in two pairs.

    Qubit pairs:
    - Q1 and Q2: virtual_dot_1 and virtual_dot_2, with virtual_sensor_1 (LF-FEM slot 2)
    - Q3 and Q4: virtual_dot_3 and virtual_dot_4, with virtual_sensor_2 (LF-FEM slot 3)

    Returns:
        Configured LossDiVincenzoQuam instance
    """
    # pylint: disable=unexpected-keyword-arg
    # Note: pylint doesn't understand quam_dataclass dynamic constructors
    machine = LossDiVincenzoQuam()

    controller = "con1"
    lf_fem_slot_1 = 2  # For qubit pair 1 (Q1, Q2)
    lf_fem_slot_2 = 3  # For qubit pair 2 (Q3, Q4)
    mw_fem_slot = 1

    # -------------------------------------------------------------------------
    # Physical Voltage Channels (4 plungers + 2 sensors)
    # Each pair uses its own LF-FEM slot
    # -------------------------------------------------------------------------
    plungers = {}
    # Pair 1: plungers 1 and 2 on LF-FEM slot 2
    for i in range(1, 3):
        plungers[i] = VoltageGate(
            id=f"plunger_{i}",
            opx_output=LFFEMAnalogOutputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_1,
                port_id=i,
                output_mode="direct",
            ),
            sticky=StickyChannelAddon(duration=16, digital=False),
        )
    # Pair 2: plungers 3 and 4 on LF-FEM slot 3
    for i in range(3, 5):
        plungers[i] = VoltageGate(
            id=f"plunger_{i}",
            opx_output=LFFEMAnalogOutputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_2,
                port_id=i - 2,  # ports 1 and 2
                output_mode="direct",
            ),
            sticky=StickyChannelAddon(duration=16, digital=False),
        )

    # Sensor DC channels
    sensor_dcs = {
        1: VoltageGate(
            id="sensor_DC_1",
            opx_output=LFFEMAnalogOutputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_1,
                port_id=3,
                output_mode="direct",
            ),
            sticky=StickyChannelAddon(duration=16, digital=False),
        ),
        2: VoltageGate(
            id="sensor_DC_2",
            opx_output=LFFEMAnalogOutputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_2,
                port_id=3,
                output_mode="direct",
            ),
            sticky=StickyChannelAddon(duration=16, digital=False),
        ),
    }

    # -------------------------------------------------------------------------
    # Readout Resonators (2 IQ resonators, one per qubit pair)
    # Each on its own LF-FEM slot
    # -------------------------------------------------------------------------
    readout_resonators = {
        1: ReadoutResonatorIQ(
            id="sensor_resonator_1",
            opx_output_I=LFFEMAnalogOutputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_1,
                port_id=4,
                output_mode="direct",
            ),
            opx_output_Q=LFFEMAnalogOutputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_1,
                port_id=5,
                output_mode="direct",
            ),
            opx_input_I=LFFEMAnalogInputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_1,
                port_id=1,
            ),
            opx_input_Q=LFFEMAnalogInputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_1,
                port_id=2,
            ),
            frequency_converter_up=FrequencyConverter(
                local_oscillator=LocalOscillator(frequency=5e9),
            ),
            intermediate_frequency=50e6,
            operations={
                "readout": pulses.SquareReadoutPulse(
                    length=1000,
                    amplitude=0.1,
                    integration_weights_angle=0.0,
                )
            },
        ),
        2: ReadoutResonatorIQ(
            id="sensor_resonator_2",
            opx_output_I=LFFEMAnalogOutputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_2,
                port_id=4,
                output_mode="direct",
            ),
            opx_output_Q=LFFEMAnalogOutputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_2,
                port_id=5,
                output_mode="direct",
            ),
            opx_input_I=LFFEMAnalogInputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_2,
                port_id=1,
            ),
            opx_input_Q=LFFEMAnalogInputPort(
                controller_id=controller,
                fem_id=lf_fem_slot_2,
                port_id=2,
            ),
            frequency_converter_up=FrequencyConverter(
                local_oscillator=LocalOscillator(frequency=5e9),
            ),
            intermediate_frequency=50e6,
            operations={
                "readout": pulses.SquareReadoutPulse(
                    length=1000,
                    amplitude=0.1,
                    integration_weights_angle=0.0,
                )
            },
        ),
    }

    # -------------------------------------------------------------------------
    # XY Drive Channels (4 channels, one per qubit)
    # -------------------------------------------------------------------------
    xy_drives = {}
    for i in range(1, 5):
        xy_drives[i] = XYDrive(
            id=f"Q{i}_xy",
            opx_output=MWFEMAnalogOutputPort(
                controller_id=controller,
                fem_id=mw_fem_slot,
                port_id=i,
                upconverter_frequency=5e9,
                band=2,
                full_scale_power_dbm=10,
            ),
            intermediate_frequency=100e6,
            add_default_pulses=True,
        )
        length = 100
        xy_drives[i].operations["X180"] = pulses.GaussianPulse(
            length=length, amplitude=0.2, sigma=length / 6
        )

    # -------------------------------------------------------------------------
    # Virtual Gate Set (all 4 dots + 2 sensors)
    # -------------------------------------------------------------------------
    machine.create_virtual_gate_set(
        virtual_channel_mapping={
            "virtual_dot_1": plungers[1],
            "virtual_dot_2": plungers[2],
            "virtual_dot_3": plungers[3],
            "virtual_dot_4": plungers[4],
            "virtual_sensor_1": sensor_dcs[1],
            "virtual_sensor_2": sensor_dcs[2],
        },
        gate_set_id="main_qpu",
        compensation_matrix=[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )

    # -------------------------------------------------------------------------
    # Register Channel Elements
    # -------------------------------------------------------------------------
    machine.register_channel_elements(
        plunger_channels=list(plungers.values()),
        sensor_resonator_mappings={
            sensor_dcs[1]: readout_resonators[1],
            sensor_dcs[2]: readout_resonators[2],
        },
        barrier_channels=[],
    )

    # -------------------------------------------------------------------------
    # Register Quantum Dot Pairs
    # -------------------------------------------------------------------------
    # Pair 1: dots 1 and 2, with sensor 1
    machine.register_quantum_dot_pair(
        quantum_dot_ids=["virtual_dot_1", "virtual_dot_2"],
        sensor_dot_ids=["virtual_sensor_1"],
        id="qd_pair_1_2",
    )

    # Pair 2: dots 3 and 4, with sensor 2
    machine.register_quantum_dot_pair(
        quantum_dot_ids=["virtual_dot_3", "virtual_dot_4"],
        sensor_dot_ids=["virtual_sensor_2"],
        id="qd_pair_3_4",
    )

    # -------------------------------------------------------------------------
    # Configure readout thresholds for sensor dots
    # -------------------------------------------------------------------------
    # pylint: disable=unsubscriptable-object
    sensor_dot_1 = machine.sensor_dots["virtual_sensor_1"]
    sensor_dot_1._add_readout_params(quantum_dot_pair_id="qd_pair_1_2", threshold=0.5)

    sensor_dot_2 = machine.sensor_dots["virtual_sensor_2"]
    sensor_dot_2._add_readout_params(quantum_dot_pair_id="qd_pair_3_4", threshold=0.5)
    # pylint: enable=unsubscriptable-object

    return machine, xy_drives


def _register_qubits_with_points(
    machine: LossDiVincenzoQuam,
    xy_drives: dict,
) -> List[LDQubit]:
    """Register 4 LDQubits with voltage points and custom macros.

    Qubit pairing for preferred readout:
    - Q1 uses virtual_dot_1, prefers readout via virtual_dot_2
    - Q2 uses virtual_dot_2, prefers readout via virtual_dot_1
    - Q3 uses virtual_dot_3, prefers readout via virtual_dot_4
    - Q4 uses virtual_dot_4, prefers readout via virtual_dot_3

    Args:
        machine: The configured machine instance
        xy_drives: Dictionary of XY drive channels keyed by qubit index (1-4)

    Returns:
        List of registered LDQubit instances
    """
    # Define qubit configurations: (qubit_name, dot_id, readout_dot_id, xy_index)
    qubit_configs = [
        ("Q1", "virtual_dot_1", "virtual_dot_2", 1),
        ("Q2", "virtual_dot_2", "virtual_dot_1", 2),
        ("Q3", "virtual_dot_3", "virtual_dot_4", 3),
        ("Q4", "virtual_dot_4", "virtual_dot_3", 4),
    ]

    qubits = []

    for qubit_name, dot_id, readout_dot_id, xy_idx in qubit_configs:
        # Register the qubit
        machine.register_qubit(
            qubit_name=qubit_name,
            quantum_dot_id=dot_id,
            xy_channel=xy_drives[xy_idx],
            readout_quantum_dot=readout_dot_id,
        )

        qubit = machine.qubits[qubit_name]  # pylint: disable=unsubscriptable-object
        # Ensure qubit.name is populated for stream tag uniqueness
        qubit.name = qubit_name

        # Define Voltage Points and create step macros
        qubit.add_point_with_step_macro(
            "empty",
            voltages={f"virtual_dot_{xy_idx}": -0.1},
            duration=500,
        )
        qubit.add_point_with_step_macro(
            "initialize",
            voltages={f"virtual_dot_{xy_idx}": 0.05},
            duration=500,
        )
        qubit.add_point(
            "measure",
            voltages={f"virtual_dot_{xy_idx}": -0.05},
        )

        # Register custom macros
        qubit.macros["x180"] = X180Macro(pulse_name="X180", amplitude_scale=1.0)
        qubit.macros["measure"] = MeasureMacro(
            pulse_name="readout",
            readout_duration=2000,
        )

        qubits.append(qubit)

    return qubits


@pytest.fixture
def minimal_quam_factory():
    """Factory fixture that creates a minimal LossDiVincenzoQuam with 4 qubits.

    Returns:
        A factory function that creates and returns a configured machine.
    """

    def _factory() -> LossDiVincenzoQuam:
        machine, xy_drives = _create_minimal_machine()
        _register_qubits_with_points(machine, xy_drives)
        # Set active qubits so get_qubits() returns them
        machine.active_qubit_names = list(machine.qubits.keys())

        # Add 'xy' property alias to each qubit (node uses qubit.xy but LDQubit has xy_channel)
        for qubit in machine.qubits.values():
            # Create a property-like access using __class__ modification
            qubit.__class__ = type(
                "LDQubitWithXY",
                (qubit.__class__,),
                {"xy": property(lambda self: self.xy_channel)},
            )

        return machine

    return _factory


# =============================================================================
# SECTION 4: Node Loader
# =============================================================================


@dataclass
class LoadedNode:
    """Container for a loaded calibration node module."""

    module: Any
    node: Any
    node_path: Path

    @property
    def description(self) -> str:
        """Return the node description."""
        return getattr(self.node, "description", "No description available")

    @property
    def parameters(self) -> Any:
        """Return the node parameters object."""
        return self.node.parameters

    def get_action(self, name: str) -> Callable:
        """Get an action function from the module by name."""
        return getattr(self.module, name)


@pytest.fixture
def node_loader():
    """Fixture that provides a function to dynamically load node modules.

    This loader handles the complexity of importing qualibration nodes:
    1. Adds the correct paths to sys.path for local imports
    2. Mocks Quam.load() to prevent loading from state file
    3. Patches ActionManager.run_action to skip auto-execution of actions

    Returns:
        A function that takes a node path and returns a LoadedNode.
    """
    import sys
    from dataclasses import dataclass as stdlib_dataclass
    from unittest.mock import patch, MagicMock

    def _load_node(node_path: Path) -> LoadedNode:
        if not node_path.exists():
            pytest.skip(f"Node file not found at {node_path}.")

        # Add the node's parent directories to sys.path for local imports
        # This handles imports like 'from quam_config import Quam' and
        # 'from calibration_utils.X import Y'
        node_dir = node_path.parent
        paths_to_add = []

        # Find the qualibration_graphs directory and add its subdirectory
        # e.g., qualibration_graphs/quantum_dots for quantum_dots nodes
        for parent in [node_dir, *node_dir.parents]:
            if parent.name == "qualibration_graphs":
                # Add the architecture-specific directory (e.g., quantum_dots)
                arch_dir = node_dir
                while arch_dir.parent.name != "qualibration_graphs":
                    arch_dir = arch_dir.parent
                if str(arch_dir) not in sys.path:
                    paths_to_add.append(str(arch_dir))
                break

        # Add paths to sys.path
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)

        spec = importlib.util.spec_from_file_location(node_path.stem, node_path)
        if spec is None or spec.loader is None:
            pytest.skip(f"Unable to load node module from {node_path}.")

        module = importlib.util.module_from_spec(spec)

        try:
            # Import required modules for patching
            from quam_config import Quam
            from qualibrate import QualibrationNode
            from qualibrate.runnables.run_action.action_manager import ActionManager

            mock_machine = MagicMock()

            # Create mock modes with external=True to help skip actions
            @stdlib_dataclass
            class MockModes:
                inspection: bool = False
                interactive: bool = False
                external: bool = True

            def patched_get_run_modes(cls, modes=None):
                return MockModes()

            # Patch ActionManager.run_action to skip auto-execution
            def no_op_run_action(self, action_name, node, *args, **kwargs):
                return None

            with patch.object(Quam, "load", return_value=mock_machine):
                with patch.object(QualibrationNode, "get_run_modes", classmethod(patched_get_run_modes)):
                    with patch.object(ActionManager, "run_action", no_op_run_action):
                        spec.loader.exec_module(module)

        except ImportError as exc:
            # If quam_config can't be imported, try loading without mock
            try:
                spec.loader.exec_module(module)
            except Exception as inner_exc:  # pragma: no cover - setup dependent
                pytest.skip(f"Failed to import node module: {inner_exc}")
        except Exception as exc:  # pragma: no cover - setup dependent
            pytest.skip(f"Failed to import node module: {exc}")

        return LoadedNode(module=module, node=module.node, node_path=node_path)

    return _load_node


# =============================================================================
# SECTION 5: QM SaaS Credentials
# =============================================================================


@pytest.fixture
def qm_saas_credentials():
    """Fixture that loads QM SaaS credentials from a JSON file.

    The credentials file should contain:
    - email: QM SaaS account email
    - password: QM SaaS account password
    - host (optional): QM SaaS host (defaults to qm-saas.dev.quantum-machines.co)

    Returns:
        Dictionary with credentials, or skips test if credentials unavailable.
    """
    if not CREDENTIALS_PATH.exists():
        pytest.skip(
            f"Missing QM SaaS credentials at {CREDENTIALS_PATH}. "
            "Create it with {email, password[, host]}."
        )

    with CREDENTIALS_PATH.open("r", encoding="utf-8") as handle:
        credentials = json.load(handle)

    email = credentials.get("email")
    password = credentials.get("password")
    host = credentials.get("host", "qm-saas.dev.quantum-machines.co")

    if not email or not password:
        pytest.skip("QM SaaS credentials file missing email or password.")

    return {"email": email, "password": password, "host": host}


# =============================================================================
# SECTION 6: Markdown Generator
# =============================================================================


@pytest.fixture
def markdown_generator():
    """Fixture that generates README.md documentation for a node.

    Returns:
        A function that generates and saves markdown documentation.
    """

    def _generate(
        loaded_node: LoadedNode,
        parameters_dict: Dict[str, Any],
        artifacts_dir: Path,
    ) -> Path:
        """Generate README.md with node documentation.

        Args:
            loaded_node: The loaded node instance
            parameters_dict: Dictionary of parameter names to values
            artifacts_dir: Directory to save the README.md

        Returns:
            Path to the generated README.md file
        """
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Build parameters table
        params_table = ["| Parameter | Value | Description |", "|-----------|-------|-------------|"]
        for name, value in parameters_dict.items():
            # Try to get docstring from parameter class
            doc = ""
            if hasattr(loaded_node.parameters, "__class__"):
                for cls in loaded_node.parameters.__class__.__mro__:
                    if hasattr(cls, "__annotations__") and name in cls.__annotations__:
                        # Try to get the docstring from the class
                        if hasattr(cls, "__pydantic_fields__"):
                            field_info = cls.__pydantic_fields__.get(name)
                            if field_info and field_info.description:
                                doc = field_info.description
                                break
            params_table.append(f"| `{name}` | `{value}` | {doc} |")

        params_section = "\n".join(params_table)

        # Build markdown content
        content = f"""# {loaded_node.node.name}

## Description

{loaded_node.description}

## Parameters

{params_section}

## Simulation Output

![Simulation](simulation.png)

---
*Generated by simulation test infrastructure*
"""

        output_path = artifacts_dir / "README.md"
        output_path.write_text(content, encoding="utf-8")
        return output_path

    return _generate


# =============================================================================
# SECTION 7: Simulation Plot Saver
# =============================================================================


@pytest.fixture
def save_simulation_plot():
    """Fixture that saves simulated samples to a PNG file.

    Returns:
        A function that saves the simulation plot.
    """

    def _save(simulated_samples, artifacts_dir: Path, title: str = "Simulated Samples") -> Path:
        """Save simulation samples plot to file.

        Args:
            simulated_samples: The simulated samples object from QM
            artifacts_dir: Directory to save the plot
            title: Title for the plot

        Returns:
            Path to the saved PNG file
        """
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        con_names = sorted(name for name in dir(simulated_samples) if name.startswith("con"))
        if not con_names:
            pytest.skip("No simulated analog connections found to plot.")

        con = getattr(simulated_samples, con_names[0])
        con.plot()
        plt.title(title)
        plt.tight_layout()

        output_path = artifacts_dir / "simulation.png"
        plt.savefig(output_path, dpi=200)
        plt.close()

        return output_path

    return _save


# =============================================================================
# SECTION 8: Simulation Test Context
# =============================================================================


@dataclass
class SimulationTestContext:
    """Combined context for simulation tests."""

    machine: LossDiVincenzoQuam
    loaded_node: LoadedNode
    credentials: Dict[str, str]
    artifacts_dir: Path

    # Default small sweep parameters for fast tests
    _small_sweep_params: Dict[str, Any] = field(default_factory=lambda: {
        "num_shots": 4,
        "min_wait_time_in_ns": 16,
        "max_wait_time_in_ns": 64,
        "time_step_in_ns": 16,
        "frequency_span_in_mhz": 4,
        "frequency_step_in_mhz": 2,
        "gap_wait_time_in_ns": 32,
    })

    def configure_small_sweep(self, custom_params: Optional[Dict[str, Any]] = None) -> None:
        """Configure node parameters for a small, fast sweep.

        Args:
            custom_params: Optional dictionary to override default small sweep parameters
        """
        params = {**self._small_sweep_params}
        if custom_params:
            params.update(custom_params)

        for key, value in params.items():
            if hasattr(self.loaded_node.parameters, key):
                setattr(self.loaded_node.parameters, key, value)

    def get_parameters_dict(self) -> Dict[str, Any]:
        """Get current parameter values as a dictionary.

        Returns:
            Dictionary of parameter names to their current values
        """
        params = {}
        for name in dir(self.loaded_node.parameters):
            if name.startswith("_"):
                continue
            try:
                val = getattr(self.loaded_node.parameters, name)
                if not callable(val):
                    params[name] = val
            except Exception:  # pylint: disable=broad-except
                pass
        return params


@pytest.fixture
def simulation_test_context(
    minimal_quam_factory,
    node_loader,
    qm_saas_credentials,
):
    """Combined fixture providing complete simulation test context.

    This fixture:
    1. Creates a minimal QuAM machine
    2. Loads the specified node module
    3. Injects the machine into the node
    4. Provides credentials and artifacts directory

    Returns:
        A factory function that creates SimulationTestContext.
    """

    def _create_context(node_path: Path, artifacts_subdir: str) -> SimulationTestContext:
        """Create a simulation test context.

        Args:
            node_path: Path to the node Python file
            artifacts_subdir: Subdirectory name under artifacts/ for outputs

        Returns:
            Configured SimulationTestContext
        """
        # Create machine and load node
        machine = minimal_quam_factory()
        loaded_node = node_loader(node_path)

        # Inject programmatic QuAM into the node
        # This overrides the Quam.load() that happens at module import time
        loaded_node.node.machine = machine

        # Set up artifacts directory
        artifacts_dir = ARTIFACTS_BASE / artifacts_subdir

        return SimulationTestContext(
            machine=machine,
            loaded_node=loaded_node,
            credentials=qm_saas_credentials,
            artifacts_dir=artifacts_dir,
        )

    return _create_context


def pytest_collection_modifyitems(config, items):
    """Ensure tests under tests/simulation are marked as simulation."""
    for item in items:
        if "tests/simulation/" in str(item.fspath) and not item.get_closest_marker("simulation"):
            item.add_marker(pytest.mark.simulation)

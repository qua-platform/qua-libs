# %% ---------------------------------------------------------------
# Refactored QUAM wiring/build script
# ---------------------------------------------------------------
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.units import unit
from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring, visualize
from qualang_tools.wirer.wirer.channel_specs import lf_fem_spec, mw_fem_spec

from quam.components import pulses
from quam_builder.architecture.superconducting.custom_gates import cross_resonance
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.superconducting import build_quam
from quam_builder.builder.superconducting.pulses import (
    add_DragCosine_pulses,
    add_default_transmon_pair_pulses,
)
from quam_config import Quam

# ---------------------------------------------------------------
# Settings / constants
# ---------------------------------------------------------------
DEFAULT_STATE_DIR = Path("/workspaces/qualibration_graphs/superconducting/quam_state")
GEN_SCRIPT_NAME = "generate_quam_iqcc.py"
HOST_IP = "10.1.1.6"
CLUSTER_NAME = "galil_arbel"

# Qubit pairs used for CR/ZZ wiring
QUBIT_PAIRS: List[Tuple[str, str]] = [
    ("A1", "A2"),
    ("A2", "A1"),
    ("A3", "A4"),
    ("A4", "A3"),
]

# Regex to parse things like "/ports/.../con{con}/{slot}/{port}"
PORT_PATH_RE = re.compile(r"/ports/\w+/con(\d+)/(\d+)/(\d+)")

# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Data models
# ---------------------------------------------------------------
@dataclass(frozen=True)
class PortSpec:
    con: int
    slot: int
    port: int


@dataclass
class QubitWires:
    rr_in_port: int
    rr_out_port: int
    rr_con: int
    rr_slot: int
    xy_con: int
    xy_slot: int
    xy_out_port: int
    z_con: int
    z_slot: int
    z_out_port: int


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def env_path(var: str, default: Path) -> Path:
    val = os.environ.get(var)
    return Path(val) if val else default


def parse_port_path(path_str: str) -> PortSpec:
    m = PORT_PATH_RE.search(path_str)
    if not m:
        raise ValueError(f"Unrecognized port path format: {path_str!r}")
    con, slot, port = map(int, m.groups())
    return PortSpec(con, slot, port)


def safe_load_json(p: Path) -> dict:
    with p.open("r") as f:
        return json.load(f)


def run_generator_script(state_dir: Path, script_name: str = GEN_SCRIPT_NAME) -> None:
    cfg_path = (state_dir / "../quam_config" / script_name).resolve()
    log.info("Running generator script: %s", cfg_path)
    subprocess.run(["python", str(cfg_path)], check=True)


def reset_state_dirs(state_dir: Path) -> None:
    """Wipe state and its copy; regenerate; then copy round-trip as in original flow."""
    copy_dir = (state_dir / "../quam_state_copy").resolve()
    for d in (state_dir, copy_dir):
        shutil.rmtree(d, ignore_errors=True)

    run_generator_script(state_dir)

    # Copy to backup then restore (matches original semantics)
    shutil.copytree(state_dir, copy_dir, dirs_exist_ok=True)
    shutil.rmtree(state_dir)
    shutil.copytree(copy_dir, state_dir, dirs_exist_ok=True)
    log.info("State regenerated and round-tripped via backup.")


def collect_wiring(latest_wiring: dict) -> Tuple[Instruments, Connectivity, Dict[str, QubitWires]]:
    connectivity = Connectivity()
    instruments = Instruments()

    mwfem_slots: Dict[int, Set[int]] = {1: set()}  # con -> {slots}
    lffem_slots: Dict[int, Set[int]] = {1: set()}
    wires: Dict[str, QubitWires] = {}

    qubits = latest_wiring["wiring"]["qubits"]
    for qb_key, elems in qubits.items():
        # skip coupler-type items (qC*)
        if qb_key.startswith("qC"):
            continue

        qb_name = qb_key[1:]  # strip leading 'q'
        rr_out = parse_port_path(elems["rr"]["opx_output"])
        rr_in = parse_port_path(elems["rr"]["opx_input"])
        xy_out = parse_port_path(elems["xy"]["opx_output"])
        z_out = parse_port_path(elems["z"]["opx_output"])

        # sanity: readout in/out must be same con/slot
        if (rr_in.con, rr_in.slot) != (rr_out.con, rr_out.slot):
            raise ValueError(f"{qb_key}.rr input/output controller/slot mismatch")

        mwfem_slots.setdefault(rr_out.con, set()).add(rr_out.slot)
        mwfem_slots.setdefault(xy_out.con, set()).add(xy_out.slot)
        lffem_slots.setdefault(z_out.con, set()).add(z_out.slot)

        wires[qb_name] = QubitWires(
            rr_in_port=rr_in.port,
            rr_out_port=rr_out.port,
            rr_con=rr_out.con,
            rr_slot=rr_out.slot,
            xy_con=xy_out.con,
            xy_slot=xy_out.slot,
            xy_out_port=xy_out.port,
            z_con=z_out.con,
            z_slot=z_out.slot,
            z_out_port=z_out.port,
        )

    # Register instruments
    for con, slots in mwfem_slots.items():
        instruments.add_mw_fem(controller=con, slots=sorted(slots))
    for con, slots in lffem_slots.items():
        instruments.add_lf_fem(controller=con, slots=sorted(slots))

    # Build connectivity per qubit
    for qb_name, w in wires.items():
        # RR (readout) bidirectional path
        connectivity.add_resonator_line(
            qubits=qb_name,
            constraints=mw_fem_spec(
                con=w.rr_con, slot=w.rr_slot, in_port=w.rr_in_port, out_port=w.rr_out_port
            ),
        )
        allocate_wiring(connectivity, instruments, block_used_channels=False)

        # XY (drive)
        connectivity.add_qubit_drive_lines(
            qubits=qb_name,
            constraints=mw_fem_spec(con=w.xy_con, slot=w.xy_slot, out_port=w.xy_out_port),
        )
        allocate_wiring(connectivity, instruments, block_used_channels=False)

        # Z (flux)
        connectivity.add_qubit_flux_lines(
            qubits=qb_name,
            constraints=lf_fem_spec(con=w.z_con, out_slot=w.z_slot, out_port=w.z_out_port),
        )
        allocate_wiring(connectivity, instruments, block_used_channels=False)

    # Two-qubit lines
    for qc, qt in QUBIT_PAIRS:
        kv = wires[qc]  # drive from control qubit's XY
        connectivity.add_qubit_pair_cross_resonance_lines(
            qubit_pairs=(qc, qt),
            constraints=mw_fem_spec(con=kv.xy_con, slot=kv.xy_slot, out_port=kv.xy_out_port),
        )
        allocate_wiring(connectivity, instruments, block_used_channels=False)

        connectivity.add_qubit_pair_zz_drive_lines(
            qubit_pairs=(qc, qt),
            constraints=mw_fem_spec(con=kv.xy_con, slot=kv.xy_slot, out_port=kv.xy_out_port),
        )
        allocate_wiring(connectivity, instruments, block_used_channels=False)

    return instruments, connectivity, wires


def visualize_wiring(connectivity: Connectivity, instruments: Instruments) -> None:
    visualize(connectivity.elements, available_channels=instruments.available_channels)
    plt.show(block=True)


def build_and_load_quam(connectivity: Connectivity) -> Quam:
    machine = Quam()
    build_quam_wiring(connectivity, HOST_IP, CLUSTER_NAME, machine)

    # reload & finalize QUAM object/state
    machine = Quam.load()
    machine.network.update(
        {
            "cloud": "true",
            "quantum_computer_backend": "arbel",
            "octave_ips": [],
            "octave_ports": [],
        }
    )
    build_quam(machine)
    return machine


def update_qubit_from_state(
    machine: Quam, latest_state: dict, wires: Dict[str, QubitWires], qubits_in_pairs: Set[str]
) -> List[str]:
    u = unit(coerce_to_integer=True)
    active_qubits: List[str] = []

    for _, qubit in machine.qubits.items():
        qb_short = qubit.name[1:]
        try:
            state_qubit = latest_state["qubits"][qubit.name]
            qubit.chi = state_qubit.get("chi")
            qubit.T1 = state_qubit.get("T1")
            qubit.T2ramsey = state_qubit.get("T2ramsey")
            qubit.phi0_current = state_qubit.get("phi0_current")
            qubit.phi0_voltage = state_qubit.get("phi0_voltage")
            qubit.grid_location = state_qubit.get("grid_location")
            qubit.freq_vs_flux_01_quad_term = state_qubit.get("freq_vs_flux_01_quad_term")

            # --- Resonator (rr)
            wr = wires[qb_short]
            rr = qubit.resonator
            rr_in = latest_state["ports"]["mw_inputs"][f"con{wr.rr_con}"][str(wr.rr_slot)][str(wr.rr_in_port)]
            rr_out = latest_state["ports"]["mw_outputs"][f"con{wr.rr_con}"][str(wr.rr_slot)][str(wr.rr_out_port)]

            rr.opx_input.band = rr_in.get("band")
            rr.opx_output.band = rr_out.get("band")
            rr.opx_output.upconverter_frequency = rr_out.get("upconverter_frequency")
            rr.opx_output.full_scale_power_dbm = rr_out.get("full_scale_power_dbm", -11)

            if rr_out.get("upconverter_frequency") != rr_in.get("downconverter_frequency"):
                rr.opx_input.downconverter_frequency = rr_in.get("downconverter_frequency")

            state_qubit_rr = state_qubit["resonator"]
            rr.thread = f"{qubit.name}_{rr.opx_output.controller_id}_slot{rr.opx_output.fem_id}"
            rr.time_of_flight = state_qubit_rr.get("time_of_flight")
            rr.confusion_matrix = state_qubit_rr.get("confusion_matrix")
            rr.RF_frequency = rr.LO_frequency + state_qubit_rr.get("intermediate_frequency")

            rr.operations["readout"].length = state_qubit_rr["operations"]["readout"].get("length")
            rr.operations["readout"].amplitude = state_qubit_rr["operations"]["readout"].get("amplitude")
            rr.operations["readout"].threshold = state_qubit_rr["operations"]["readout"].get("threshold", 0)
            rr.operations["readout"].rus_exit_threshold = state_qubit_rr["operations"]["readout"].get(
                "rus_exit_threshold", 0
            )
            rr.operations["readout"].integration_weights_angle = state_qubit_rr["operations"]["readout"].get(
                "integration_weights_angle", 0
            )

            # --- XY (drive)
            xy = qubit.xy
            xy_out = latest_state["ports"]["mw_outputs"][f"con{wr.xy_con}"][str(wr.xy_slot)][str(wr.xy_out_port)]
            xy.thread = f"{qubit.name}_{xy.opx_output.controller_id}_slot{xy.opx_output.fem_id}"
            xy.opx_output.band = xy_out.get("band")
            xy.opx_output.full_scale_power_dbm = xy_out.get("full_scale_power_dbm", -11)

            state_qubit_xy = state_qubit["xy"]
            if qb_short in qubits_in_pairs:
                # dual-upconverter setup (index 1 = fixed, 2 = CR/ZZ use)
                xy.opx_output.upconverters = {
                    1: {"frequency": xy_out.get("upconverter_frequency")},
                    2: {"frequency": 123.0},  # placeholder preserved from original
                }
                xy.upconverter = 1
                xy.RF_frequency = xy.opx_output.upconverters[1]["frequency"] + state_qubit_xy.get(
                    "intermediate_frequency"
                )
            else:
                xy.opx_output.upconverter_frequency = xy_out.get("upconverter_frequency")
                xy.RF_frequency = xy.upconverter_frequency + state_qubit_xy.get("intermediate_frequency")

            # saturation operation from state
            xy.operations["saturation"].length = state_qubit_xy["operations"]["saturation"]["length"]
            xy.operations["saturation"].amplitude = state_qubit_xy["operations"]["saturation"]["amplitude"]

            # Single-qubit gates (DragCosine defaults, then overwrite from state)
            add_DragCosine_pulses(
                qubit,
                amplitude=0.5,
                length=40,
                anharmonicity=250 * u.MHz,
                alpha=0.0,
                detuning=0,
            )
            xy.operations["x180_DragCosine"].length = state_qubit_xy["operations"]["x180_DragCosine"].get("length")
            xy.operations["x180_DragCosine"].amplitude = state_qubit_xy["operations"]["x180_DragCosine"].get(
                "amplitude", 1
            )
            xy.operations["x90_DragCosine"].length = state_qubit_xy["operations"]["x90_DragCosine"].get("length")
            xy.operations["x90_DragCosine"].amplitude = state_qubit_xy["operations"]["x90_DragCosine"].get(
                "amplitude", 1
            )

            # --- Z (flux)
            z = qubit.z
            z_out = latest_state["ports"]["analog_outputs"][f"con{wr.z_con}"][str(wr.z_slot)][str(wr.z_out_port)]
            z.opx_output.output_mode = z_out.get("output_mode")
            z.opx_output.upsampling_mode = z_out.get("upsampling_mode")

            state_qubit_z = state_qubit["z"]
            z.operations["const"].length = state_qubit_z["operations"]["const"]["length"]
            z.operations["const"].amplitude = state_qubit_z["operations"]["const"]["amplitude"]
            z.min_offset = state_qubit_z.get("min_offset")
            z.joint_offset = state_qubit_z.get("joint_offset")
            z.offset_settle_time = state_qubit_z.get("offset_settle_time")

            # Convenience constant pulses
            qubit.resonator.operations["const"] = pulses.SquarePulse(length=1 * u.us, amplitude=0.5, axis_angle=0.0)
            qubit.xy.operations["const"] = pulses.SquarePulse(length=1 * u.us, amplitude=0.5, axis_angle=0.0)

            active_qubits.append(qubit.name)

        except Exception as e:
            # Preserve original fallback behavior
            qubit.resonator.intermediate_frequency = 0
            qubit.xy.intermediate_frequency = 0
            log.warning("Failed updating %s: %s", qubit.name, e)

    return active_qubits


def configure_two_qubit_ops(machine: Quam) -> None:
    u = unit(coerce_to_integer=True)

    for _, qb_pair in machine.qubit_pairs.items():
        add_default_transmon_pair_pulses(qb_pair)

        qbt = qb_pair.qubit_target
        qc = qb_pair.qubit_control

        # Cross-resonance references
        qb_pair.cross_resonance.target_qubit_LO_frequency = f"#/qubits/{qbt.name}/xy/LO_frequency"
        qb_pair.cross_resonance.target_qubit_IF_frequency = f"#/qubits/{qbt.name}/xy/intermediate_frequency"
        qb_pair.cross_resonance.LO_frequency = f"#/qubits/{qbt.name}/xy/LO_frequency"
        qb_pair.cross_resonance.intermediate_frequency = f"#./inferred_intermediate_frequency"

        qb_pair.cross_resonance.core = f"{qc.name}_{qc.xy.opx_output.controller_id}_slot{qc.xy.opx_output.fem_id}"
        qc.xy.opx_output.upconverters.parent = None
        qc.xy.opx_output.upconverters[2]["frequency"] = qb_pair.cross_resonance.target_qubit_LO_frequency
        qb_pair.cross_resonance.upconverter = 2
        qb_pair.cross_resonance.opx_output.upconverter_frequency = None

        # ZZ drive references
        qb_pair.zz_drive.target_qubit_LO_frequency = f"#/qubits/{qbt.name}/xy/LO_frequency"
        qb_pair.zz_drive.target_qubit_IF_frequency = f"#/qubits/{qbt.name}/xy/intermediate_frequency"
        qb_pair.zz_drive.LO_frequency = f"#/qubits/{qbt.name}/xy/LO_frequency"
        qb_pair.zz_drive.detuning = -10 * u.MHz
        qb_pair.zz_drive.intermediate_frequency = f"#./inferred_intermediate_frequency"
        qb_pair.zz_drive.core = f"{qc.name}_{qc.xy.opx_output.controller_id}_slot{qc.xy.opx_output.fem_id}"

        # Macros & pulse library
        try:
            qb_pair.macros["cr"] = cross_resonance.CRGate(qc_correction_phase=0.0)

            # Square / Cosine / Gaussian variants on CR and target XY side
            qb_pair.cross_resonance.operations["square"] = pulses.SquarePulse(length=100, amplitude=1.0, axis_angle=0.0)
            qbt.xy.operations[f"cr_square_{qb_pair.name}"] = pulses.SquarePulse(length=100, amplitude=1.0, axis_angle=0.0)

            qb_pair.cross_resonance.operations["cosine"] = pulses.DragCosinePulse(
                length=100, amplitude=1.0, axis_angle=0.0, anharmonicity=260 * u.MHz, alpha=0.0, detuning=0
            )
            qbt.xy.operations[f"cr_cosine_{qb_pair.name}"] = pulses.DragCosinePulse(
                length=100, amplitude=1.0, axis_angle=0.0, anharmonicity=260 * u.MHz, alpha=0.0, detuning=0
            )

            qb_pair.cross_resonance.operations["gauss"] = pulses.DragGaussianPulse(
                length=100, sigma=20, amplitude=1.0, axis_angle=0.0, anharmonicity=260 * u.MHz, alpha=0.0, detuning=0
            )
            qbt.xy.operations[f"cr_gauss_{qb_pair.name}"] = pulses.DragGaussianPulse(
                length=100, sigma=20, amplitude=1.0, axis_angle=0.0, anharmonicity=260 * u.MHz, alpha=0.0, detuning=0
            )

            # Flattop sweep
            rise_fall = 8
            for flattop_len in [0, 20, 40, 60]:
                length = rise_fall + flattop_len + rise_fall
                qb_pair.cross_resonance.operations[f"flattop_{flattop_len:04d}"] = pulses.FlatTopGaussianPulse(
                    amplitude=1.0, length=length, flat_length=flattop_len, axis_angle=0.0
                )
                qbt.xy.operations[f"cr_flattop_{flattop_len:04d}"] = pulses.FlatTopGaussianPulse(
                    amplitude=1.0, length=length, flat_length=flattop_len, axis_angle=0.0
                )

        except Exception as e:
            log.warning("Two-qubit pulse configuration failed for %s: %s", qb_pair.name, e)


def save_outputs(machine: Quam, out_config: Path = Path("qua_config.json")) -> None:
    from pprint import pprint

    machine.save()
    cfg = machine.generate_config()
    pprint(cfg)
    with out_config.open("w+") as f:
        json.dump(cfg, f, indent=4)
    log.info("Saved QUAM state and QUA config to %s", out_config.resolve())


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main(state_dir: Path | None = None) -> None:
    state_dir = state_dir or env_path("QUAM_STATE_PATH", DEFAULT_STATE_DIR)
    log.info("Using state dir: %s", state_dir.resolve())

    # Regenerate state (match original behavior)
    reset_state_dirs(state_dir)

    # Load latest state & wiring snapshots
    latest_state = safe_load_json((state_dir / "state.json").resolve())
    latest_wiring = safe_load_json((state_dir / "wiring.json").resolve())

    # Build instruments/connectivity from wiring
    instruments, connectivity, wires = collect_wiring(latest_wiring)

    # Visualize
    visualize_wiring(connectivity, instruments)

    # Build QUAM and load machine
    machine = build_and_load_quam(connectivity)

    # Prepare qubit metadata
    qubits_in_pairs: Set[str] = set(q for pair in QUBIT_PAIRS for q in pair)

    # Update qubits from latest_state and wiring
    machine.active_qubit_names = update_qubit_from_state(
        machine=machine, latest_state=latest_state, wires=wires, qubits_in_pairs=qubits_in_pairs
    )

    # Configure two-qubit ops/pulses
    configure_two_qubit_ops(machine)

    # Persist results
    save_outputs(machine)


if __name__ == "__main__":
    main()

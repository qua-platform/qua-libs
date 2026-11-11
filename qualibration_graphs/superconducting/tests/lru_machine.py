"""Two-qubit MW-FEM + LF-FEM machines for the f-LRU tests, built in a temporary folder."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring  # noqa: E402
from qualang_tools.wirer.wirer.channel_specs import mw_fem_spec  # noqa: E402
from quam.core import quam_dataclass  # noqa: E402
from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuam, FluxTunableQuam  # noqa: E402
from quam_builder.builder.qop_connectivity import build_quam_wiring  # noqa: E402
from quam_builder.builder.superconducting import build_quam  # noqa: E402

QUBITS = ("q1", "q2")
# Typical frequencies: f_01, anharmonicity, f_r; f0-g1 = 2 * f_01 - |anharmonicity| - f_r is about 3 GHz.
FREQUENCIES = {"q1": (4.5735e9, -160.5e6, 5.91e9), "q2": (4.82e9, -200e6, 6.05e9)}


@quam_dataclass
class PlainFluxQuam(FluxTunableQuam):
    """Flux-tunable QUAM with the stock qubit class, as in states saved before the LRU classes."""


@quam_dataclass
class PlainFixedQuam(FixedFrequencyQuam):
    """Fixed-frequency QUAM with the stock qubit class."""


def build_machine(quam_class, state_dir: Path, monkeypatch, flux: bool = None):
    """Two qubits with their own MW-FEM drive port, frequencies set as after spectroscopy."""
    if flux is None:
        flux = issubclass(quam_class, FluxTunableQuam)
    monkeypatch.setenv("QUAM_STATE_PATH", str(state_dir))
    instruments = Instruments()
    instruments.add_mw_fem(controller=1, slots=[1])
    instruments.add_lf_fem(controller=1, slots=[3])
    connectivity = Connectivity()
    connectivity.add_resonator_line(qubits=[1, 2], constraints=mw_fem_spec(con=1, slot=1, in_port=1, out_port=1))
    connectivity.add_qubit_drive_lines(qubits=[1, 2])
    if flux:
        connectivity.add_qubit_flux_lines(qubits=[1, 2])
    allocate_wiring(connectivity, instruments)
    build_quam_wiring(connectivity, "127.0.0.1", "test_cluster", quam_class())
    machine = build_quam(quam_class.load())
    for name, (f_01, anharmonicity, f_r) in FREQUENCIES.items():
        qubit = machine.qubits[name]
        qubit.xy.RF_frequency = f_01
        qubit.anharmonicity = anharmonicity
        qubit.resonator.RF_frequency = f_r
    return machine


def snapshot(machine) -> str:
    """The machine's state as a string, to compare before and after a change."""
    return json.dumps(machine.to_dict(), sort_keys=True, default=str)

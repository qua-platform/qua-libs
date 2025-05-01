import shutil
import builtins
import matplotlib.pyplot
import importlib
import pytest
from quam.core import QuamRoot
from qualibrate import QualibrationLibrary


@pytest.fixture
def machine(qualibrate_calibrations_config, superconducting_folder, quam_state_path, monkeypatch, request):
    variant = getattr(request, "param", "lffem_mwfem")

    # Delete the folder at quam_state_path
    if quam_state_path.exists():
        shutil.rmtree(quam_state_path)
    assert not quam_state_path.exists()

    wiring_file = superconducting_folder / "quam_config" / "wiring_examples" / "wiring_lffem_mwfem.py"
    assert wiring_file.exists()

    # Monkeypatch matplotlib.pyplot.show to prevent blocking
    monkeypatch.setattr(matplotlib.pyplot, "show", lambda **kwargs: None)
    monkeypatch.setattr(builtins, "input", lambda *args: "y")

    # Execute the wiring script through import
    importlib.import_module(f"quam_config.wiring_examples.wiring_{variant}")

    assert quam_state_path.exists()

    return QuamRoot.load()  # type: ignore


@pytest.mark.parametrize(
    "machine",
    ["lffem_mwfem", "lffem_octave", "mwfem_cross_resonance", "opxp_external_mixers", "opxp_octave"],
    indirect=True,
)
def test_create_quam_variants(machine):
    pass


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node(machine, library: QualibrationLibrary):

    machine.network
    pass

import shutil
import builtins
import matplotlib.pyplot
import importlib
import pytest
from quam.core import QuamRoot
from qualibrate import QualibrationLibrary


@pytest.fixture
def machine(qualibrate_calibrations_config, env_vars, superconducting_folder, quam_state_path, monkeypatch, request):
    variant = getattr(request, "param", "lffem_mwfem")

    # Delete the folder at quam_state_path
    # if quam_state_path.exists():
    #     shutil.rmtree(quam_state_path)
    # assert not quam_state_path.exists()

    wiring_file = superconducting_folder / "quam_config" / "wiring_examples" / "wiring_lffem_mwfem.py"
    assert wiring_file.exists()

    # Monkeypatch matplotlib.pyplot.show to prevent blocking
    monkeypatch.setattr(matplotlib.pyplot, "show", lambda **kwargs: None)
    monkeypatch.setattr(builtins, "input", lambda *args: "y")

    # Execute the wiring script through import
    importlib.import_module(f"quam_config.wiring_examples.wiring_{variant}")

    assert quam_state_path.exists()

    if variant in ["lffem_mwfem"]:
        importlib.import_module("quam_config.populate_quam_lf_mw_fems")
    elif variant in ["opxp_octave"]:
        importlib.import_module("quam_config.populate_quam_opxp_octave")
    else:
        print(f"Variant {variant} not supported, skipping populate_quam script")

    machine = QuamRoot.load()  # type: ignore
    if env_vars is not None:
        try:
            machine.network["cluster_name"] = env_vars["network"][variant]["cluster_name"]
            machine.network["host"] = env_vars["network"][variant]["host"]
            machine.save()
            machine.active_qubit_names = ["q1", "q2"]
        except KeyError:
            print(f"Network config for variant {variant} not found in .env file, using default values")

    return machine


@pytest.mark.parametrize(
    "machine",
    ["lffem_mwfem", "lffem_octave", "mwfem_cross_resonance", "opxp_external_mixers", "opxp_octave"],
    indirect=True,
)
def test_create_quam_variants(machine):
    pass


def test_run_node_00_close_other_qms(qualibrate_calibrations_config, library: QualibrationLibrary):
    library.nodes["00_close_other_qms"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_00_hello_qua(qualibrate_calibrations_config, machine, library: QualibrationLibrary):
    library.nodes["00_hello_qua"].run()


# @pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
# def test_run_node_01a_mixer_calibration(qualibrate_calibrations_config, machine, library: QualibrationLibrary):
#     library.nodes["01a_mixer_calibration"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_01a_time_of_flight(qualibrate_calibrations_config, machine, library: QualibrationLibrary):
    library.nodes["01a_time_of_flight"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_01b_time_of_flight_mw_fem(qualibrate_calibrations_config, machine, library: QualibrationLibrary):
    library.nodes["01b_time_of_flight_mw_fem"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_02a_resonator_spectroscopy(qualibrate_calibrations_config, machine, library: QualibrationLibrary):
    library.nodes["02a_resonator_spectroscopy"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_02b_resonator_spectroscopy_vs_power(
    qualibrate_calibrations_config, machine, library: QualibrationLibrary
):
    library.nodes["02b_resonator_spectroscopy_vs_power"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_02c_resonator_spectroscopy_vs_flux(
    qualibrate_calibrations_config, machine, library: QualibrationLibrary
):
    library.nodes["02c_resonator_spectroscopy_vs_flux"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_03a_qubit_spectroscopy(qualibrate_calibrations_config, machine, library: QualibrationLibrary):
    library.nodes["03a_qubit_spectroscopy"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_03b_qubit_spectroscopy_vs_flux(qualibrate_calibrations_config, machine, library: QualibrationLibrary):
    library.nodes["03b_qubit_spectroscopy_vs_flux"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_04a_rabi_chevron(qualibrate_calibrations_config, machine, library: QualibrationLibrary):
    library.nodes["04a_rabi_chevron"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_04b_power_rabi(qualibrate_calibrations_config, machine, library: QualibrationLibrary):
    library.nodes["04b_power_rabi"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_05_T1(qualibrate_calibrations_config, machine, library: QualibrationLibrary):
    library.nodes["05_T1"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_06a_ramsey(qualibrate_calibrations_config, machine, library: QualibrationLibrary):
    library.nodes["06a_ramsey"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_06b_echo(qualibrate_calibrations_config, machine, library: QualibrationLibrary):
    library.nodes["06b_echo"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_07_iq_blobs(qualibrate_calibrations_config, machine, library: QualibrationLibrary):
    library.nodes["07_iq_blobs"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_08a_readout_frequency_optimization(
    qualibrate_calibrations_config, machine, library: QualibrationLibrary
):
    library.nodes["08a_readout_frequency_optimization"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_08b_readout_power_optimization(qualibrate_calibrations_config, machine, library: QualibrationLibrary):
    library.nodes["08b_readout_power_optimization"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_09_ramsey_vs_flux_calibration(qualibrate_calibrations_config, machine, library: QualibrationLibrary):
    library.nodes["09_ramsey_vs_flux_calibration"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_10b_drag_calibration_180_minus_180(
    qualibrate_calibrations_config, machine, library: QualibrationLibrary
):
    library.nodes["10b_drag_calibration_180_minus_180"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_11a_single_qubit_randomized_benchmarking(
    qualibrate_calibrations_config, machine, library: QualibrationLibrary
):
    library.nodes["11a_single_qubit_randomized_benchmarking"].run()


@pytest.mark.parametrize("machine", ["lffem_mwfem"], indirect=True)
def test_run_node_11b_single_qubit_randomized_benchmarking_interleaved(
    qualibrate_calibrations_config, machine, library: QualibrationLibrary
):
    library.nodes["11b_single_qubit_randomized_benchmarking_interleaved"].run()

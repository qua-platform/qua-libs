import os
import subprocess
from pathlib import Path
import pytest
import json
import tempfile
from qualibrate import QualibrationLibrary


def get_package_root(package_name="quam_config") -> Path:
    try:
        package = __import__(package_name)
        return Path(package.__file__).parent.parent
    except ImportError:
        raise Exception(f"Package {package_name} not found")


def run_cmd(command) -> str:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Command failed with error: {result.stderr}")
    return result.stdout


def setup_qualibrate_config(use_test_calibrations=True):
    print("Setting up qualibrate config")
    package_root = get_package_root()
    test_path = package_root / ".." / ".." / "tests"
    quam_state_path = test_path / "assets" / "quam_state"
    config_path = test_path / "assets" / "config.toml"
    env_path = package_root / ".." / ".." / "tests" / "assets" / ".env"

    if use_test_calibrations:
        calibration_library_folder = test_path / "assets" / "calibrations"
    else:
        calibration_library_folder = package_root / "calibrations"

    if "QUA_LIBS_STORAGE_LOCATION" in os.environ:
        print("Using QUA_LIBS_STORAGE_LOCATION from environment variable")
        storage_location = os.environ["QUA_LIBS_STORAGE_LOCATION"]
    elif env_path.exists():
        print("Using QUA_LIBS_STORAGE_LOCATION from .env file")
        file_contents = env_path.read_text()
        env_vars = json.loads(file_contents)
        storage_location = env_vars["QUA_LIBS_STORAGE_LOCATION"]
    else:

        # with tempfile.TemporaryDirectory() as temp_dir:
        #     print("Created temporary folder at:", temp_dir)
        #     storage_location = temp_dir
        raise Exception("QUA_LIBS_STORAGE_LOCATION is not set")

    command = [
        "qualibrate",
        "config",
        "--quam-state-path",
        f'"{quam_state_path}"',
        "--calibration-library-folder",
        f'"{calibration_library_folder}"',
        "--storage-location",
        f'"{storage_location}"',
        "--config-path",
        f'"{config_path}"',
        "--auto-accept",
    ]

    run_cmd(" ".join(command))

    os.environ["QUALIBRATE_CONFIG_FILE"] = str(config_path)
    os.environ["QUAM_CONFIG_FILE"] = str(config_path)
    return str(config_path)


@pytest.fixture(scope="session")
def qualibrate_test_config():
    setup_qualibrate_config(use_test_calibrations=True)


@pytest.fixture(scope="session")
def qualibrate_calibrations_config():
    setup_qualibrate_config(use_test_calibrations=False)


@pytest.fixture(scope="session")
def library():
    return QualibrationLibrary.get_active_library()


@pytest.fixture(scope="session")
def superconducting_folder():
    return Path("qualibration_graphs/superconducting").resolve()


@pytest.fixture(scope="session")
def quam_state_path():
    package_root = get_package_root()
    test_path = package_root / ".." / ".." / "tests"
    quam_state_path = test_path / "assets" / "quam_state"
    return quam_state_path


if __name__ == "__main__":
    setup_qualibrate_config()

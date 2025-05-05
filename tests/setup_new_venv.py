import subprocess
import sys
import shutil
from pathlib import Path


def run_tests():
    # Navigate to the working directory
    root_dir = Path(".").resolve()
    base_dir = Path("qualibration_graphs/superconducting").resolve()
    print(f"Using base directory: {base_dir}")
    venv_dir = base_dir / ".venv-test"

    # Delete existing .venv-test if it exists
    if venv_dir.exists() and venv_dir.is_dir():
        print("Removing existing .venv-test folder...")
        shutil.rmtree(venv_dir)

    # Create virtual environment
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)], cwd=base_dir)

    # Determine paths to pip and python inside the venv
    if sys.platform == "win32":
        python = venv_dir / "Scripts" / "python.exe"
        pip = venv_dir / "Scripts" / "pip.exe"
    else:
        python = venv_dir / "bin" / "python"
        pip = venv_dir / "bin" / "pip"

    # Upgrade pip and install dependencies
    subprocess.check_call([str(pip), "install", "--upgrade", "pip"], cwd=base_dir)
    subprocess.check_call([str(pip), "install", "-e", "."], cwd=base_dir)
    subprocess.check_call([str(pip), "install", "pytest"], cwd=base_dir)

    # Run tests using pytest
    subprocess.check_call([str(python), "-m", "pytest", "tests"], cwd=root_dir)


if __name__ == "__main__":
    run_tests()

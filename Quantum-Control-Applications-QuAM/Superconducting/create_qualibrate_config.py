from pathlib import Path
import subprocess
import sys


def get_input_with_default(prompt, default):
    user_input = input(f"{prompt} (default: {default}): ").strip()
    return user_input if user_input else default


config_supports_quam_state = False
try:
    import qualibrate_config

    config_supports_quam_state = True
except ImportError:
    pass  # Keep config_supports_quam_state as False if import fails

current_dir = Path(__file__).parent.absolute()

# Define default values
parameters = {
    "project": "QPU_project",
    "storage_location": str(current_dir / "data"),
    "calibration_library_folder": str(current_dir / "calibration_graph"),
    "quam_state_path": str(current_dir / "quam_state"),
}

# Ask if user wants to use all defaults
print("\nDefault values:")
for key, value in parameters.items():
    print(f"{key}: {value}")
msg = "\nUse all default values? (y/n):\nSelecting 'n' will allow you to customize each entry. (y/n): "
use_all_defaults = input(msg).strip().lower() == "y"

if not use_all_defaults:
    parameters["project"] = get_input_with_default("Enter project name", parameters["project"])
    parameters["storage_location"] = get_input_with_default("Enter storage location", parameters["storage_location"])
    parameters["calibration_library_folder"] = get_input_with_default(
        "Enter calibration library folder", parameters["calibration_library_folder"]
    )
    if config_supports_quam_state:
        parameters["quam_state_path"] = get_input_with_default("Enter QUAM state path", parameters["quam_state_path"])

# Build args list
config_args = [
    "qualibrate",
    "config",
    "--project",
    parameters["project"],
    "--storage-location",
    parameters["storage_location"],
    "--calibration-library-folder",
    parameters["calibration_library_folder"],
]

if config_supports_quam_state:
    config_args.extend(["--quam-state-path", parameters["quam_state_path"]])

subprocess.run(config_args)

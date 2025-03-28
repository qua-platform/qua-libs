from pathlib import Path
import subprocess
import sys
from colorama import init, Fore, Style

init(autoreset=True)


def get_input_with_default(prompt, default):
    user_input = input(
        f"{Fore.CYAN}{prompt}{Style.RESET_ALL} (default: {Fore.YELLOW}{default}{Style.RESET_ALL}): "
    ).strip()
    return user_input if user_input else default


config_supports_quam_state = False
try:
    import qualibrate_config

    config_supports_quam_state = True
except ImportError:
    print(
        f"{Fore.RED}Warning:{Style.RESET_ALL} 'qualibrate_config' not found. QUAM state configuration will be skipped."
    )

current_dir = Path(__file__).parent.absolute()

# Define default values
parameters = {
    "project": "QPU_project",
    "storage_location": str(current_dir.parent.absolute() / "data"),
    "calibration_library_folder": str(current_dir.parent.absolute() / "calibration_graph"),
    "quam_state_path": str(current_dir / "quam_state"),
}

# Display default values
print(f"\n{Fore.MAGENTA}Default values:{Style.RESET_ALL}")
for key, value in parameters.items():
    print(f"{Fore.BLUE}{key}{Style.RESET_ALL}: {Fore.GREEN}{value}{Style.RESET_ALL}")

print(f"\n{Fore.LIGHTBLACK_EX}Selecting 'n' will allow you to customize each entry.{Style.RESET_ALL}")
use_all_defaults = (
    input(f"{Fore.CYAN}Use all default values?{Style.RESET_ALL} {Fore.YELLOW}(y/n){Style.RESET_ALL} ").strip().lower()
    == "y"
)

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

print(f"\n{Fore.YELLOW}Running configuration command...{Style.RESET_ALL}")
result = subprocess.run(config_args)

if result.returncode == 0:
    print(f"{Fore.GREEN}Configuration completed successfully!{Style.RESET_ALL}")
else:
    print(f"{Fore.RED}Configuration failed with exit code {result.returncode}.{Style.RESET_ALL}")

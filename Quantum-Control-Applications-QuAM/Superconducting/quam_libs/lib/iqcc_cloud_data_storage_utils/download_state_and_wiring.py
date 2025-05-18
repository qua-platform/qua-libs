import os
import json
import logging
from iqcc_cloud_client import IQCC_Cloud

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_state_and_wiring(quantum_computer_backend: str = "arbel") -> None:
    """
    Download the latest state and wiring files from the quantum computer backend.
    
    Args:
        quantum_computer_backend (str): The name of the quantum computer backend to use.
    """
    try:
        logger.info(f"Connecting to quantum computer backend: {quantum_computer_backend}")
        qc = IQCC_Cloud(quantum_computer_backend=quantum_computer_backend)

        logger.info("Fetching latest wiring and state files")
        latest_wiring = qc.state.get_latest("wiring")
        latest_state = qc.state.get_latest("state")

        # Get the state folder path from environment variable
        quam_state_folder_path = os.environ["QUAM_STATE_PATH"]
        logger.info(f"State folder path: {quam_state_folder_path}")

        # Create the directory if it doesn't exist
        os.makedirs(quam_state_folder_path, exist_ok=True)
        logger.info("Created state directory if it didn't exist")

        # Save the files
        wiring_path = os.path.join(quam_state_folder_path, "wiring.json")
        state_path = os.path.join(quam_state_folder_path, "state.json")

        with open(wiring_path, "w") as f:
            json.dump(latest_wiring.data, f, indent=4)
        logger.info(f"Saved wiring file to: {wiring_path}")

        with open(state_path, "w") as f:
            json.dump(latest_state.data, f, indent=4)
        logger.info(f"Saved state file to: {state_path}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    download_state_and_wiring() 
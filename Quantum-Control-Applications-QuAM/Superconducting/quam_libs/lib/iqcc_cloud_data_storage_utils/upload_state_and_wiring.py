from iqcc_cloud_client import IQCC_Cloud
import os
import json
import logging

# Configure logging with purple color only for our logger
PURPLE = '\033[95m'
RESET = '\033[0m'

# Create a custom formatter for our logger
class PurpleFormatter(logging.Formatter):
    def format(self, record):
        record.msg = f"{PURPLE}{record.msg}{RESET}"
        return super().format(record)

# Configure root logger with default format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configure our specific logger with purple color
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(PurpleFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.propagate = False  # Prevent propagation to root logger

def save_quam_state_to_cloud(quam_state_folder_path: str = None, as_new_parent: bool = False):
    
    if quam_state_folder_path is None:
        if "QUAM_STATE_PATH" in os.environ:
            quam_state_folder_path = os.environ["QUAM_STATE_PATH"]
            logger.info(f"Using QUAM_STATE_PATH from environment: {quam_state_folder_path}")
        else:
            logger.error("QUAM_STATE_PATH environment variable is not set")
            raise ValueError("QUAM_STATE_PATH is not set")
    
    wiring_path = quam_state_folder_path + "/wiring.json"
    with open(wiring_path, "r") as f:
        wiring = json.load(f)
    
    quantum_computer_backend = wiring["network"]["quantum_computer_backend"]
    logger.info(f"Initializing IQCC_Cloud with backend: {quantum_computer_backend}")
    qc = IQCC_Cloud(quantum_computer_backend=quantum_computer_backend)
    
    if as_new_parent:
        logger.info("Pushing new wiring configuration as new parent")
        qc.state.push("wiring", wiring, comment="", parent_id=None)
    else:
        logger.info("Using existing parent for wiring configuration")
    
    latest_dataset = qc.state.get_latest("wiring")
    logger.info(f"Retrieved latest wiring dataset with ID: {latest_dataset.id}")
    
    state_path = quam_state_folder_path + "/state.json"
    logger.info(f"Reading state configuration from: {state_path}")
    with open(state_path, "r") as f:
        state = json.load(f)
    
    logger.info(f"Pushing state configuration with parent ID: {latest_dataset.id}")
    qc.state.push("state", state, comment="", parent_id=latest_dataset.id)
    logger.info("Done !")

if __name__ == "__main__":
    save_quam_state_to_cloud(as_new_parent=False)
   

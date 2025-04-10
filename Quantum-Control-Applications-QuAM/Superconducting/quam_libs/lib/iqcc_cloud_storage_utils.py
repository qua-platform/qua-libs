from iqcc_cloud_client import IQCC_Cloud
import os
import json

def save_quam_state_to_cloud(quam_state_folder_path: str = None, as_new_parent: bool = False):
    if quam_state_folder_path is None:
        if "QUAM_STATE_PATH" in os.environ:
            quam_state_folder_path = os.environ["QUAM_STATE_PATH"]
        else:
            raise ValueError("QUAM_STATE_PATH is not set")
    
    
    wiring_path = quam_state_folder_path + "/wiring.json"    
    with open(wiring_path, "r") as f:
        wiring = json.load(f)
    quantum_computer_backend = wiring["network"]["quantum_computer_backend"]
    qc = IQCC_Cloud(quantum_computer_backend=quantum_computer_backend)
    
    if as_new_parent:
        qc.state.push("wiring", wiring, comment="", parent_id=None)
    
    latest_dataset = qc.state.get_latest("wiring")
    state_path = quam_state_folder_path + "/state.json"
    with open(state_path, "r") as f:
        state = json.load(f)
    
    qc.state.push("state", state, comment="", parent_id=latest_dataset.id)
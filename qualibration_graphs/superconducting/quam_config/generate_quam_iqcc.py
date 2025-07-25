# %%
import os
import json
from iqcc_cloud_client import IQCC_Cloud


qc = IQCC_Cloud("arbel")

# Get the latest state and wiring
latest_wiring = qc.state.get_latest("wiring")
latest_state = qc.state.get_latest("state")

# Get the state folder path from environment variable
quam_state_folder_path = os.environ["QUAM_STATE_PATH"]

# Create the folder if it doesn't exist
os.makedirs(quam_state_folder_path, exist_ok=True)

# Save the files
with open(os.path.join(quam_state_folder_path, "wiring.json"), "w") as f:
    json.dump(latest_wiring.data, f, indent=4)

with open(os.path.join(quam_state_folder_path, "state.json"), "w") as f:
    json.dump(latest_state.data, f, indent=4)
# %%

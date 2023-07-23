import json
import quam_sdk.constructor

with open("quam_bootstrap_state.json", 'r') as file:
    state = json.load(file)

quam_sdk.constructor.quamConstructor(state, flat_data=False)
"""
Created on 11/11/2021
@author barnaby
"""

elements = {
    "LB": {
        "singleInput": {"port": ("con1", 1)},
        "hold_offset": {"duration": int(1e2)},
        "operations": {
            "jump": "jump",
            "ramp": "ramp",
        },
    },
    "RB": {
        "singleInput": {"port": ("con1", 2)},
        "hold_offset": {"duration": int(1e2)},
        "operations": {"jump": "jump", "ramp": "ramp"},
    },
    "RF": {
        "singleInput": {"port": ("con1", 3)},
        "time_of_flight": 304,
        "smearing": 0,
        "intermediate_frequency": 100e6,
        "outputs": {"out1": ("con1", 1)},
        "operations": {"measure": "measure"},
    },
}

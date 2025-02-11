from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from laboneq.contrib.example_helpers.generate_device_setup import (
    generate_device_setup_qubits,
)
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation

# Helpers:
# additional imports needed for Clifford gate calculation
from laboneq.contrib.example_helpers.randomized_benchmarking_helper import (
    clifford_parametrized,
    generate_play_rb_pulses,
    make_pauli_gate_map,
)

# LabOne Q:
from laboneq.simple import *
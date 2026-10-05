"""QUA helpers for playing compiled CAFE circuits."""

from qm.qua import assign

from .circuits import NUM_ANGLES_PER_QUBIT_LAYER, NUM_STATES

# Flat-array strides, matching the C-order layout of CafeCircuitAngles
QUBIT_STRIDE = NUM_ANGLES_PER_QUBIT_LAYER
LAYER_STRIDE = 2 * QUBIT_STRIDE
CIRCUIT_STRIDE = 2 * LAYER_STRIDE


def play_compiled_layer(qubit, angles, offset, angle_var) -> None:
    """Play ``Rz(a) · Ry(π/2) · Rz(b) · Ry(π/2) · Rz(c)`` on one qubit.

    ``angles[offset : offset + 3]`` holds (a, b, c) in units of 2π. Z rotations are virtual
    frame rotations; ``angle_var`` is a QUA fixed variable used to load each angle.
    """
    # Time order: Rz(c), y90, Rz(b), y90, Rz(a)
    for k, pulse in ((2, "y90"), (1, "y90"), (0, None)):
        assign(angle_var, angles[offset + k])
        qubit.xy.frame_rotation_2pi(angle_var)
        if pulse is not None:
            qubit.xy.play(pulse)


def preparation_offset(state_index):
    """Start of one state's preparation angles in the flat preparation array."""
    return state_index * CIRCUIT_STRIDE


def undo_offset(variant_index: int, depth_index, state_index, num_depths: int):
    """Start of one circuit's undo angles in the flat undo array."""
    return ((variant_index * num_depths + depth_index) * NUM_STATES + state_index) * CIRCUIT_STRIDE

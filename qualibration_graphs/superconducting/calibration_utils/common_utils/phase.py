def wrap_phase(phase):
    """Wrap a phase in 2π units to the [-0.5, 0.5) range (representing -π to π)."""
    return (phase + 0.5) % 1 - 0.5

from quam_builder.architecture.superconducting.components.flux_line import FluxLine


def test_class_attribute():
    flux_line = FluxLine(opx_output="")
    attrs = ["independent_offset", "joint_offset", "min_offset", "arbitrary_offset", "flux_point", "settle_time"]
    initial_values = [0.0, 0.0, 0.0, 0.0, "independent", None]
    for i, attr in enumerate(attrs):
        assert hasattr(flux_line, attr)
        assert getattr(flux_line, attr) == initial_values[i]

def test_class_methods():
    flux_line = FluxLine(opx_output="")
    assert flux_line.settle() is None

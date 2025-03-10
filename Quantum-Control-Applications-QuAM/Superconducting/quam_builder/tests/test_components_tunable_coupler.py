from quam_builder.architecture.superconducting.components.tunable_coupler import (
    TunableCoupler,
)


def test_class_attribute():
    coupler = TunableCoupler(opx_output="")
    attrs = [
        "decouple_offset",
        "interaction_offset",
        "arbitrary_offset",
        "flux_point",
        "settle_time",
    ]
    initial_values = [0.0, 0.0, 0.0, "off", None]
    for i, attr in enumerate(attrs):
        assert hasattr(coupler, attr)
        assert getattr(coupler, attr) == initial_values[i]


def test_class_methods():
    coupler = TunableCoupler(opx_output="")
    assert coupler.settle() is None

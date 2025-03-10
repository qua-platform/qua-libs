from quam_builder.architecture.superconducting.components.readout_resonator import (
    ReadoutResonatorMW,
    ReadoutResonatorIQ,
)

ALL_RES = [
    ReadoutResonatorIQ(
        opx_input_I="",
        opx_input_Q="",
        opx_output_I="",
        opx_output_Q="",
        frequency_converter_up="",
    ),
    ReadoutResonatorMW(opx_input="", opx_output=""),
]


def test_class_attribute():
    attrs = [
        "depletion_time",
        "frequency_bare",
        "f_01",
        "f_12",
        "confusion_matrix",
        "gef_centers",
        "gef_confusion_matrix",
        "GEF_frequency_shift",
    ]
    initial_values = [16, None, None, None, None, None, None, None]
    for res in ALL_RES:
        for i, attr in enumerate(attrs):
            assert hasattr(res, attr)
            assert getattr(res, attr) == initial_values[i]


def test_class_methods():
    for res in ALL_RES:
        assert res.calculate_voltage_scaling_factor(0, 6) == 1.9952623149688795

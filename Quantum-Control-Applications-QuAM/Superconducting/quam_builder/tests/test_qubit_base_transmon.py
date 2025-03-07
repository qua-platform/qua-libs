import pytest
from quam.components.channels import IQChannel
from quam_builder.architecture.superconducting.qubit.base_transmon import BaseTransmon


def test_class_attribute():
    transmon = BaseTransmon(id="q1")
    attrs = ["xy", "resonator", "f_01", "f_12", "anharmonicity", "T1", "T2ramsey", "T2echo",
             "thermalization_time_factor", "sigma_time_factor", "GEF_frequency_shift", "chi", "grid_location"]
    initial_values = [None, None, None, None, None, None, None, None, 5, 5, None, None, None]
    for i, attr in enumerate(attrs):
        assert hasattr(transmon, attr)
        assert getattr(transmon, attr) == initial_values[i]

def test_name():
    for name in ["AbCd", "q12", 0, 51]:
        transmon = BaseTransmon(id=name)
        if type(name) == str:
            assert transmon.name == name
        else:
            assert transmon.name == f"q{name}"

def test_inferred_frequencies():
    transmon = BaseTransmon(id=1)
    with pytest.raises(AttributeError):
        print(transmon.inferred_f_12)
    with pytest.raises(AttributeError):
        print(transmon.inferred_anharmonicity)

    transmon.f_01 = 6e9
    with pytest.raises(AttributeError):
        print(transmon.inferred_f_12)
    with pytest.raises(AttributeError):
        print(transmon.inferred_anharmonicity)

    transmon.anharmonicity = 50e6
    transmon.f_12 = transmon.inferred_f_12
    print(transmon.inferred_anharmonicity)

def test_thermalization_time():
    transmon = BaseTransmon(id=1)
    assert transmon.thermalization_time == 50000
    transmon.T1 = 120e-9
    assert transmon.thermalization_time == 600

def test_set_gate_shape():
    transmon = BaseTransmon(id=1)
    with pytest.raises(AttributeError):
        transmon.set_gate_shape("not_registered_shape")
    transmon.xy = IQChannel(
        opx_output_I="{wiring_path}/opx_output_I",
        opx_output_Q="{wiring_path}/opx_output_Q",
        frequency_converter_up="{wiring_path}/frequency_converter_up",
        intermediate_frequency=-200e6,
    )
    for gate in ["x180", "x90", "-x90", "y180", "y90", "-y90"]:
        transmon.xy.operations[f"{gate}_registered_shape"] = "registered_shape"
    transmon.set_gate_shape("registered_shape")

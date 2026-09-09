"""Compile fixed-length measurements with matching integration weights."""

from dataclasses import replace

import xarray as xr
from qm.qua import align, declare, declare_stream, fixed, for_, program, save, stream_processing
from qualang_tools.units import unit
from qualibration_libs.parameters import get_qubits
from quam.components.pulses import SquareReadoutPulse

from .parameters import duration_values


def duration_pulse(qubit, operation, duration):
    source = qubit.resonator.operations.get(operation, qubit.resonator.operations["readout"])
    if type(source) is not SquareReadoutPulse:
        raise ValueError(f"{qubit.name}: duration optimization requires a SquareReadoutPulse")
    if source.get_raw_value("integration_weights") != "#./default_integration_weights":
        raise ValueError(f"{qubit.name}: recalibrate with default integration weights before sweeping duration")
    return replace(
        source, id=None, length=int(duration), integration_weights="#./default_integration_weights",
        threshold=None, rus_exit_threshold=None,
    )


def create_qua_program(node):
    durations = duration_values(node.parameters)
    gef = node.parameters.operation == "readout_GEF"
    states = "gef" if gef else "ge"
    node.namespace["qubits"] = qubits = get_qubits(node)
    if not len(qubits):
        raise ValueError("Select at least one qubit")
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray(range(1, node.parameters.num_shots + 1), attrs={"long_name": "number of shots"}),
        "duration": xr.DataArray(durations, attrs={"long_name": "readout duration", "units": "ns"}),
    }
    u = unit(coerce_to_integer=True)
    temporary_operations = []
    operation_names = {}
    try:
        for qubit in qubits:
            if gef and "EF_x180" not in qubit.xy.operations:
                raise ValueError(f"{qubit.name}: calibrate EF_x180 before running this node")
            for duration in durations:
                name = f"_duration_opt_{int(duration)}"
                if name in qubit.resonator.operations:
                    raise ValueError(f"{qubit.name}: temporary operation {name} already exists")
                qubit.resonator.operations[name] = duration_pulse(qubit, node.parameters.operation, duration)
                temporary_operations.append((qubit.resonator, name))
                operation_names[int(duration)] = name

        with program() as node.namespace["qua_program"]:
            n = declare(int)
            n_st = declare_stream()
            iq = {state: [(declare(fixed), declare(fixed)) for _ in qubits] for state in states}
            streams = {state: [(declare_stream(), declare_stream()) for _ in qubits] for state in states}
            for batch in qubits.batch():
                for qubit in batch.values():
                    node.machine.initialize_qpu(target=qubit)
                    if gef:
                        qubit.resonator.update_frequency(
                            qubit.resonator.intermediate_frequency + (qubit.resonator.GEF_frequency_shift or 0)
                        )
                align()
                with for_(n, 0, n < node.parameters.num_shots, n + 1):
                    save(n, n_st)
                    # Python unrolling selects pulses whose integration weights match each length.
                    for duration in durations:
                        for state in states:
                            for qubit in batch.values():
                                qubit.wait((2 if gef else 1) * qubit.thermalization_time * u.ns)
                            align()
                            for i, qubit in batch.items():
                                if state in "ef":
                                    qubit.xy.play("x180")
                                if state == "f":
                                    qubit.xy.update_frequency(qubit.xy.intermediate_frequency - qubit.anharmonicity)
                                    qubit.xy.play("EF_x180")
                                    qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                                qubit.align()
                                qubit.resonator.measure(operation_names[int(duration)], qua_vars=iq[state][i])
                                qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                                for value, stream in zip(iq[state][i], streams[state][i]):
                                    save(value, stream)
                            align()
                for qubit in batch.values():
                    if gef:
                        qubit.resonator.update_frequency(qubit.resonator.intermediate_frequency)
                align()
            with stream_processing():
                n_st.save("n")
                for state in states:
                    for i in range(len(qubits)):
                        for quadrature, stream in zip("IQ", streams[state][i]):
                            stream.buffer(len(durations)).buffer(node.parameters.num_shots).save(
                                f"{quadrature}{state}{i + 1}"
                            )
        node.namespace["config"] = node.machine.generate_config()
    finally:
        # Sweep pulses are config-only; never persist them into QUAM state.
        for resonator, name in temporary_operations:
            del resonator.operations[name]


def update_state(node):
    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            result = node.results["fit_results"][qubit.name]
            if not result["success"]:
                continue
            duration = result["optimal_duration"]
            operation_name = node.parameters.operation
            if operation_name not in qubit.resonator.operations:
                qubit.resonator.operations[operation_name] = duration_pulse(qubit, operation_name, duration)
            operation = qubit.resonator.operations[operation_name]
            operation.length = duration
            if operation_name == "readout_GEF":
                qubit.resonator.gef_centers = (
                    node.results["ds_iq_blobs"].sel(qubit=qubit.name).centers.values * duration / 2**12
                ).tolist()
            else:
                operation.integration_weights_angle -= result["iw_angle"]
                operation.threshold = result["ge_threshold"] * duration / 2**12
                operation.rus_exit_threshold = result["rus_threshold"] * duration / 2**12
                qubit.resonator.confusion_matrix = result["confusion_matrix"]

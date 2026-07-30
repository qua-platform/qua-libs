"""Local no-hardware tests for timestamp_tools.py."""

from pathlib import Path

import numpy as np
import pytest
from qm import QopCaps
from qm.qua import declare, declare_stream, for_, if_, measure, play, program, save, stream_processing, strict_timing_

import timestamp_tools
from timestamp_tools import TimestampRecorder


class FakeResultHandle:
    def __init__(self, values):
        self.values = values

    def fetch_all(self):
        return self.values


class FakeResultHandles:
    def __init__(self, values_by_handle):
        self.values_by_handle = values_by_handle
        self.waited = False

    def wait_for_all_values(self):
        self.waited = True

    def get(self, name):
        value = self.values_by_handle.get(name)
        return None if value is None else FakeResultHandle(value)


class FakeJob:
    def __init__(self, values_by_handle):
        self.result_handles = FakeResultHandles(values_by_handle)


def build_natural_program():
    with program() as qua_program:
        n = declare(int)
        n_st = declare_stream()

        with for_(n, 0, n < 2, n + 1):
            play("x180", "qubit")
            with if_(n == 0):
                measure("readout", "resonator")

        with strict_timing_():
            play("x90", "qubit")

        save(n, n_st)
        with stream_processing():
            n_st.save("existing_result")

    return qua_program


def test_import_resolves_to_the_local_prototype():
    expected = Path(__file__).with_name("timestamp_tools.py").resolve()
    assert Path(timestamp_tools.__file__).resolve() == expected


def test_completed_program_is_instrumented_without_changing_qua_body():
    natural_program = build_natural_program()
    original_proto = natural_program.qua_program.SerializeToString()

    timing = TimestampRecorder(natural_program)

    assert natural_program.qua_program.SerializeToString() == original_proto
    assert timing.program is not natural_program
    assert timing.names == (
        "play_0_x180_qubit",
        "measure_1_readout_resonator",
        "play_2_x90_qubit",
    )
    assert timing.handles == {
        "play_0_x180_qubit": "qua_timestamps_0",
        "measure_1_readout_resonator": "qua_timestamps_1",
        "play_2_x90_qubit": "qua_timestamps_2",
    }

    serialized_program = str(timing.program.qua_program)
    assert serialized_program.count("timestampLabel") == 3
    assert all(handle in serialized_program for handle in timing.handles.values())
    assert "existing_result" in serialized_program
    assert QopCaps.command_timestamps in timing.program.used_capabilities


def test_fetch_converts_cycles_and_preserves_loop_occurrences():
    timing = TimestampRecorder(build_natural_program())
    job = FakeJob(
        {
            timing.handles[timing.names[0]]: np.array([10, 30]),
            timing.handles[timing.names[1]]: {"value": np.array([20])},
            timing.handles[timing.names[2]]: np.array([40]),
        }
    )

    result = timing.fetch(job)

    assert job.result_handles.waited
    np.testing.assert_array_equal(result[0].clock_cycles, [10, 30])
    np.testing.assert_array_equal(result[0].nanoseconds, [40, 120])
    np.testing.assert_array_equal(result[1].nanoseconds, [80])
    np.testing.assert_array_equal(result[2].nanoseconds, [160])
    assert result[0].occurrences == 2
    assert result.names == timing.names


def test_relative_results_and_sorted_rows_use_automatic_index():
    timing = TimestampRecorder(build_natural_program())
    job = FakeJob(
        {
            timing.handles[timing.names[0]]: np.array([30, 10]),
            timing.handles[timing.names[1]]: np.array([20]),
            timing.handles[timing.names[2]]: np.array([40]),
        }
    )

    result = timing.fetch(job)
    relative = result.relative_to(0, occurrence=1)
    rows = result.as_rows(reference=0, reference_occurrence=1)

    np.testing.assert_array_equal(relative[timing.names[0]], [80, 0])
    np.testing.assert_array_equal(relative[timing.names[1]], [40])
    np.testing.assert_array_equal(relative[timing.names[2]], [120])
    assert [row["name"] for row in rows] == [
        timing.names[0],
        timing.names[1],
        timing.names[0],
        timing.names[2],
    ]
    assert [row["relative_ns"] for row in rows] == [0, 40, 80, 120]
    assert all(row["statement_path"].startswith("program.script.body") for row in rows)


def test_existing_manual_timestamp_is_rejected():
    with program() as manually_timestamped_program:
        play("x180", "qubit", timestamp_stream="manual")

    with pytest.raises(ValueError, match="already has timestamp stream"):
        TimestampRecorder(manually_timestamped_program)


def test_program_must_be_complete():
    program_context = program()
    with program_context as in_scope_program:
        with pytest.raises(RuntimeError, match="Finish the QUA program context"):
            TimestampRecorder(in_scope_program)


def test_loop_indexing_maps_nested_sweep_point():
    from qualang_tools.loops import from_array

    with program() as qua_program:
        n = declare(int)
        a = declare(fixed)
        n_rabi = declare(int)
        n2 = declare(int)

        with for_(n, 0, n < 2, n + 1):
            with for_(*from_array(n_rabi, np.array([1, 3], dtype=int))):
                with for_(*from_array(a, np.linspace(0.9, 1.0, 2))):
                    with for_(n2, 0, n2 < n_rabi, n2 + 1):
                        play("x180", "qubit")
                    measure("readout", "resonator")

    timing = TimestampRecorder(qua_program)
    measure_name = timing.names[1]
    play_name = timing.names[0]

    assert timing.loop_mapper.expected_occurrences(measure_name) == 8
    assert timing.loop_mapper.expected_occurrences(play_name) == 16
    assert timing.loop_mapper.coords_to_flat(measure_name, {0: 1, 1: 1, 2: 1}) == 7

    job = FakeJob(
        {
            timing.handles[play_name]: np.arange(16),
            timing.handles[measure_name]: np.arange(100, 108),
        }
    )
    result = timing.fetch(job)
    shot = result.select_shot({0: 1, 1: 1, 2: 1}, reference=measure_name)

    assert shot[measure_name]["occurrence"] == 7
    assert shot[measure_name]["time_ns"] == 107.0
    np.testing.assert_array_equal(shot[play_name]["clock_cycles"], [13, 14, 15])


def test_missing_result_handle_is_reported():
    timing = TimestampRecorder(build_natural_program())
    job = FakeJob({timing.handles[timing.names[0]]: np.array([10])})

    with pytest.raises(KeyError, match=timing.handles[timing.names[1]]):
        timing.fetch(job)

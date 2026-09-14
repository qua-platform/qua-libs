"""Lightweight analysis unit tests for 01a_mixer_calibration helpers.

Covers extract / log / plot key handling for sensor-only and qubit-only
calibration payloads (no Octave hardware required).
"""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from calibration_utils.mixer_calibration.analysis import (
    extract_relevant_fit_parameters,
    log_fitted_results,
)
from calibration_utils.mixer_calibration.plotting import plot_all

SENSOR_NAME = "virtual_sensor_1"
QUBIT_NAME = "q1"


def _make_node(calibration_results: dict) -> SimpleNamespace:
    return SimpleNamespace(
        parameters=SimpleNamespace(calibrate_resonator=True, calibrate_drive=True),
        namespace={
            "calibration_results": calibration_results,
            "sensors": [SimpleNamespace(name=SENSOR_NAME)],
            "qubits": [SimpleNamespace(name=QUBIT_NAME)],
        },
    )


@pytest.mark.analysis
def test_01a_mixer_extract_log_handles_sensor_and_qubit_channels():
    """Missing resonator/xy_drive keys must not raise; success follows present metrics."""
    node = _make_node(
        {
            SENSOR_NAME: {"resonator": None},
            QUBIT_NAME: {"xy_drive": None},
        }
    )

    fits = extract_relevant_fit_parameters(node)
    assert SENSOR_NAME in fits and QUBIT_NAME in fits
    assert fits[SENSOR_NAME].resonator is None
    assert fits[SENSOR_NAME].xy_drive is None
    assert fits[SENSOR_NAME].success is False
    assert fits[QUBIT_NAME].resonator is None
    assert fits[QUBIT_NAME].xy_drive is None
    assert fits[QUBIT_NAME].success is False

    logs: list[str] = []
    log_fitted_results({k: asdict(v) for k, v in fits.items()}, log_callable=logs.append)
    assert len(logs) == 2
    assert all("FAIL" in line for line in logs)
    assert all(name in "".join(logs) for name in (SENSOR_NAME, QUBIT_NAME))


@pytest.mark.analysis
def test_01a_mixer_extract_success_when_channel_metrics_present():
    """A present calibration payload that yields metrics marks the element successful."""
    fake_cal = object()
    node = _make_node({SENSOR_NAME: {"resonator": fake_cal}, QUBIT_NAME: {"xy_drive": None}})

    with patch("calibration_utils.mixer_calibration.analysis.CalibrationResultPlotter") as plotter_cls:
        plotter = plotter_cls.return_value
        plotter.get_lo_leakage_rejection.return_value = 40.0
        plotter.get_image_rejection.return_value = 35.0

        fits = extract_relevant_fit_parameters(node)

    assert fits[SENSOR_NAME].success is True
    assert fits[SENSOR_NAME].resonator == {
        "lo_leakage": 40.0,
        "image_rejection": 35.0,
    }
    assert fits[QUBIT_NAME].success is False


@pytest.mark.analysis
def test_01a_mixer_plot_all_uses_element_names_and_skips_missing_channels():
    """plot_all keys figures by element name and ignores missing resonator/xy_drive."""
    node = _make_node(
        {
            SENSOR_NAME: {"resonator": None},
            QUBIT_NAME: {"xy_drive": None},
        }
    )

    figures = plot_all(node)
    assert set(figures) == {SENSOR_NAME, QUBIT_NAME}
    assert "qubit.name" not in figures
    assert figures[SENSOR_NAME] == {}
    assert figures[QUBIT_NAME] == {}


@pytest.mark.analysis
def test_01a_mixer_plot_all_builds_figures_for_present_channel():
    """When a channel payload exists, plot_all stores lo_leakage/image_rejection figures."""
    fake_cal = object()
    node = _make_node(
        {
            SENSOR_NAME: {"resonator": fake_cal},
            QUBIT_NAME: {},
        }
    )

    fake_lo = MagicMock()
    fake_lo._suptitle.get_text.return_value = "LO"
    fake_img = MagicMock()
    fake_img._suptitle.get_text.return_value = "IMG"

    with patch("calibration_utils.mixer_calibration.plotting.CalibrationResultPlotter") as plotter_cls:
        plotter = plotter_cls.return_value
        plotter.show_lo_leakage_calibration_result.return_value = fake_lo
        plotter.show_image_rejection_calibration_result.return_value = fake_img

        figures = plot_all(node)

    assert SENSOR_NAME in figures
    assert "resonator" in figures[SENSOR_NAME]
    assert figures[SENSOR_NAME]["resonator"]["lo_leakage"] is fake_lo
    assert figures[SENSOR_NAME]["resonator"]["image_rejection"] is fake_img
    assert figures[QUBIT_NAME] == {}

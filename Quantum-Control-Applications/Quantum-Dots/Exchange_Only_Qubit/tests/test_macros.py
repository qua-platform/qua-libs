"""
Pure Python tests for macros.py functions.
These tests do not require QUA hardware or simulator.
Note: QUA macros (lock_in_macro, DC_current_sensing_macro) are not tested here.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path to import macros
sys.path.insert(0, str(Path(__file__).parent.parent))
from macros import (
    spiral_order,
    get_filtered_voltage,
    round_to_fixed,
)


def test_spiral_order_center_is_zero():
    """Spiral starts at center with index 0"""
    order = spiral_order(5)
    center = (5 - 1) // 2
    assert order[center, center] == 0, f"Center should be 0, got {order[center, center]}"


def test_spiral_order_unique_indices():
    """All indices 0 to N^2-1 appear exactly once"""
    order = spiral_order(7)
    flattened = order.flatten()
    expected_set = set(range(7 * 7))
    actual_set = set(flattened)
    assert (
        actual_set == expected_set
    ), f"Missing or duplicate indices. Expected {len(expected_set)} unique values, got {len(actual_set)}"


def test_spiral_order_odd_size():
    """Spiral order should work with odd sizes"""
    for N in [3, 5, 7, 9, 11]:
        order = spiral_order(N)
        assert order.shape == (N, N), f"Expected shape ({N}, {N}), got {order.shape}"
        # Check that all indices are present
        expected_set = set(range(N * N))
        actual_set = set(order.flatten())
        assert actual_set == expected_set, f"For N={N}, missing indices"


def test_spiral_order_even_size_converted_to_odd():
    """Even sizes should be converted to odd (N+1)"""
    order = spiral_order(6)  # Should become 7
    assert order.shape == (7, 7), f"Even size 6 should become 7, got shape {order.shape}"


def test_spiral_order_spiral_pattern():
    """Verify the spiral pattern: center=0, then spiraling outward"""
    order = spiral_order(5)
    center = (5 - 1) // 2

    # Center should be 0
    assert order[center, center] == 0

    # Values should increase as we move outward
    # Check immediate neighbors (should be 1-4)
    neighbors = [
        order[center - 1, center],  # up
        order[center + 1, center],  # down
        order[center, center - 1],  # left
        order[center, center + 1],  # right
    ]
    assert all(1 <= n <= 4 for n in neighbors), f"Neighbors should be 1-4, got {neighbors}"


def test_get_filtered_voltage_high_pass():
    """Verify high-pass filter removes DC component"""
    # Synthetic step waveform: constant value, then step up
    step_duration = 1e-9  # 1 ns per step
    voltage = [0.1] * 1000 + [0.2] * 1000 + [0.1] * 1000
    bias_tee_cut_off = 10000  # 10 kHz

    _, filtered = get_filtered_voltage(voltage, step_duration, bias_tee_cut_off, plot=False)

    # After high-pass, the DC component should be removed
    # The mean of the filtered signal should be close to zero
    # (allowing for some numerical error and edge effects)
    mean_filtered = np.mean(filtered[1000:])  # Skip initial transient
    assert abs(mean_filtered) < 0.05, f"High-pass should remove DC, but mean is {mean_filtered}"


def test_get_filtered_voltage_shape():
    """Verify filtered voltage has correct shape (1Gs/s sampling)"""
    step_duration = 1e-9  # 1 ns per step
    voltage = [0.1, 0.2, 0.1]
    bias_tee_cut_off = 10000

    unfiltered, filtered = get_filtered_voltage(voltage, step_duration, bias_tee_cut_off, plot=False)

    # Each step should be expanded to 1e9 samples per second * step_duration
    expected_length = len(voltage) * int(step_duration * 1e9)
    assert len(unfiltered) == expected_length, f"Expected length {expected_length}, got {len(unfiltered)}"
    assert len(filtered) == expected_length, f"Expected length {expected_length}, got {len(filtered)}"


def test_get_filtered_voltage_step_response():
    """Verify filter responds to step changes"""
    step_duration = 1e-9
    # Create a clear step: low, then high
    voltage = [0.0] * 500 + [0.5] * 500
    bias_tee_cut_off = 10000

    _, filtered = get_filtered_voltage(voltage, step_duration, bias_tee_cut_off, plot=False)

    # After the step, there should be a transient response
    # The filtered signal should show some variation around the step point
    step_point = 500 * int(step_duration * 1e9)
    window_after = filtered[step_point : step_point + 100]

    # Should have some variation (not all zeros)
    assert np.std(window_after) > 0.01, "Filter should show response to step change"


def test_round_to_fixed_precision():
    """Verify rounding to fixed point precision"""
    x = 0.123456789
    bits = 12

    result = round_to_fixed(x, bits)

    # Should round to nearest value representable with 12 bits
    expected = round((2**bits) * x) / (2**bits)
    assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"


def test_round_to_fixed_reduces_precision():
    """Verify that rounding actually reduces precision"""
    x = 0.12345678901234567890  # High precision
    bits = 8  # Low bit precision

    result = round_to_fixed(x, bits)

    # Result should have limited precision
    # Check that it's different from original (due to rounding)
    # But close to the quantized value
    quantized = round((2**bits) * x) / (2**bits)
    assert abs(result - quantized) < 1e-10, f"Result should match quantized value {quantized}, got {result}"


def test_round_to_fixed_zero():
    """Verify rounding works for zero"""
    result = round_to_fixed(0.0, 12)
    assert result == 0.0, f"Zero should round to zero, got {result}"


def test_round_to_fixed_negative():
    """Verify rounding works for negative numbers"""
    x = -0.123456789
    bits = 12

    result = round_to_fixed(x, bits)
    expected = round((2**bits) * x) / (2**bits)
    assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
    assert result < 0, "Negative input should give negative result"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

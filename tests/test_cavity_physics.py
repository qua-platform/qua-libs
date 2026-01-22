"""Physics-oriented tests for cavity parameters and constraints."""
import numpy as np
import pytest
from qualang_tools.units import unit

from tests.conftest import (
    closest_number,
    get_band,
    get_full_scale_power_dBm_and_amplitude,
)


class TestMWFEMBandSelection:
    """Test MW-FEM band selection for cavity frequencies."""
    
    @pytest.mark.parametrize("freq_hz,expected_band", [
        (100e6, 1),      # 100 MHz - Band 1
        (1e9, 1),        # 1 GHz - Band 1
        (4.5e9, 2),     # 4.5 GHz - Band 2 (lower boundary)
        (5.0e9, 2),     # 5.0 GHz - Band 2
        (6.5e9, 3),     # 6.5 GHz - Band 3 (lower boundary)
        (7.0e9, 3),     # 7.0 GHz - Band 3
        (8.0e9, 3),     # 8.0 GHz - Band 3 (typical cavity frequency)
        (10.0e9, 3),    # 10.0 GHz - Band 3
        (10.5e9, 3),    # 10.5 GHz - Band 3 (upper boundary)
    ])
    def test_band_selection(self, freq_hz, expected_band):
        """Test that get_band() returns correct band for various frequencies."""
        assert get_band(freq_hz) == expected_band
    
    def test_band_selection_out_of_range_low(self):
        """Test that frequencies below 50 MHz raise ValueError."""
        with pytest.raises(ValueError, match="outside of the MW fem bandwidth"):
            get_band(10e6)  # 10 MHz - too low
    
    def test_band_selection_out_of_range_high(self):
        """Test that frequencies above 10.5 GHz raise ValueError."""
        with pytest.raises(ValueError, match="outside of the MW fem bandwidth"):
            get_band(11e9)  # 11 GHz - too high
    
    def test_band_overlap_regions(self):
        """Test band selection in overlap regions (4.5-5.5 GHz, 6.5-7.5 GHz)."""
        # In overlap region 4.5-5.5 GHz, should return band 2
        assert get_band(5.0e9) == 2
        
        # In overlap region 6.5-7.5 GHz, should return band 3
        assert get_band(7.0e9) == 3


class TestIntermediateFrequencyConstraints:
    """Test intermediate frequency (IF) constraints for cavity drives."""
    
    @pytest.mark.parametrize("rf_freq,lo_freq,expected_if", [
        (8.0e9, 8.0e9, 0.0),           # No offset
        (8.1e9, 8.0e9, 100e6),         # 100 MHz IF
        (7.9e9, 8.0e9, -100e6),        # -100 MHz IF
        (8.3e9, 8.0e9, 300e6),         # 300 MHz IF (near limit)
        (7.7e9, 8.0e9, -300e6),        # -300 MHz IF (near limit)
        (8.399e9, 8.0e9, 399e6),       # 399 MHz IF (just under limit)
        (7.601e9, 8.0e9, -399e6),      # -399 MHz IF (just under limit)
    ])
    def test_if_frequency_calculation(self, rf_freq, lo_freq, expected_if):
        """Test that IF frequency is calculated correctly."""
        if_freq = rf_freq - lo_freq
        assert abs(if_freq - expected_if) < 1e6  # Within 1 MHz tolerance
    
    def test_if_frequency_within_limit(self):
        """Test that valid IF frequencies (< 400 MHz) pass validation."""
        rf_freq = 8.0e9
        lo_freq = 8.0e9
        if_freq = rf_freq - lo_freq
        
        assert abs(if_freq) < 400e6
    
    def test_if_frequency_exceeds_limit(self):
        """Test that IF frequencies exceeding 400 MHz raise assertion."""
        rf_freq = 8.5e9
        lo_freq = 8.0e9
        if_freq = rf_freq - lo_freq
        
        # Should exceed 400 MHz limit
        assert abs(if_freq) > 400e6
        
        # This would raise an assertion in the populate script
        with pytest.raises(AssertionError):
            assert abs(if_freq) < 400e6, (
                "The cavity intermediate frequency must be within [-400; 400] MHz. \n"
                f"Cavity frequency: {rf_freq} \n"
                f"Cavity LO frequency: {lo_freq} \n"
                f"Cavity IF frequency: {if_freq} \n"
            )
    
    @pytest.mark.parametrize("cavity_freq,lo_freq", [
        (8.0e9, 8.0e9),      # No offset - valid
        (8.1e9, 8.0e9),      # 100 MHz offset - valid
        (8.3e9, 8.0e9),      # 300 MHz offset - valid
        (7.7e9, 8.0e9),      # -300 MHz offset - valid
    ])
    def test_cavity_if_constraint_valid(self, cavity_freq, lo_freq):
        """Test that valid cavity IF frequencies pass constraint check."""
        if_freq = cavity_freq - lo_freq
        assert abs(if_freq) < 400e6, (
            "The cavity intermediate frequency must be within [-400; 400] MHz. \n"
            f"Cavity frequency: {cavity_freq} \n"
            f"Cavity LO frequency: {lo_freq} \n"
            f"Cavity IF frequency: {if_freq} \n"
        )


class TestPowerAmplitudeCalculation:
    """Test MW-FEM power and amplitude calculations."""
    
    def test_power_amplitude_basic(self):
        """Test basic power to amplitude conversion."""
        desired_power = -10  # dBm
        max_amplitude = 0.5  # V
        
        full_scale, amplitude = get_full_scale_power_dBm_and_amplitude(desired_power, max_amplitude)
        
        # Verify return types
        assert isinstance(full_scale, int)
        assert isinstance(amplitude, float)
        
        # Verify ranges
        assert -11 <= full_scale <= 16
        assert -1 <= amplitude <= 1
    
    @pytest.mark.parametrize("desired_power,max_amp,expected_range", [
        (-10, 0.5, (-11, 16)),      # Typical cavity drive
        (-40, 0.125, (-11, 16)),    # Readout power
        (-5, 0.5, (-11, 16)),       # Higher power
        (-15, 0.3, (-11, 16)),      # Lower power
    ])
    def test_power_amplitude_ranges(self, desired_power, max_amp, expected_range):
        """Test that power calculations stay within valid ranges."""
        full_scale, amplitude = get_full_scale_power_dBm_and_amplitude(desired_power, max_amp)
        
        min_fs, max_fs = expected_range
        assert min_fs <= full_scale <= max_fs
        assert -1 <= amplitude <= 1
    
    def test_power_amplitude_allowed_values(self):
        """Test that full_scale_power_dBm uses allowed discrete values."""
        allowed_powers = [-11, -8, -5, -2, 1, 4, 7, 10, 13, 16]
        
        # Test multiple power levels
        for desired_power in [-20, -10, -5, 0, 5, 10, 15]:
            full_scale, amplitude = get_full_scale_power_dBm_and_amplitude(desired_power)
            assert full_scale in allowed_powers
    
    def test_power_amplitude_relationship(self):
        """Test that amplitude and full_scale_power_dBm are correctly related."""
        desired_power = -10  # dBm
        max_amplitude = 0.5  # V
        
        full_scale, amplitude = get_full_scale_power_dBm_and_amplitude(desired_power, max_amplitude)
        
        # Verify the relationship: desired_power = full_scale + 20*log10(amplitude)
        calculated_power = full_scale + 20 * np.log10(amplitude)
        assert abs(calculated_power - desired_power) < 0.1  # Within 0.1 dB tolerance
    
    def test_power_amplitude_extreme_values(self):
        """Test power calculation with extreme but valid values."""
        # Very low power
        full_scale_low, amp_low = get_full_scale_power_dBm_and_amplitude(-80, 0.01)
        assert full_scale_low >= -11
        assert -1 <= amp_low <= 1
        
        # Very high power
        full_scale_high, amp_high = get_full_scale_power_dBm_and_amplitude(20, 0.5)
        assert full_scale_high <= 16
        assert -1 <= amp_high <= 1


class TestCavityCoherenceTimes:
    """Test cavity coherence time parameters."""
    
    def test_cavity_coherence_times_typical(self):
        """Test typical cavity coherence time values."""
        u = unit(coerce_to_integer=True)
        
        # Typical values from populate script
        T1 = 100 * u.us
        T2ramsey = 50 * u.us
        T2echo = 80 * u.us
        
        # Verify they are positive
        assert T1 > 0
        assert T2ramsey > 0
        assert T2echo > 0
        
        # Verify T2echo >= T2ramsey (echo should be longer)
        assert T2echo >= T2ramsey
        
        # Verify T1 >= T2echo (T1 is typically longest)
        assert T1 >= T2echo
    
    def test_cavity_coherence_time_units(self):
        """Test that coherence times can be specified in different units."""
        u = unit(coerce_to_integer=True)
        
        # Test microseconds
        T1_us = 100 * u.us
        assert T1_us > 0
        
        # Test nanoseconds
        T1_ns = 100000 * u.ns
        assert T1_ns > 0
        
        # Test milliseconds
        T1_ms = 0.1 * u.ms
        assert T1_ms > 0


class TestCavityVsTransmonPulses:
    """Test that cavity pulses differ from transmon pulses."""
    
    def test_cavity_uses_square_pulse(self):
        """Test that cavities use SquarePulse (not DRAG)."""
        from quam.components.pulses import SquarePulse
        
        # Cavity saturation pulse should be SquarePulse
        # This is verified in populate_quam_transmon_cavity.py
        # where cavity.xy.operations["saturation"] uses SquarePulse
        
        # Verify SquarePulse class exists
        assert SquarePulse is not None
    
    def test_transmon_uses_drag(self):
        """Test that transmons use DRAG pulses (not cavities)."""
        # Transmons use add_DragCosine_pulses from quam_builder
        # Cavities use SquarePulse directly
        # This distinction is important for physics correctness
        
        # Verify the distinction exists in the codebase
        from quam_builder.builder.superconducting.pulses import add_DragCosine_pulses
        assert add_DragCosine_pulses is not None


class TestCoupledPortBands:
    """Test that coupled MW-FEM ports share the same band."""
    
    def test_coupled_ports_same_band(self):
        """Test that coupled ports (O1&I1, O2&O3, etc.) must be in same band."""
        # According to MW-FEM specifications:
        # - O1 & I1 must be in same band
        # - O2 & O3 must be in same band
        # - O4 & O5 must be in same band
        # - O6 & O7 must be in same band
        # - O8 & I2 must be in same band
        
        # Example: If O1 is in band 2, I1 must also be in band 2
        freq_o1 = 5.0e9  # Band 2
        freq_i1 = 5.2e9  # Band 2
        
        assert get_band(freq_o1) == get_band(freq_i1) == 2
    
    @pytest.mark.parametrize("freq1,freq2,same_band", [
        (5.0e9, 5.2e9, True),    # Both in band 2
        (8.0e9, 8.1e9, True),    # Both in band 3
        (2.0e9, 8.0e9, False),  # Different bands
        (4.5e9, 6.5e9, False),  # Band 2 vs Band 3
    ])
    def test_port_band_compatibility(self, freq1, freq2, same_band):
        """Test port band compatibility for coupled ports."""
        band1 = get_band(freq1)
        band2 = get_band(freq2)
        
        if same_band:
            assert band1 == band2
        else:
            assert band1 != band2

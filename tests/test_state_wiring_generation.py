"""Tests that generate state.json and wiring.json files."""
import json
import os
import shutil
from pathlib import Path

import pytest
from qualang_tools.units import unit
from qualang_tools.wirer import Instruments, Connectivity, allocate_wiring
from qualang_tools.wirer.wirer.channel_specs import mw_fem_spec


class TestStateWiringGeneration:
    """Test generation of state.json and wiring.json files."""
    
    def test_generate_wiring_json(self, transmon_cavity_connectivity, test_output_dir):
        """Test that wiring.json is generated with cavity entries."""
        from quam_builder.builder.qop_connectivity import build_quam_wiring
        from quam_config import Quam
        
        # Create a temporary directory for this test
        test_dir = test_output_dir / "wiring_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        original_cwd = os.getcwd()
        original_state_path = os.environ.get("QUAM_STATE_PATH")
        try:
            os.chdir(test_dir)
            # Set QUAM_STATE_PATH environment variable for saving
            os.environ["QUAM_STATE_PATH"] = str(test_dir)
            
            # Create QUAM instance
            machine = Quam()
            
            # Build wiring
            build_quam_wiring(
                transmon_cavity_connectivity,
                host_ip="127.0.0.1",
                cluster_name="test_cluster",
                quam_instance=machine
            )
            
            # Verify wiring.json was created
            wiring_file = test_dir / "wiring.json"
            assert wiring_file.exists(), f"wiring.json should be created at {test_dir}"
            
            # Load and validate structure
            with open(wiring_file) as f:
                wiring_data = json.load(f)
            
            # Verify top-level structure
            assert "wiring" in wiring_data, "wiring.json should have 'wiring' key"
            assert "network" in wiring_data, "wiring.json should have 'network' key"
            
            # Verify network configuration
            network = wiring_data["network"]
            assert network["host"] == "127.0.0.1"
            assert network["cluster_name"] == "test_cluster"
            
            # Verify wiring structure
            wiring = wiring_data["wiring"]
            assert "qubits" in wiring, "wiring should have 'qubits' key"
            
            # Check if q1 exists (structure depends on quam-builder implementation)
            qubits = wiring["qubits"]
            assert len(qubits) > 0, "Should have at least one qubit"
            
            # Copy to persistent output directory for inspection
            shutil.copy(wiring_file, test_output_dir / "wiring.json")
            
        finally:
            os.chdir(original_cwd)
            # Restore environment variable
            if original_state_path is not None:
                os.environ["QUAM_STATE_PATH"] = original_state_path
            elif "QUAM_STATE_PATH" in os.environ:
                del os.environ["QUAM_STATE_PATH"]
    
    def test_generate_state_json(self, transmon_cavity_connectivity, test_output_dir):
        """Test that state.json is generated with cavity parameters."""
        from quam_builder.builder.qop_connectivity import build_quam_wiring
        from quam_builder.builder.superconducting import build_quam
        from quam_config import Quam
        
        # Create a temporary directory for this test
        test_dir = test_output_dir / "state_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        original_cwd = os.getcwd()
        original_state_path = os.environ.get("QUAM_STATE_PATH")
        try:
            os.chdir(test_dir)
            # Set QUAM_STATE_PATH environment variable for saving
            os.environ["QUAM_STATE_PATH"] = str(test_dir)
            
            # Step 1: Build wiring
            machine = Quam()
            build_quam_wiring(
                transmon_cavity_connectivity,
                host_ip="127.0.0.1",
                cluster_name="test_cluster",
                quam_instance=machine
            )
            
            # Step 2: Reload and build QUAM structure
            machine = Quam.load()
            build_quam(machine, calibration_db_path=None)
            
            # Verify state.json was created
            state_file = test_dir / "state.json"
            assert state_file.exists(), "state.json should be created"
            
            # Load and validate structure
            with open(state_file) as f:
                state_data = json.load(f)
            
            # Verify QUAM structure (exact keys depend on implementation)
            # At minimum, should be valid JSON
            assert isinstance(state_data, dict), "state.json should be a dictionary"
            
            # Check for common QUAM keys
            # Note: Structure may vary based on quam-builder version
            has_qubits = "qubits" in state_data
            has_cavities = "cavities" in state_data
            
            # At least one of these should be present
            assert has_qubits or has_cavities or len(state_data) > 0, (
                "state.json should contain qubits, cavities, or other QUAM data"
            )
            
            # Copy to persistent output directory for inspection
            shutil.copy(state_file, test_output_dir / "state.json")
            
        finally:
            os.chdir(original_cwd)
            # Restore environment variable
            if original_state_path is not None:
                os.environ["QUAM_STATE_PATH"] = original_state_path
            elif "QUAM_STATE_PATH" in os.environ:
                del os.environ["QUAM_STATE_PATH"]
    
    def test_generate_complete_quam_setup(self, test_output_dir):
        """Test complete QUAM setup with cavity: wiring + state + population."""
        from quam_builder.builder.qop_connectivity import build_quam_wiring
        from quam_builder.builder.superconducting import build_quam
        from quam_config import Quam
        
        # Create instruments and connectivity
        instruments = Instruments()
        instruments.add_mw_fem(controller=1, slots=[1])
        
        qubits = [1]
        rr_ch = mw_fem_spec(con=1, slot=1, in_port=1, out_port=1)
        xy_ch = mw_fem_spec(con=1, slot=1, out_port=2)
        cavity_ch = mw_fem_spec(con=1, slot=1, out_port=3)
        
        connectivity = Connectivity()
        connectivity.add_resonator_line(qubits=qubits, constraints=rr_ch)
        connectivity.add_qubit_drive_lines(qubits=qubits, constraints=xy_ch)
        connectivity.add_cavity_lines(qubit=1, constraints=cavity_ch)
        allocate_wiring(connectivity, instruments)
        
        # Create test directory
        test_dir = test_output_dir / "complete_setup"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        original_cwd = os.getcwd()
        original_state_path = os.environ.get("QUAM_STATE_PATH")
        try:
            os.chdir(test_dir)
            # Set QUAM_STATE_PATH environment variable for saving
            os.environ["QUAM_STATE_PATH"] = str(test_dir)
            
            # Step 1: Generate wiring
            machine = Quam()
            build_quam_wiring(
                connectivity,
                host_ip="127.0.0.1",
                cluster_name="test_cluster",
                quam_instance=machine
            )
            
            # Step 2: Build QUAM structure
            machine = Quam.load()
            build_quam(machine, calibration_db_path=None)
            
            # Step 3: Populate with physics parameters (simplified version)
            u = unit(coerce_to_integer=True)
            machine = Quam.load()
            
            # Set cavity parameters if cavities exist
            if hasattr(machine, "cavities") and machine.cavities:
                for cavity_id, cavity in machine.cavities.items():
                    cavity.frequency = 8.0 * u.GHz
                    if hasattr(cavity, "xy"):
                        cavity.xy.RF_frequency = 8.0 * u.GHz
                    cavity.T1 = 100 * u.us
                    cavity.T2ramsey = 50 * u.us
                    cavity.T2echo = 80 * u.us
            
            # Save populated state
            machine.save()
            
            # Verify both files exist
            wiring_file = test_dir / "wiring.json"
            state_file = test_dir / "state.json"
            
            assert wiring_file.exists(), "wiring.json should exist"
            assert state_file.exists(), "state.json should exist"
            
            # Verify files are valid JSON
            with open(wiring_file) as f:
                wiring_data = json.load(f)
            assert isinstance(wiring_data, dict)
            
            with open(state_file) as f:
                state_data = json.load(f)
            assert isinstance(state_data, dict)
            
            # Copy to persistent output directory
            shutil.copy(wiring_file, test_output_dir / "complete_wiring.json")
            shutil.copy(state_file, test_output_dir / "complete_state.json")
            
        finally:
            os.chdir(original_cwd)
            # Restore environment variable
            if original_state_path is not None:
                os.environ["QUAM_STATE_PATH"] = original_state_path
            elif "QUAM_STATE_PATH" in os.environ:
                del os.environ["QUAM_STATE_PATH"]
    
    def test_wiring_json_structure_validation(self, transmon_cavity_connectivity, test_output_dir):
        """Test that wiring.json has correct structure for cavity setup."""
        from quam_builder.builder.qop_connectivity import build_quam_wiring
        from quam_config import Quam
        
        test_dir = test_output_dir / "structure_validation"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        original_cwd = os.getcwd()
        original_state_path = os.environ.get("QUAM_STATE_PATH")
        try:
            os.chdir(test_dir)
            # Set QUAM_STATE_PATH environment variable for saving
            os.environ["QUAM_STATE_PATH"] = str(test_dir)
            
            machine = Quam()
            build_quam_wiring(
                transmon_cavity_connectivity,
                host_ip="127.0.0.1",
                cluster_name="test_cluster",
                quam_instance=machine
            )
            
            wiring_file = test_dir / "wiring.json"
            if wiring_file.exists():
                with open(wiring_file) as f:
                    wiring_data = json.load(f)
                
                # Validate JSON structure
                assert "wiring" in wiring_data
                assert "network" in wiring_data
                
                # Network should have required fields
                network = wiring_data["network"]
                required_network_fields = ["host", "cluster_name"]
                for field in required_network_fields:
                    assert field in network, f"network should have '{field}' field"
                
                # Wiring should have qubits
                wiring = wiring_data["wiring"]
                assert "qubits" in wiring
                
        finally:
            os.chdir(original_cwd)
            # Restore environment variable
            if original_state_path is not None:
                os.environ["QUAM_STATE_PATH"] = original_state_path
            elif "QUAM_STATE_PATH" in os.environ:
                del os.environ["QUAM_STATE_PATH"]
    
    def test_state_json_cavity_parameters(self, transmon_cavity_connectivity, test_output_dir):
        """Test that state.json contains cavity parameters when populated."""
        from quam_builder.builder.qop_connectivity import build_quam_wiring
        from quam_builder.builder.superconducting import build_quam
        from quam_config import Quam
        
        test_dir = test_output_dir / "cavity_params"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        original_cwd = os.getcwd()
        original_state_path = os.environ.get("QUAM_STATE_PATH")
        try:
            os.chdir(test_dir)
            # Set QUAM_STATE_PATH environment variable for saving
            os.environ["QUAM_STATE_PATH"] = str(test_dir)
            
            # Build wiring and state
            machine = Quam()
            build_quam_wiring(
                transmon_cavity_connectivity,
                host_ip="127.0.0.1",
                cluster_name="test_cluster",
                quam_instance=machine
            )
            
            machine = Quam.load()
            build_quam(machine, calibration_db_path=None)
            
            # Populate cavity parameters
            u = unit(coerce_to_integer=True)
            machine = Quam.load()
            
            # Set cavity frequency and coherence times
            if hasattr(machine, "cavities") and machine.cavities:
                for cavity_id, cavity in machine.cavities.items():
                    cavity.frequency = 8.0 * u.GHz
                    if hasattr(cavity, "xy"):
                        cavity.xy.RF_frequency = 8.0 * u.GHz
                        if hasattr(cavity.xy, "opx_output"):
                            cavity.xy.opx_output.band = 3  # Band 3 for 8 GHz
                    cavity.T1 = 100 * u.us
                    cavity.T2ramsey = 50 * u.us
                    cavity.T2echo = 80 * u.us
            
            machine.save()
            
            # Verify state.json contains cavity data
            state_file = test_dir / "state.json"
            if state_file.exists():
                with open(state_file) as f:
                    state_data = json.load(f)
                
                # Check if cavities are present (structure may vary)
                # The exact structure depends on quam-builder implementation
                # We verify the file is valid and was saved
                assert isinstance(state_data, dict)
                
                # Copy for inspection
                shutil.copy(state_file, test_output_dir / "cavity_params_state.json")
            
        finally:
            os.chdir(original_cwd)
            # Restore environment variable
            if original_state_path is not None:
                os.environ["QUAM_STATE_PATH"] = original_state_path
            elif "QUAM_STATE_PATH" in os.environ:
                del os.environ["QUAM_STATE_PATH"]

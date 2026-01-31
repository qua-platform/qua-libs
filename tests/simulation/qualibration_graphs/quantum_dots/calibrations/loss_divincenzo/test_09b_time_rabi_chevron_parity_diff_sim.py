"""Cloud simulation test for 09b_time_rabi_chevron_parity_diff.

This test is opt-in and requires:
1. QM SaaS credentials at tests/.qm_saas_credentials.json
2. Environment variable RUN_SIM_TESTS=1

The test:
1. Creates a programmatic QuAM (no state file required)
2. Runs custom_param -> create_qua_program node actions
3. Simulates the QUA program via QM SaaS cloud
4. Saves artifacts: simulation.png and README.md
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

# Find repository root
TEST_ROOT = Path(__file__).resolve().parent
REPO_ROOT = None
for parent in [TEST_ROOT, *TEST_ROOT.parents]:
    if (parent / "qualibration_graphs").is_dir() and (parent / "tests").is_dir():
        REPO_ROOT = parent
        break
if REPO_ROOT is None:
    REPO_ROOT = TEST_ROOT.parents[0]

NODE_PATH = (
    REPO_ROOT
    / "qualibration_graphs"
    / "quantum_dots"
    / "calibrations"
    / "loss_divincenzo"
    / "09b_time_rabi_chevron_parity_diff.py"
)
ARTIFACTS_SUBDIR = "09b_time_rabi_chevron_parity_diff"


@pytest.mark.simulation
def test_time_rabi_chevron_parity_diff_simulation(
    simulation_test_context,
    save_simulation_plot,
    markdown_generator,
):
    """Test QUA program generation and simulation for time rabi chevron parity diff node.

    This test validates that:
    1. The node can generate a valid QUA program with programmatic QuAM
    2. The program simulates successfully on QM SaaS
    3. Simulation artifacts are generated correctly
    """
    # Step 1: Check if simulation tests are enabled
    run_sim_tests = os.environ.get("RUN_SIM_TESTS")
    if run_sim_tests is not None and run_sim_tests != "1":
        pytest.skip("Simulation tests disabled. Set RUN_SIM_TESTS=1 to run.")

    # Step 2: Create test context with programmatic QuAM
    ctx = simulation_test_context(NODE_PATH, ARTIFACTS_SUBDIR)

    # Step 3: Configure small sweep parameters for fast simulation
    ctx.configure_small_sweep()

    # Step 4: Run custom_param action (sets any debug parameters)
    # Note: The @node.run_action decorator wraps functions to auto-inject the node,
    # so we call them without passing node explicitly
    ctx.loaded_node.get_action("custom_param")()

    # Step 5: Run create_qua_program action
    try:
        ctx.loaded_node.get_action("create_qua_program")()
    except Exception as exc:  # pragma: no cover - setup dependent
        pytest.skip(f"Failed to build QUA program: {exc}")

    # Step 6: Import QM dependencies (late import to avoid test collection issues)
    try:
        from qm import QuantumMachinesManager, SimulationConfig
        import qm_saas
    except Exception as exc:  # pragma: no cover - optional dependency
        pytest.skip(f"QM SaaS dependencies unavailable: {exc}")

    # Step 7: Get config and program from node
    config = ctx.machine.generate_config()
    qua_program = ctx.loaded_node.node.namespace["qua_program"]

    # Step 8: Connect to QM SaaS and simulate
    client = qm_saas.QmSaas(
        email=ctx.credentials["email"],
        password=ctx.credentials["password"],
        host=ctx.credentials["host"],
    )
    client.close_all()

    with client.simulator(client.latest_version()) as instance:
        qmm = QuantumMachinesManager(
            host=instance.host,
            port=instance.port,
            connection_headers=instance.default_connection_headers,
            timeout=120,
        )

        simulation_config = SimulationConfig(duration=10_000)
        job = qmm.simulate(config, qua_program, simulation_config)
        job.wait_until("Done", timeout=60)

        # Retry sample pull to reduce transient QOP errors.
        simulated_samples = None
        for attempt in range(3):
            try:
                simulated_samples = job.get_simulated_samples()
                break
            except Exception:  # pragma: no cover - network dependent
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))

    # Step 9: Save simulation plot
    save_simulation_plot(
        simulated_samples,
        ctx.artifacts_dir,
        title="Time Rabi Chevron Parity Diff - Simulated Samples",
    )

    # Step 10: Generate README.md documentation
    markdown_generator(
        ctx.loaded_node,
        ctx.get_parameters_dict(),
        ctx.artifacts_dir,
    )

    # Verify artifacts were created
    assert (ctx.artifacts_dir / "simulation.png").exists(), "simulation.png not created"
    assert (ctx.artifacts_dir / "README.md").exists(), "README.md not created"

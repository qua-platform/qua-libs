"""Serialization round-trip test: 18a → QuAM state → 18b.

Node ``18a_swap_oscillations`` fits a polynomial ``T_2π(V)`` model and stores
it on the CZ macro (``exchange_decay_model``).  Node
``18b_geometric_cz_amplitude_phase_calibration`` consumes that model (when
``use_t2pi_model=True``) to compute a per-amplitude exchange duration while
*generating its QUA program* (``_build_duration_array`` →
``create_qua_program``).

This test runs the two nodes in turn and verifies that the model survives a
real serialisation round-trip:

1.  Simulate swap oscillations with the virtual_qpu Loss-DiVincenzo device
    (same backend as the other analysis tests) and run 18a's
    ``analyse_data`` + ``update_state`` so the fitted model lands on the CZ
    macro in the in-memory QuAM machine.
2.  ``machine.save(path)`` — write the QuAM state to disk.
3.  ``Quam.load(path)`` — reload it from disk.
4.  Run 18b's ``create_qua_program`` (with ``use_t2pi_model=True``) on the
    *reloaded* machine and confirm the per-amplitude duration array it builds
    matches the model coefficients.

If ``exchange_decay_model`` were not a declared, serialisable field on the CZ
macro, step 2/3 would silently drop it and step 4 would raise
``"No exchange_decay_model ..."`` — so this test guards the serialisation
contract between the two nodes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

import dynamiqs as dq
import jax.numpy as jnp

from virtual_qpu.dynamics import simulate as vqpu_simulate
from virtual_qpu.operators import expval as vqpu_expval
from virtual_qpu.pulse import SquarePulse
from virtual_qpu.schedule import Schedule
from virtual_qpu._sweep import sweep as vqpu_sweep

from shared_fixtures import (
    apply_param_overrides,
    call_node_action,
    ensure_quam_config_stub,
    patch_action_manager_register_only,
    reimport_node_to_register_actions,
)

from .conftest import (
    CALIBRATION_LIBRARY_ROOT,
    QUBIT_PAIR_NAMES,
    DEFAULT_SOLVER,
)

NODE_18A = "18a_swap_oscillations"
NODE_18B = "18b_geometric_cz_amplitude_phase_calibration"

# ── Compact swap-oscillation sweep grid (kept small for ODE speed) ──────────
# Voltage window chosen so T_2π(V) sweeps from a few hundred ns down to a few
# tens of ns over the range (J(V) = J0 exp((V - Vref)/lever_arm)).
N_VOLTAGES = 16
V_MIN = 0.20
V_MAX = 0.30

DURATION_STEP_NS = 4.0
DUR_MIN = 0.0
DUR_MAX = 400.0
_DUR_STEPS = int(np.floor((DUR_MAX - DUR_MIN) / DURATION_STEP_NS))
DUR_LAST_SWEPT = DUR_MIN + _DUR_STEPS * DURATION_STEP_NS
N_DURATIONS = _DUR_STEPS + 1

RETURN_TO_INIT_NS = 16.0
NOISE_STD = 0.01

_SOLVER_KW = {"method": dq.method.Tsit5(max_steps=250_000)}

# Perfect X180 on the target qubit (mode 0) to prepare |10⟩.
_I2 = jnp.eye(2, dtype=jnp.complex64)
_Rx_pi = jnp.array([[0.0, -1j], [-1j, 0.0]], dtype=jnp.complex64)
_U_PI_Q0 = jnp.kron(_Rx_pi, _I2)
_PROJ_1 = jnp.array([[0, 0], [0, 1]], dtype=jnp.complex64)


def _make_exchange_schedule(exchange_voltage, exchange_duration):
    """Exchange-only schedule: exchange pulse + return-to-init idle."""
    sched = Schedule()
    ref_ex = sched.play(
        SquarePulse(
            duration=exchange_duration, amplitude=exchange_voltage, frequency=0.0
        ),
        channel="exchange_0_1",
    )
    sched.play(
        SquarePulse(duration=RETURN_TO_INIT_NS, amplitude=0.0, frequency=0.0),
        channel="exchange_0_1",
        after=[ref_ex],
    )
    return sched.resolve()


def _simulate_swap_ds_raw(device) -> tuple[xr.Dataset, np.ndarray, np.ndarray]:
    """Build the 18a ``ds_raw`` from a virtual_qpu 2-D exchange sweep."""
    voltages_np = np.linspace(V_MIN, V_MAX, N_VOLTAGES, dtype=np.float64)
    voltages = jnp.asarray(voltages_np, dtype=jnp.float32)
    duration_grid = DUR_MIN + DURATION_STEP_NS * np.arange(
        N_DURATIONS, dtype=np.float64
    )
    durations = jnp.asarray(duration_grid, dtype=jnp.float32)

    psi_prepared = _U_PI_Q0 @ device.ground_state()
    obs_target = dq.asqarray(device.embed(_PROJ_1, mode=0))
    obs_control = dq.asqarray(device.embed(_PROJ_1, mode=1))
    jump_ops = device.collapse_operators() if DEFAULT_SOLVER == "me" else None

    def _sweep_2d(observable, seed):
        def _inner(**kwargs):
            resolved = _make_exchange_schedule(**kwargs)
            H_t = device.hamiltonian(resolved)
            dur = jnp.asarray(kwargs["exchange_duration"], dtype=jnp.float32)
            ts = jnp.stack(
                [jnp.float32(0.0), dur + jnp.float32(RETURN_TO_INIT_NS)]
            )
            sol = vqpu_simulate(
                H_t, psi_prepared, ts, solver=DEFAULT_SOLVER,
                jump_ops=jump_ops, options=_SOLVER_KW,
            )
            return vqpu_expval(sol.states, observable)

        result = np.asarray(
            vqpu_sweep(
                _inner, exchange_voltage=voltages, exchange_duration=durations
            )
        )
        rng = np.random.default_rng(seed=seed)
        result = result + rng.normal(0, NOISE_STD, size=result.shape)
        return np.clip(result, 0.0, 1.0)

    data_control = np.asarray(_sweep_2d(obs_control, seed=42)[..., -1], dtype=np.float64)
    data_target = np.asarray(_sweep_2d(obs_target, seed=43)[..., -1], dtype=np.float64)

    qp_name = QUBIT_PAIR_NAMES[0]
    ds_raw = xr.Dataset(
        {
            f"state_control_{qp_name}": xr.DataArray(
                data_control,
                dims=["exchange_amplitude", "exchange_duration"],
                coords={
                    "exchange_amplitude": voltages_np,
                    "exchange_duration": duration_grid,
                },
            ),
            f"state_target_{qp_name}": xr.DataArray(
                data_target,
                dims=["exchange_amplitude", "exchange_duration"],
                coords={
                    "exchange_amplitude": voltages_np,
                    "exchange_duration": duration_grid,
                },
            ),
        }
    )
    return ds_raw, voltages_np, duration_grid


def _load_node_with_machine(node_name: str, machine):
    """Re-import a calibration node with its actions registered against *machine*."""
    ensure_quam_config_stub(machine)
    from quam_config import Quam

    with (
        patch.object(Quam, "load", return_value=machine),
        patch_action_manager_register_only(),
    ):
        node = reimport_node_to_register_actions(node_name, CALIBRATION_LIBRARY_ROOT)
    assert node is not None, f"Failed to import node {node_name}"
    node.machine = machine
    return node


@pytest.mark.analysis
def test_18a_18b_exchange_decay_model_serialization(
    ld_device,
    analysis_runner,
    tmp_path: Path,
):
    """Model fitted in 18a survives save/reload and drives 18b program generation."""
    qp_name = QUBIT_PAIR_NAMES[0]

    # ── Step 1: run 18a (simulate → analyse → update_state) ─────────────
    ds_raw, voltages_np, _ = _simulate_swap_ds_raw(ld_device)
    amplitude_step = float((V_MAX - V_MIN) / max(N_VOLTAGES - 1, 1))

    node_18a = analysis_runner(
        node_name=NODE_18A,
        ds_raw=ds_raw,
        param_overrides={
            "num_shots": 4,
            "min_exchange_amplitude": float(V_MIN),
            "max_exchange_amplitude": float(V_MAX),
            "amplitude_step": amplitude_step,
            "min_exchange_duration_in_ns": int(DUR_MIN),
            "max_exchange_duration_in_ns": int(np.ceil(DUR_LAST_SWEPT)) + 1,
            "duration_step_in_ns": int(DURATION_STEP_NS),
            "snr_threshold": 3.0,
            "analysis_role": "best",
        },
        analyse_qubit_pairs=QUBIT_PAIR_NAMES,
    )

    fit = node_18a.results["fit_results"][qp_name]
    assert fit["success"], f"18a analysis failed: {fit}"
    assert fit["model_fit_success"], f"18a polynomial fit failed: {fit}"
    model = fit["exchange_decay_model"]
    assert model and model["type"] == "polynomial"

    # update_state must have written the model onto the in-memory CZ macro.
    machine = node_18a.machine
    cz_macro = machine.qubit_pairs[qp_name].macros["cz"]
    assert cz_macro.exchange_decay_model is not None, (
        "18a update_state did not store exchange_decay_model on the CZ macro"
    )
    assert cz_macro.exchange_decay_model["coeffs"] == model["coeffs"]

    # ── Step 2: serialise the QuAM state to disk ────────────────────────
    state_path = tmp_path / "quam_state"
    machine.save(state_path)
    assert state_path.exists(), "machine.save produced no state"

    # ── Step 3: reload the QuAM state from disk ─────────────────────────
    # NOTE: ``analysis_runner`` installs a ``quam_config`` stub whose
    # ``Quam.load()`` just returns the in-memory machine, which would bypass
    # serialisation entirely.  Load through the *real* root class so we
    # genuinely deserialise the on-disk JSON.
    from quam_builder.architecture.quantum_dots.qpu.loss_divincenzo_quam import (
        LossDiVincenzoQuam,
    )

    reloaded = LossDiVincenzoQuam.load(state_path)
    assert reloaded is not machine, (
        "expected a freshly deserialised machine, not the in-memory object"
    )
    reloaded_cz = reloaded.qubit_pairs[qp_name].macros["cz"]

    # The model is the whole point: it MUST survive serialisation.
    reloaded_model = getattr(reloaded_cz, "exchange_decay_model", None)
    assert reloaded_model is not None, (
        "exchange_decay_model was lost during save/reload — it is not a "
        "serialisable field on the CZ macro"
    )
    assert reloaded_model["coeffs"] == model["coeffs"]
    assert reloaded_model["degree"] == model["degree"]

    # The reloaded macro's evaluator agrees with the raw polynomial.
    v_probe = float(fit["best_amplitude"])
    assert np.isclose(
        reloaded_cz.t_2pi(v_probe), np.polyval(model["coeffs"], v_probe), rtol=1e-6
    )
    assert np.isclose(reloaded_cz.t_cz(v_probe), reloaded_cz.t_2pi(v_probe) / 2.0)

    # ── Step 4: consume the model in 18b's program generation ───────────
    node_18b = _load_node_with_machine(NODE_18B, reloaded)
    apply_param_overrides(
        node_18b,
        {
            "qubit_pairs": list(QUBIT_PAIR_NAMES),
            "simulate": False,
            "load_data_id": None,
            "use_t2pi_model": True,
            "num_shots": 2,
            "num_phases": 4,
            "min_exchange_amplitude": float(V_MIN),
            "max_exchange_amplitude": float(V_MAX),
            "amplitude_step": amplitude_step,
        },
    )
    call_node_action(node_18b, "create_qua_program")

    # create_qua_program must have evaluated the reloaded model to build the
    # per-amplitude duration array (this is where the model is "consumed").
    duration_array = node_18b.namespace.get("duration_array")
    assert duration_array is not None, (
        "18b create_qua_program did not build a per-amplitude duration_array — "
        "the serialised exchange_decay_model was not consumed"
    )

    amplitude_array = np.arange(V_MIN, V_MAX, amplitude_step)
    expected_t2pi = np.polyval(model["coeffs"], amplitude_array)
    expected = np.clip((np.round(expected_t2pi / 2.0 / 4) * 4).astype(int), 16, None)
    np.testing.assert_array_equal(np.asarray(duration_array), expected)

    # The QUA program itself should have been generated.
    assert node_18b.namespace.get("qua_program") is not None, (
        "18b did not generate a QUA program"
    )
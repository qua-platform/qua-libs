import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from scipy.stats import chi2 as chi2_dist

# The four sliced dual-demod projections (one per demod-weight/output-port pairing) and the
# two per-shot second moments derived from them in stream_processing -- shared by the node's
# stream_processing block, fetch_sliced_iq_traces, and process_raw_dataset so the key list
# only needs to change in one place.
SLICE_KEYS = ("II", "IQ", "QI", "QQ")
MOMENT_KEYS = ("I_sq", "Q_sq")


def fetch_sliced_iq_traces(job, qubits, conditions: list, states: list, num_segments: int) -> dict[str, np.ndarray]:
    """Fetch each (condition, state, qubit) sliced dual-demod II/IQ/QI/QQ result, plus the
    I/Q second moments (I_sq, Q_sq) used to derive the amplitude's shot-noise std, directly
    from job.result_handles and stack across qubits.

    num_segments is fixed and shared across qubits (a Parameters field), so every qubit's
    sliced-demod result already has the same shape -- no padding needed.
    """
    traces: dict[str, np.ndarray] = {}
    for condition in conditions:
        for state in states:
            for key in SLICE_KEYS + MOMENT_KEYS:
                stacked = []
                for i, qubit in enumerate(qubits):
                    name = f"{key}_{condition}_{state}{i + 1}"
                    stacked.append(np.asarray(job.result_handles.get(name).fetch_all()))
                traces[f"{key}_{condition}_{state}"] = np.stack(stacked, axis=0)
    return traces


def resolve_conditions(parameters) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The conditions this run acquires/analyses: the selected parameters.test_operation and the
    "no_operation" reference. The ground-vs-excited distinguishability test only checks
    test_operation.

    Must be called from parameters that are already final -- i.e. after custom_param, or
    (on the load_data_id replay path) after load_from_id has overwritten node.parameters with
    the historical run's values, so a replayed dataset is analysed under the operation it was
    actually acquired with.
    """
    return (parameters.test_operation, "no_operation"), (parameters.test_operation,)


def convert_sliced_demod_to_volts(ds: xr.Dataset, keys: list[str], squared: bool = False) -> xr.Dataset:
    """Convert sliced dual-demod counts to volts.

    Each measure_sliced segment integrates only segment_length_ns (pulse_length/num_segments),
    not the full pulse -- so that per-qubit coordinate is the correct divisor here, unlike
    qualibration_libs.convert_IQ_to_V, which divides by a hard-coded operations["readout"]'s
    FULL length and can't be reused for a sliced-demod result. factor is 2x for the
    single-demod correction (matching convert_IQ_to_V's single_demod=True) and 2**12 for the
    fixed-point scale; squared=True (for I_sq/Q_sq, which are products of two demod-scaled
    quadratures) squares the whole factor instead of applying it twice.
    """
    factor = 2 * 2**12 / ds["segment_length_ns"]
    if squared:
        factor = factor**2
    return ds.assign({key: ds[key] * factor for key in keys})


def process_raw_dataset(ds: xr.Dataset, conditions: list, states: list) -> xr.Dataset:
    """Convert the sliced dual-demod counts to volts, stack the per-(condition, state) streams
    into "condition" and "state" dimensions, combine the four projections into I/Q (I = II+IQ,
    Q = QI+QQ -- the same projection pairing quam's InOutIQChannel.measure() uses for full dual
    demod), and compute the residual-field amplitude |IQ| vs segment and its shot-noise std.

    Pure function of `ds` -- callers own where the result is stored (node.results["ds_fit"]),
    so re-running this on the same loaded ds_raw is idempotent.
    """
    raw_vars = [f"{key}_{c}_{s}" for key in SLICE_KEYS for c in conditions for s in states]
    ds = convert_sliced_demod_to_volts(ds, raw_vars)

    sq_vars = [f"{key}_{c}_{s}" for key in MOMENT_KEYS for c in conditions for s in states]
    ds = convert_sliced_demod_to_volts(ds, sq_vars, squared=True)

    # Each (condition, state)'s II/IQ/QI/QQ/I_sq/Q_sq lives in its own
    # <key>_<condition>_<state> variable (they were streamed separately in
    # create_qua_program); stack them into "condition"/"state" axes. Concat over state first
    # (per condition), then over condition -- final dims are (condition, state, qubit, segment).
    state_coord = xr.DataArray(list(states), dims="state", name="state")
    condition_coord = xr.DataArray(list(conditions), dims="condition", name="condition")
    stacked = {
        key: xr.concat(
            [xr.concat([ds[f"{key}_{c}_{s}"] for s in states], dim=state_coord) for c in conditions],
            dim=condition_coord,
        )
        for key in SLICE_KEYS + MOMENT_KEYS
    }
    ds = ds.assign(**stacked)

    # Time (ns) of the start of each probe segment -- the same definition the depletion time
    # uses (segment index * segment length). Identical for every qubit (shared segment length).
    segment_length_ns = float(ds["segment_length_ns"].values[0])
    ds = ds.assign_coords(time_ns=("segment", ds["segment"].values * segment_length_ns))
    ds.time_ns.attrs = {"long_name": "probe time", "units": "ns"}

    # Combine the four sliced dual-demod projections into I/Q -- the IF demodulation already
    # happened on-chip (via the integration weights), so no further digital demod is needed.
    I = ds["II"] + ds["IQ"]
    Q = ds["QI"] + ds["QQ"]
    ds = ds.assign(I=I, Q=Q, IQ_abs=np.abs(I + 1j * Q))
    ds.IQ_abs.attrs = {"long_name": "residual cavity field |IQ|", "units": "V"}

    # Shot-noise std of the amplitude, via first-order error propagation (delta method) of
    # amp = sqrt(I^2+Q^2):
    #   Var(amp) ~= (I^2*Var(I) + Q^2*Var(Q)) / amp^2
    # The I-Q cross/covariance term is dropped as negligible for the signals this node's tests
    # actually use -- see README.md for the measurement that established that.
    # Clipped to >=0 (guards against floating-point noise pushing an already-tiny variance
    # slightly negative) and persisted on ds -- reused as-is by compute_stat_depletion_time
    # and compute_ge_depletion_time.
    var_I = (ds["I_sq"] - I**2).clip(min=0)
    var_Q = (ds["Q_sq"] - Q**2).clip(min=0)
    var_amp = xr.where(ds["IQ_abs"] > 0, (I**2 * var_I + Q**2 * var_Q) / ds["IQ_abs"] ** 2, np.nan)
    ds = ds.assign(var_I=var_I, var_Q=var_Q, IQ_abs_std=np.sqrt(var_amp))
    ds.var_I.attrs = {"long_name": "Var(I) (shot noise)", "units": "V^2"}
    ds.var_Q.attrs = {"long_name": "Var(Q) (shot noise)", "units": "V^2"}
    ds.IQ_abs_std.attrs = {"long_name": "residual cavity field |IQ| std (shot noise)", "units": "V"}

    return ds


def _chi2_pvalue(dI: xr.DataArray, dQ: xr.DataArray, SE_I: xr.DataArray, SE_Q: xr.DataArray) -> xr.DataArray:
    """Two-sample z-test per quadrature (dI/SE_I, dQ/SE_Q), combined into
    chi2_stat = z_I^2 + z_Q^2 ~ chi2(df=2) under the null hypothesis that the two samples
    dI/dQ were computed from are equal at that segment. Assumes Cov(I,Q)=0 -- see
    process_raw_dataset's IQ_abs_std, where the cross term was measured and dropped as
    negligible."""
    z_I = dI / SE_I
    z_Q = dQ / SE_Q
    chi2_stat = z_I**2 + z_Q**2
    return xr.apply_ufunc(chi2_dist.sf, chi2_stat, kwargs={"df": 2})


def _first_sustained_segment(passed: np.ndarray, debounce: int) -> int | None:
    """First index k such that passed[k:k+debounce] are all True, or None if no such k exists
    within the measured window. debounce is clamped to >=1: with debounce=0, passed[k:k+0] is
    an empty slice and ndarray.all() on an empty array is vacuously True, which would resolve
    to segment 0 regardless of the actual data."""
    debounce = max(1, debounce)
    num_segments = len(passed)
    for k in range(num_segments - debounce + 1):
        if passed[k : k + debounce].all():
            return k
    return None


def _scan_depletion_time(
    p_value: xr.DataArray,
    segment_length_ns: xr.DataArray,
    alpha: float,
    debounce: int,
    dims: tuple[str, ...],
    log,
    not_depleted_message,
) -> xr.DataArray:
    """Depletion time = first segment where p_value > alpha holds for `debounce` consecutive
    segments, scanned independently for every (test_condition, *dims) combination (dims is
    e.g. ("qubit", "state") or just ("qubit",)). NaN (logged via not_depleted_message, called
    with the failing combination's coordinate values as kwargs) if no such run is found within
    the measured window -- "never reached significance" is a meaningfully different outcome
    from "reached it exactly at the last segment", so this never falls back to the last
    segment."""
    coords = {"test_condition": p_value.coords["test_condition"].values}
    coords.update({dim: p_value.coords[dim].values for dim in dims})
    shape = tuple(len(values) for values in coords.values())
    t_dep_ns = np.full(shape, np.nan)

    for index in np.ndindex(shape):
        sel = {name: values[i] for (name, values), i in zip(coords.items(), index)}
        segment_length = float(segment_length_ns.sel(qubit=sel["qubit"]))
        passed = p_value.sel(**sel).values > alpha
        segment_idx = _first_sustained_segment(passed, debounce)
        if segment_idx is not None:
            t_dep_ns[index] = segment_idx * segment_length
        else:
            log(not_depleted_message(**sel))

    return xr.DataArray(t_dep_ns, dims=tuple(coords.keys()), coords=coords)


def compute_stat_depletion_time(
    ds: xr.Dataset, node: QualibrationNode, test_conditions: tuple
) -> tuple[xr.DataArray, xr.DataArray]:
    """Statistical (chi-squared, df=2) alternative depletion-time test, per (test_condition,
    qubit, state): compares each test_condition's per-segment mean I/Q against the
    "no_operation" baseline via a two-sample z-test per quadrature, combined via _chi2_pvalue.
    Both samples' own pooled-over-segments variance contribute to the standard error (the same
    two-sample form compute_ge_depletion_time uses for ground vs excited).

    N_d(t) = N_n(t) = node.parameters.num_shots for every condition/state/segment: every
    condition and state is measured inside the same single QUA shot loop in
    create_qua_program, so the shot count streamed into every mean is identical and constant.
    """
    num_shots = node.parameters.num_shots
    alpha = node.parameters.alpha
    debounce = node.parameters.depletion_debounce_segments

    var_I_noop = ds["var_I"].sel(condition="no_operation").mean(dim="segment")
    var_Q_noop = ds["var_Q"].sel(condition="no_operation").mean(dim="segment")
    I_noop = ds["I"].sel(condition="no_operation")
    Q_noop = ds["Q"].sel(condition="no_operation")

    # Stacked along a NEW "test_condition" dim rather than reusing the existing "condition"
    # dim, which also contains "no_operation" -- avoids xarray reindex/alignment ambiguity
    # when test_conditions is a subset of the full condition coordinate.
    p_slices = []
    for condition in test_conditions:
        var_I_cond = ds["var_I"].sel(condition=condition).mean(dim="segment")
        var_Q_cond = ds["var_Q"].sel(condition=condition).mean(dim="segment")
        SE_I = np.sqrt((var_I_cond + var_I_noop) / num_shots)
        SE_Q = np.sqrt((var_Q_cond + var_Q_noop) / num_shots)
        dI = ds["I"].sel(condition=condition) - I_noop
        dQ = ds["Q"].sel(condition=condition) - Q_noop
        p_slices.append(_chi2_pvalue(dI, dQ, SE_I, SE_Q))

    test_condition_coord = xr.DataArray(list(test_conditions), dims="test_condition", name="test_condition")
    p_value = xr.concat(p_slices, dim=test_condition_coord)
    p_value.attrs = {"long_name": "p-value (chi2 test, df=2) vs no_operation baseline"}

    t_dep_stat_ns = _scan_depletion_time(
        p_value,
        ds["segment_length_ns"],
        alpha,
        debounce,
        dims=("qubit", "state"),
        log=node.log,
        not_depleted_message=lambda test_condition, qubit, state: (
            f"Statistical depletion test: {qubit} ({state}, vs {test_condition}) not depleted "
            "within the measured window."
        ),
    )
    t_dep_stat_ns.attrs = {
        "long_name": "statistical depletion time (chi2 test, p>alpha sustained)",
        "units": "ns",
    }

    return p_value, t_dep_stat_ns


def select_depletion_time(t_dep_stat_ns: xr.DataArray) -> xr.DataArray:
    """Depletion time per (test_condition, qubit): the longer of the ground and excited
    statistical-test times, so the resonator is empty for both states. NaN (not depleted within
    the window) if either state never depleted -- xarray's max would silently skip a NaN."""
    t_dep_ns = t_dep_stat_ns.max(dim="state", skipna=False)
    t_dep_ns.attrs = {
        "long_name": "depletion time (longer of ground/excited vs no_operation)",
        "units": "ns",
    }
    return t_dep_ns


def compute_ge_depletion_time(
    ds: xr.Dataset, node: QualibrationNode, test_conditions: tuple
) -> tuple[xr.DataArray, xr.DataArray]:
    """Third, independent depletion-time test: instead of comparing a test_condition against
    the "no_operation" baseline (as compute_stat_depletion_time does), directly compare the
    "ground" and "excited" state means against each other within that same condition -- if
    they're statistically indistinguishable, no state information remains in the readout,
    confirming depletion without relying on the no_operation reference at all.

    Same chi-squared (df=2) two-sample z-test machinery as compute_stat_depletion_time, but the
    two samples being compared are ground vs excited (each with node.parameters.num_shots
    shots) rather than test_condition vs no_operation -- so there is no "state" dimension in
    the output, just (test_condition, qubit).
    """
    num_shots = node.parameters.num_shots
    alpha = node.parameters.alpha
    debounce = node.parameters.depletion_debounce_segments

    p_slices = []
    for condition in test_conditions:
        var_I = (
            ds["var_I"].sel(condition=condition, state="ground").mean(dim="segment") / num_shots
            + ds["var_I"].sel(condition=condition, state="excited").mean(dim="segment") / num_shots
        )
        var_Q = (
            ds["var_Q"].sel(condition=condition, state="ground").mean(dim="segment") / num_shots
            + ds["var_Q"].sel(condition=condition, state="excited").mean(dim="segment") / num_shots
        )
        SE_I = np.sqrt(var_I)
        SE_Q = np.sqrt(var_Q)
        dI = ds["I"].sel(condition=condition, state="ground") - ds["I"].sel(condition=condition, state="excited")
        dQ = ds["Q"].sel(condition=condition, state="ground") - ds["Q"].sel(condition=condition, state="excited")
        p_slices.append(_chi2_pvalue(dI, dQ, SE_I, SE_Q))

    test_condition_coord = xr.DataArray(list(test_conditions), dims="test_condition", name="test_condition")
    p_value_ge = xr.concat(p_slices, dim=test_condition_coord)
    p_value_ge.attrs = {"long_name": "p-value (chi2 test, df=2) of ground vs excited distinguishability"}

    t_dep_ge_ns = _scan_depletion_time(
        p_value_ge,
        ds["segment_length_ns"],
        alpha,
        debounce,
        dims=("qubit",),
        log=node.log,
        not_depleted_message=lambda test_condition, qubit: (
            f"Ground-vs-excited depletion test: {qubit} (vs {test_condition}) not depleted "
            "within the measured window."
        ),
    )
    t_dep_ge_ns.attrs = {
        "long_name": "ground-vs-excited depletion time (chi2 test, p>alpha sustained)",
        "units": "ns",
    }

    return p_value_ge, t_dep_ge_ns


def log_depletion_summary(
    ds: xr.Dataset,
    t_dep_stat_ns: xr.DataArray,
    t_dep_ge_ns: xr.DataArray,
    test_conditions: tuple,
    ge_test_conditions: tuple,
    t_dep_ns: xr.DataArray | None = None,
) -> None:
    """Print the two depletion-time tests' results as plain tables (one row per
    qubit/state/condition), NaN rendered as "not depleted"."""

    def label(t_ns: float) -> str:
        return f"{t_ns:.0f} ns" if not np.isnan(t_ns) else "not depleted"

    print(f"{'qubit':6s} {'state':9s} {'condition':13s} {'t_dep (stat test)':s}")
    for condition in test_conditions:
        for qname in ds.qubit.values:
            for state in ds.state.values:
                t_ns = float(t_dep_stat_ns.sel(test_condition=condition, qubit=qname, state=state))
                print(f"{qname:6s} {state:9s} {condition:13s} {label(t_ns)}")

    print(f"{'qubit':6s} {'condition':13s} {'t_dep (ground-vs-excited test)':s}")
    for condition in ge_test_conditions:
        for qname in ds.qubit.values:
            t_ns = float(t_dep_ge_ns.sel(test_condition=condition, qubit=qname))
            print(f"{qname:6s} {condition:13s} {label(t_ns)}")

    if t_dep_ns is not None:
        print(f"{'qubit':6s} {'condition':13s} {'t_dep (longer of ground/excited, written to state)':s}")
        for condition in test_conditions:
            for qname in ds.qubit.values:
                print(f"{qname:6s} {condition:13s} {label(float(t_dep_ns.sel(test_condition=condition, qubit=qname)))}")

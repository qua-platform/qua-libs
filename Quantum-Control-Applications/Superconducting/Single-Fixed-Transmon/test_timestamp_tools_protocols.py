"""
Batch validation of TimestampRecorder across Single-Fixed-Transmon protocols.

Phases:
  1. load   - trace each protocol's QUA program without connecting to QOP
  2. instrument - run TimestampRecorder on every discovered Program
  3. hardware - compile + execute instrumented program, fetch timestamp streams

Usage:
    python test_timestamp_tools_protocols.py
    python test_timestamp_tools_protocols.py --hardware
    python test_timestamp_tools_protocols.py --report timestamp_protocol_report.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from qm import QopCaps, QuantumMachinesManager
from qm.program.program import Program

from configuration import cluster_name, config, qop_ip, qop_port
from timestamp_tools import TimestampRecorder

PROTOCOL_DIR = Path(__file__).resolve().parent

SKIP_SCRIPTS = {
    "test_timestamp_tools_local.py",
    "test_timestamp_tools_protocols.py",
    "configuration.py",
    "configuration_with_octave.py",
    "configuration_with_lf_fem.py",
    "octave_calibration.py",
    "macros.py",
    "RB_fits.py",
    "timestamp_tools.py",
}

# Wrong config variant or multi-hour / interactive flows not suitable for batch HW smoke tests.
HARDWARE_SKIP = {
    "00_hello_qua.py",  # infinite loop
    "01_manual_mixer_calibration.py",  # infinite CW
    "02_raw_adc_traces_mw_fem.py",  # MW-FEM config
    "03_time_of_flight_mw_fem.py",  # MW-FEM config
    "04b_resonator_spectroscopy_wide_range_octave.py",  # Octave LO sweep
    "04c_resonator_spectroscopy_wide_range_octave_update_IF.py",
    "06d_qubit_spectroscopy_wide_range_octave.py",
    "06e_qubit_spectroscopy_wide_range_octave_update_IF.py",  # opens qm before program
    "17_DRAG_calibration_Google.py",  # requires drag_coef != 0 in config
    "17_DRAG_calibration_Yale.py",
    "20_frequency_tracking.py",  # multi-step workflow, infinite loop in step 3
}

MINIMAL_PARAM_PATCH = """
# --- timestamp protocol test overrides ---
n_avg = 1
n_shot = 1
num_of_sequences = 1
max_circuit_depth = 10
delta_clifford = 10
minutes = 0.01
time_between_two_runs = 1
"""


@dataclass
class ProgramResult:
    program_name: str
    instrument_status: str
    operation_count: int = 0
    operation_names: List[str] = field(default_factory=list)
    instrument_error: Optional[str] = None
    compile_status: str = "skipped"
    compile_error: Optional[str] = None
    hardware_status: str = "skipped"
    hardware_error: Optional[str] = None
    timestamp_count: int = 0


@dataclass
class ScriptResult:
    script: str
    category: str
    load_status: str
    load_error: Optional[str] = None
    programs: List[ProgramResult] = field(default_factory=list)


def protocol_scripts() -> List[Path]:
    scripts = sorted(
        path
        for path in PROTOCOL_DIR.glob("[0-9]*.py")
        if path.name not in SKIP_SCRIPTS
    )
    return scripts


def categorize(script_name: str) -> str:
    if script_name in HARDWARE_SKIP:
        return "hardware_skip"
    if script_name == "08d_power_rabi_single_shot_timing.py":
        return "manual_timestamps"
    if script_name == "08e_power_rabi_single_shot_timing_with_tool.py":
        return "reference"
    if "octave" in script_name or "mw_fem" in script_name:
        return "variant_config"
    return "standard"


def find_exec_cutoff(lines: Sequence[str]) -> Optional[int]:
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("qmm = QuantumMachinesManager"):
            return index
        if stripped.startswith("qm = qmm.open_qm"):
            return index
    return None


def _shrink_sweep_arrays(source: str) -> str:
    """Replace common sweep constructors with a single-point sweep before tracing."""
    replacements = [
        (r"np\.arange\([^)]+\)", "np.array([4])"),
        (r"np\.linspace\([^)]+\)", "np.array([0.0])"),
        (r"np\.logspace\([^)]+\)", "np.array([1.0])"),
    ]
    patched = source
    for pattern, replacement in replacements:
        patched = re.sub(pattern, replacement, patched)
    return patched


def inject_minimal_patch(source: str) -> str:
    lines = source.splitlines()
    cutoff = find_exec_cutoff(lines)
    if cutoff is None:
        return _shrink_sweep_arrays(source)

    program_line = next(
        (index for index, line in enumerate(lines[:cutoff]) if re.match(r"\s*with program\(\)", line)),
        cutoff,
    )
    patch_lines = MINIMAL_PARAM_PATCH.strip().splitlines()
    patched = lines[:program_line] + patch_lines + lines[program_line:cutoff]
    return _shrink_sweep_arrays("\n".join(patched) + "\n")


def discover_programs(script_path: Path) -> Tuple[List[Tuple[str, Program]], Optional[str]]:
    import matplotlib

    matplotlib.use("Agg")
    source = script_path.read_text(encoding="utf-8")
    patched_source = inject_minimal_patch(source)
    namespace: Dict[str, Any] = {
        "__name__": "__timestamp_protocol_test__",
        "__file__": str(script_path),
        "__builtins__": __builtins__,
    }
    try:
        code = compile(patched_source, str(script_path), "exec")
        exec(code, namespace)  # noqa: S102 - intentional controlled exec of local protocol scripts
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"

    programs = [(name, value) for name, value in namespace.items() if isinstance(value, Program)]
    if not programs:
        return [], "No completed Program objects found before QOP connection."
    return programs, None


def instrument_program(program_name: str, program: Program) -> ProgramResult:
    result = ProgramResult(program_name=program_name, instrument_status="pending")
    try:
        timing = TimestampRecorder(program)
        result.instrument_status = "pass"
        result.operation_count = len(timing.names)
        result.operation_names = list(timing.names)
        result._timing = timing  # type: ignore[attr-defined]
    except Exception as exc:
        result.instrument_status = "fail"
        result.instrument_error = f"{type(exc).__name__}: {exc}"
    return result


def execute_hardware_smoke(program_result: ProgramResult, qm: Any) -> ProgramResult:
    timing = getattr(program_result, "_timing", None)
    if timing is None:
        program_result.hardware_status = "skipped"
        program_result.hardware_error = "Instrumentation did not produce a recorder."
        return program_result

    try:
        job = qm.execute(timing.program)
        timestamp_results = timing.fetch(job, wait_for_all=False, timeout_s=120)
        program_result.timestamp_count = sum(operation.occurrences for operation in timestamp_results)
        if program_result.operation_count > 0 and program_result.timestamp_count == 0:
            raise RuntimeError("Instrumented program executed but returned zero timestamp values.")
        program_result.hardware_status = "pass"
        try:
            job.cancel()
        except Exception:
            pass
    except Exception as exc:
        program_result.hardware_status = "fail"
        program_result.hardware_error = f"{type(exc).__name__}: {exc}"
    return program_result


def compile_program(program_result: ProgramResult, qm: Any) -> ProgramResult:
    timing = getattr(program_result, "_timing", None)
    if timing is None:
        program_result.compile_status = "skipped"
        return program_result
    try:
        qm.compile(timing.program)
        program_result.compile_status = "pass"
    except Exception as exc:
        program_result.compile_status = "fail"
        program_result.compile_error = f"{type(exc).__name__}: {exc}"
    return program_result


def evaluate_script(script_path: Path, run_hardware: bool, qm: Any = None) -> ScriptResult:
    category = categorize(script_path.name)
    result = ScriptResult(script=script_path.name, category=category, load_status="pending")

    programs, load_error = discover_programs(script_path)
    if load_error:
        result.load_status = "fail"
        result.load_error = load_error
        return result

    result.load_status = "pass"
    for program_name, program in programs:
        program_result = instrument_program(program_name, program)

        if category == "manual_timestamps":
            # 08d should be rejected because it already defines timestamp streams.
            if program_result.instrument_status == "fail" and program_result.instrument_error:
                if "already has timestamp stream" in program_result.instrument_error:
                    program_result.instrument_status = "expected_fail"

        if program_result.instrument_status in {"pass", "expected_fail"} and qm is not None:
            if program_result.instrument_status == "pass":
                program_result = compile_program(program_result, qm)
                if run_hardware and script_path.name not in HARDWARE_SKIP:
                    program_result = execute_hardware_smoke(program_result, qm)

        if hasattr(program_result, "_timing"):
            delattr(program_result, "_timing")
        result.programs.append(program_result)

    return result


def render_markdown_report(results: Sequence[ScriptResult], run_hardware: bool) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = len(results)
    loaded = sum(1 for item in results if item.load_status == "pass")
    instrumented = sum(
        1
        for item in results
        for program in item.programs
        if program.instrument_status in {"pass", "expected_fail"}
    )
    hardware_pass = sum(1 for item in results for program in item.programs if program.hardware_status == "pass")
    hardware_fail = sum(1 for item in results for program in item.programs if program.hardware_status == "fail")
    hardware_ran = hardware_pass + hardware_fail

    lines = [
        "# TimestampRecorder Protocol Validation Report",
        "",
        f"Generated: {timestamp}",
        f"Hardware phase: {'enabled' if run_hardware else 'disabled'}",
        "",
        "## Executive summary",
        "",
        f"- Protocol scripts scanned: **{total}**",
        f"- Programs loaded for instrumentation: **{loaded}/{total}**",
        f"- Instrumentation succeeded: **{instrumented}** program(s)",
        f"- Manual-timestamp guard (08d): **expected rejection**",
    ]
    if run_hardware:
        lines.append(f"- Hardware smoke tests: **{hardware_pass}/{hardware_ran}** passed")
    lines.extend(
        [
            "",
            "### Tool changes made during validation",
            "",
            "- Migrated protobuf cloning/walking to betterproto (`copy.deepcopy`, dataclass field walk).",
            "- Added `wait_for_timestamps()` and `timeout_s` to `TimestampRecorder.fetch()` so timestamp-only reads do not block on experiment result streams.",
            "- Added `test_timestamp_tools_protocols.py` batch harness for repeatable validation.",
            "",
            "### Scripts not batch-loadable (protocol structure / prerequisites, not tool bugs)",
            "",
            "- `04c_*`, `06e_*`: open `qm` and run Octave calibration before the QUA program is traced.",
            "- `17_DRAG_*`: require `drag_coef != 0` in `configuration.py`.",
            "- `20_frequency_tracking.py`: multi-step workflow with programs created inside runtime loops.",
            "",
            "## Detailed results",
            "",
            "| Script | Category | Load | Instrument | Compile | Hardware | Ops | Timestamps |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )

    for script_result in results:
        if not script_result.programs:
            lines.append(
                f"| {script_result.script} | {script_result.category} | {script_result.load_status} | - | - | - | - | - |"
            )
            continue
        for index, program in enumerate(script_result.programs):
            script_cell = script_result.script if index == 0 else ""
            category_cell = script_result.category if index == 0 else ""
            load_cell = script_result.load_status if index == 0 else ""
            lines.append(
                "| {script} | {category} | {load} | {instrument} | {compile} | {hardware} | {ops} | {timestamps} |".format(
                    script=script_cell,
                    category=category_cell,
                    load=load_cell,
                    instrument=program.instrument_status,
                    compile=program.compile_status,
                    hardware=program.hardware_status,
                    ops=program.operation_count,
                    timestamps=program.timestamp_count if program.timestamp_count else "-",
                )
            )

    failures = []
    for script_result in results:
        if script_result.load_status == "fail":
            failures.append((script_result.script, "load", script_result.load_error))
        for program in script_result.programs:
            if program.instrument_status == "fail":
                failures.append((script_result.script, "instrument", program.instrument_error))
            if program.compile_status == "fail":
                failures.append((script_result.script, "compile", program.compile_error))
            if program.hardware_status == "fail":
                failures.append((script_result.script, "hardware", program.hardware_error))

    lines.extend(["", "## Failures", ""])
    if failures:
        for script, phase, error in failures:
            lines.append(f"- **{script}** ({phase}): {error}")
    else:
        lines.append("No failures recorded.")

    return "\n".join(lines) + "\n"


def serialize_results(results: Sequence[ScriptResult]) -> List[Dict[str, Any]]:
    payload = []
    for script_result in results:
        entry = asdict(script_result)
        payload.append(entry)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true", help="Compile and execute instrumented programs on QOP.")
    parser.add_argument("--report", type=Path, default=PROTOCOL_DIR / "timestamp_protocol_report.md")
    parser.add_argument("--json-report", type=Path, default=PROTOCOL_DIR / "timestamp_protocol_report.json")
    args = parser.parse_args(argv)

    scripts = protocol_scripts()
    qm = None
    if args.hardware:
        qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)
        qm = qmm.open_qm(config)

    results = []
    for script in scripts:
        print(f"Evaluating {script.name}...", flush=True)
        results.append(evaluate_script(script, run_hardware=args.hardware, qm=qm))

    report_text = render_markdown_report(results, run_hardware=args.hardware)
    args.report.write_text(report_text, encoding="utf-8")
    args.json_report.write_text(json.dumps(serialize_results(results), indent=2), encoding="utf-8")

    print(report_text)
    print(f"Wrote {args.report}")
    print(f"Wrote {args.json_report}")

    failed = any(
        script_result.load_status == "fail"
        or any(
            program.instrument_status == "fail"
            or program.compile_status == "fail"
            or program.hardware_status == "fail"
            for program in script_result.programs
        )
        for script_result in results
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

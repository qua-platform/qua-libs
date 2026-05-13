"""One-off script: emit configuration_quam_comparison.html (run from this directory)."""

from __future__ import annotations

import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "configuration_quam_comparison.html"

JSON_FILES = {
    "manual": "quam_state_lf_mw_fem.json",
    "builder_full": "quam_state_builder_lf_mw_fem.json",
    "builder_components": "quam_state_builder_components_lf_mw_fem.json",
    "channel_manual": "quam_state_builder_channel_lf_mw_fem.json",
    "channel_wirer": "quam_state_builder_channel_wirer_lf_mw_fem.json",
}

ROWS = [
    {
        "id": "manual",
        "file": "configuration_quam_lf_fem_and_mw_fem.py",
        "label": "Manual (core QUAM)",
        "root": "BasicFEMQuam",
        "wiring": "None",
        "logical": "Flat machine.channels (MWChannel, InOutMWChannel, SingleChannel)",
        "lines": 283,
        "json": "quam_state_lf_mw_fem.json",
        "deps": "quam only (+ numpy, qualang_tools.units)",
    },
    {
        "id": "builder_full",
        "file": "configuration_quam_builder_lf_mw_fem.py",
        "label": "Builder — full pipeline",
        "root": "FluxTunableQuam",
        "wiring": "wirer + build_quam_wiring + build_quam",
        "logical": "machine.qubits['q1'] FluxTunableTransmon (xy / resonator / z)",
        "lines": 221,
        "json": "quam_state_builder_lf_mw_fem.json",
        "deps": "quam_builder + qualang_tools.wirer",
    },
    {
        "id": "builder_components",
        "file": "configuration_quam_builder_components_lf_mw_fem.py",
        "label": "Builder — components, no wirer",
        "root": "FluxTunableQuam",
        "wiring": "None",
        "logical": "machine.qubits['q1'] with XYDriveMW / ReadoutResonatorMW / FluxLine",
        "lines": 231,
        "json": "quam_state_builder_components_lf_mw_fem.json",
        "deps": "quam_builder (no wirer)",
    },
    {
        "id": "channel_manual",
        "file": "configuration_quam_builder_channel_lf_mw_fem.py",
        "label": "Builder channels — manual FEM",
        "root": "BasicFEMQuam",
        "wiring": "None",
        "logical": "Flat machine.channels (XYDriveMW, ReadoutResonatorMW, FluxLine)",
        "lines": 265,
        "json": "quam_state_builder_channel_lf_mw_fem.json",
        "deps": "quam + quam_builder components",
    },
    {
        "id": "channel_wirer",
        "file": "configuration_quam_builder_channel_wirer_lf_mw_fem.py",
        "label": "Builder channels — wirer FEM",
        "root": "BasicFEMQuam",
        "wiring": "allocate_wiring only (no build_quam_wiring)",
        "logical": "Flat machine.channels (same builder types)",
        "lines": 312,
        "json": "quam_state_builder_channel_wirer_lf_mw_fem.json",
        "deps": "quam + quam_builder + qualang_tools.wirer",
    },
]


def _json_bytes(fn: str) -> int:
    p = HERE / fn
    return p.stat().st_size if p.exists() else 0


def _safe_script_json(raw: str) -> str:
    return raw.replace("</script>", "<\\/script>")


def main() -> None:
    payloads: dict[str, str] = {}
    for key, fn in JSON_FILES.items():
        p = HERE / fn
        if p.exists():
            payloads[key] = _safe_script_json(p.read_text(encoding="utf-8"))
        else:
            payloads[key] = json.dumps({"error": f"missing file {fn}"})

    tab_labels = {r["id"]: r["label"] for r in ROWS}
    tab_order = [r["id"] for r in ROWS]

    parts: list[str] = []
    parts.append(
        """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QUAM configuration comparison — Single Flux-Tunable Transmon (LF/MW FEM)</title>
<style>
:root { --bg:#0f1419; --panel:#1a2332; --text:#e6edf3; --muted:#8b9cb3; --accent:#58a6ff; --ok:#3fb950; --warn:#d29922; }
* { box-sizing: border-box; }
body { margin:0; font-family: "Segoe UI", system-ui, sans-serif; background:var(--bg); color:var(--text); line-height:1.55; }
header { padding:2rem 1.5rem; background:linear-gradient(135deg,#1a2332 0%,#0f1419 100%); border-bottom:1px solid #30363d; }
header h1 { margin:0 0 .5rem; font-size:1.45rem; font-weight:600; }
header p { margin:0; color:var(--muted); max-width:58rem; }
nav { display:flex; flex-wrap:wrap; gap:.5rem; padding:1rem 1.5rem; background:var(--panel); position:sticky; top:0; z-index:10; border-bottom:1px solid #30363d; }
nav a { color:var(--accent); text-decoration:none; font-size:.9rem; }
nav a:hover { text-decoration:underline; }
main { max-width:72rem; margin:0 auto; padding:1.5rem; }
section { margin-bottom:2.5rem; scroll-margin-top:3.5rem; }
h2 { font-size:1.15rem; border-bottom:1px solid #30363d; padding-bottom:.35rem; margin-top:0; }
table { width:100%; border-collapse:collapse; font-size:.82rem; margin:1rem 0; }
th, td { border:1px solid #30363d; padding:.5rem .55rem; vertical-align:top; text-align:left; }
th { background:#21262d; color:#c9d1d9; font-weight:600; }
tr:nth-child(even) td { background:#161b22; }
.tag { display:inline-block; padding:.12rem .45rem; border-radius:4px; font-size:.72rem; font-weight:600; margin-right:.2rem; }
.tag-core { background:#23863633; color:var(--ok); }
.tag-builder { background:#1f6feb33; color:var(--accent); }
.tag-wirer { background:#d2992233; color:var(--warn); }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
@media (max-width:900px) { .grid2 { grid-template-columns:1fr; } }
.card { background:var(--panel); border:1px solid #30363d; border-radius:8px; padding:1rem 1.1rem; }
.card h3 { margin:0 0 .5rem; font-size:.95rem; }
ul.compact { margin:.3rem 0; padding-left:1.2rem; }
ul.compact li { margin:.2rem 0; }
pre.json { background:#0d1117; border:1px solid #30363d; border-radius:6px; padding:.75rem; overflow:auto; max-height:26rem; font-size:.72rem; line-height:1.35; margin:.5rem 0; white-space:pre; }
.tabs { display:flex; flex-wrap:wrap; gap:.35rem; margin:.75rem 0; }
.tabs button { background:#21262d; border:1px solid #30363d; color:var(--text); padding:.45rem .8rem; border-radius:6px; cursor:pointer; font-size:.78rem; }
.tabs button.active { background:var(--accent); color:#0f1419; border-color:var(--accent); }
.note { color:var(--muted); font-size:.88rem; }
footer { padding:1rem 1.5rem 2rem; color:var(--muted); font-size:.8rem; border-top:1px solid #30363d; }
code { font-size:.85em; }
</style>
</head>
<body>
<header>
<h1>Technical comparison: five QUAM configuration generators</h1>
<p>Folder: <code>Quantum-Control-Applications/Superconducting/Single-Flux-Tunable-Transmon/</code>. Compares manual QUAM vs quam-builder transmon-centric vs flat builder-channel layouts, wiring choices, JSON state shape, readability, and trade-offs. JSON snapshots are embedded for offline viewing.</p>
</header>
<nav>
<a href="#matrix">Matrix</a>
<a href="#similarities">Similarities</a>
<a href="#differences">Differences</a>
<a href="#readability">Readability</a>
<a href="#json">JSON</a>
</nav>
<main>
"""
    )

    parts.append('<section id="matrix"><h2>1. Configuration matrix</h2>')
    parts.append(
        "<table><thead><tr><th>Variant</th><th>Root QUAM</th><th>Wiring</th>"
        "<th>Logical model</th><th>Lines (.py)</th><th>State JSON</th></tr></thead><tbody>"
    )
    for r in ROWS:
        tags = '<span class="tag tag-core">quam</span>'
        if "quam_builder" in r["deps"]:
            tags += '<span class="tag tag-builder">quam-builder</span>'
        if "wirer" in r["deps"]:
            tags += '<span class="tag tag-wirer">wirer</span>'
        jb = _json_bytes(r["json"])
        parts.append(
            f"<tr><td><strong>{html.escape(r['label'])}</strong><br>"
            f"<code>{html.escape(r['file'])}</code><br>{tags}</td>"
            f"<td>{html.escape(r['root'])}</td>"
            f"<td>{html.escape(r['wiring'])}</td>"
            f"<td>{html.escape(r['logical'])}</td>"
            f"<td>{r['lines']}</td>"
            f"<td><code>{html.escape(r['json'])}</code><br>{jb} bytes</td></tr>"
        )
    parts.append("</tbody></table></section>")

    parts.append(
        """
<section id="similarities"><h2>2. Similarities</h2>
<div class="grid2">
<div class="card"><h3>Hardware & pulses</h3>
<ul class="compact">
<li>Identical LF/MW FEM topology: resonator in+out on MW-FEM, XY on a second MW output, flux on LF-FEM (plus sticky flux alias).</li>
<li>Same pulse library pattern as the manual reference: DRAG Gaussian gates on the qubit drive, square readout, square flux const.</li>
<li>All variants expose <code>config = machine.generate_config()</code> and the same legacy scalar exports at module bottom.</li>
</ul></div>
<div class="card"><h3>Persistence</h3>
<ul class="compact">
<li>Each Python entry point saves a sibling JSON state file on first import; later imports load from JSON.</li>
<li>Builder MW channel types may require clearing string-reference frequency fields before assigning numeric LO/IF/RF (handled in the builder-based modules).</li>
</ul></div>
</div></section>
"""
    )

    parts.append(
        """
<section id="differences"><h2>3. Differences & discrepancies</h2>
<div class="grid2">
<div class="card"><h3>JSON & object graph</h3>
<ul class="compact">
<li><strong>BasicFEMQuam</strong> variants serialize <code>channels</code> + <code>ports</code> + empty <code>octaves</code>; no <code>qubits</code> or <code>wiring</code>.</li>
<li><strong>FluxTunableQuam</strong> variants add <code>qubits</code>, <code>active_qubit_names</code>, <code>wiring</code>, <code>network</code>, and empty <code>mixers</code>/<code>twpas</code> — larger JSON and more indirection.</li>
<li><strong>Class paths in JSON</strong>: manual uses <code>quam.components.channels.*</code>; builder-based use <code>quam_builder.architecture.superconducting.components.*</code> for XY/resonator/flux.</li>
</ul></div>
<div class="card"><h3>Code paths</h3>
<ul class="compact">
<li><strong>Full builder</strong> is the only variant that calls <code>build_quam_wiring</code> and <code>build_quam</code> — maximum automation, minimum explicit port typing in Python.</li>
<li><strong>Channel + wirer</strong> uses allocation results to call <code>get_mw_output</code> / <code>get_mw_input</code> / <code>get_analog_output</code> — good when hardware addressing should follow the same allocator as larger apps.</li>
<li><strong>Discrepancy</strong>: line count is not complexity: the full builder module is shorter because heavy lifting lives in libraries.</li>
</ul></div>
</div></section>
"""
    )

    parts.append(
        """
<section id="readability"><h2>4. Readability & simplicity (qualitative)</h2>
<table><thead><tr><th>Variant</th><th>Newcomer readability</th><th>Scaling to many qubits</th><th>Explicitness / “magic”</th></tr></thead><tbody>
<tr><td>Manual</td><td>Best — only core QUAM types, linear flow.</td><td>Poor — hand duplication.</td><td>Maximum explicitness.</td></tr>
<tr><td>Builder full</td><td>Hardest — reload + builder steps.</td><td>Best — intended for generated machines.</td><td>Most magic in one place.</td></tr>
<tr><td>Builder components</td><td>Medium — transmon object + manual FEM.</td><td>Medium.</td><td>Balanced.</td></tr>
<tr><td>Builder channels (manual FEM)</td><td>High — same as manual but richer types.</td><td>Low–medium.</td><td>High explicitness, more typing for pulses.</td></tr>
<tr><td>Builder channels + wirer</td><td>Medium — extra wirer vocabulary.</td><td>Medium — addresses from allocator.</td><td>Medium.</td></tr>
</tbody></table>
<p class="note">Suggested use: <strong>manual</strong> for tutorials; <strong>builder full</strong> for production-scale QUAM generation; <strong>builder channels</strong> when you want builder channel helpers without a transmon wrapper; <strong>builder components</strong> when you want transmon helpers but not wirer.</p>
</section>
"""
    )

    parts.append('<section id="json"><h2>5. Embedded JSON state files</h2>')
    parts.append(
        '<p class="note">Select a tab to pretty-print the embedded snapshot. '
        "If a file was missing at build time, the tab shows an error object.</p>"
    )
    parts.append('<div class="tabs" id="json-tabs"></div>')
    parts.append('<pre class="json" id="json-panel"></pre>')

    for key, raw in payloads.items():
        parts.append(f'<script type="application/json" id="embed-{key}">\n{raw}\n</script>')

    tab_meta = [{"id": k, "label": tab_labels[k], "fn": JSON_FILES[k]} for k in tab_order]
    parts.append(
        "<script>\n"
        "(function(){\n"
        "  var meta = "
        + json.dumps(tab_meta)
        + ";\n"
        "  var tabs = document.getElementById('json-tabs');\n"
        "  var panel = document.getElementById('json-panel');\n"
        "  function show(key){\n"
        "    var el = document.getElementById('embed-' + key);\n"
        "    if(!el){ panel.textContent = 'Missing embed for ' + key; return; }\n"
        "    try {\n"
        "      var obj = JSON.parse(el.textContent);\n"
        "      panel.textContent = JSON.stringify(obj, null, 2);\n"
        "    } catch(e) { panel.textContent = el.textContent; }\n"
        "    [].forEach.call(tabs.querySelectorAll('button'), function(b){\n"
        "      b.classList.toggle('active', b.getAttribute('data-key') === key);\n"
        "    });\n"
        "  }\n"
        "  meta.forEach(function(m, i){\n"
        "    var b = document.createElement('button');\n"
        "    b.textContent = m.label;\n"
        "    b.setAttribute('data-key', m.id);\n"
        "    b.addEventListener('click', function(){ show(m.id); });\n"
        "    tabs.appendChild(b);\n"
        "    if(i===0) b.classList.add('active');\n"
        "  });\n"
        "  if(meta.length) show(meta[0].id);\n"
        "})();\n"
        "</script>\n"
    )

    parts.append(
        """
</main>
<footer>
Generated by <code>_build_config_comparison_html.py</code> in this folder. Re-run after changing JSON or Python configs to refresh embedded data.
</footer>
</body>
</html>
"""
    )

    OUT.write_text("".join(parts), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()

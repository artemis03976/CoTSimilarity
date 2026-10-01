"""Visualization script for DAG analysis results."""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import html
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import sort_records
from utils.io import atomic_text
from utils.web_report import (
    build_common_css,
    build_navigation_html,
    build_escape_html_js,
    build_navigation_js,
    build_show_tab_js,
    build_runtime_bootstrap_js,
    build_dag_helpers_js,
)


from utils.visualization.dag_records import compress_dag_analysis, load_analyzed_records


def generate_dag_graph(dag_analysis: List[Dict]) -> str:
    """Generate Mermaid diagram for dependency DAG (compressed format).

    Args:
        dag_analysis: List of dependency objects (compressed or original)

    Returns:
        Mermaid diagram code
    """
    lines = ["graph TD"]
    lines.append("    Problem[\"[0] Problem\"]")

    # Add nodes for each step in dag_analysis
    for dep in dag_analysis:
        step_id = dep["step_id"]
        tag = dep.get("macro_action_tag", "")
        merged_label = dep.get("merged_label", str(step_id))

        # Simplified label: [merged_ids] Tag
        if tag:
            label = f"[{merged_label}] {tag}"
        else:
            label = f"[{merged_label}]"

        lines.append(f"    Step{step_id}[\"{label}\"]")

    # Add edges based on dependencies
    for dep in dag_analysis:
        step_id = dep["step_id"]
        depends_on = dep["depends_on"]

        for dependency in depends_on:
            if dependency == 0:
                lines.append(f"    Problem --> Step{step_id}")
            elif dependency == "External":
                if "External" not in [line.split("[")[0].strip() for line in lines]:
                    lines.append("    External[\"[Ext] External\"]")
                lines.append(f"    External -.-> Step{step_id}")
            else:
                lines.append(f"    Step{dependency} --> Step{step_id}")

    # Add styling
    lines.append("    style Problem fill:#e1f5ff")
    lines.append("    style External fill:#fff4e1")

    return "\n".join(lines)


def generate_dependency_table(dag_analysis: List[Dict]) -> str:
    """Generate HTML table for dependency analysis (compressed format)."""
    rows = []
    for dep in dag_analysis:
        merged_label = dep.get("merged_label", str(dep["step_id"]))
        depends_on = dep["depends_on"]
        tag = dep.get("macro_action_tag", "N/A")

        # Format dependencies
        dep_str = ", ".join([
            f"[{d}]" if isinstance(d, int) else f"[{d}]"
            for d in depends_on
        ])

        # Color code by dependency type
        if depends_on == [0]:
            dep_class = "dep-problem"
        elif "External" in depends_on:
            dep_class = "dep-external"
        else:
            dep_class = "dep-steps"

        # Color code by tag
        tag_class = f"tag-{tag.lower()}" if tag != "N/A" else ""

        rows.append(f"""
        <tr>
            <td class="step-id">{merged_label}</td>
            <td class="{tag_class}">{tag}</td>
            <td class="analysis">{html.escape(dep.get("analysis", ""))}</td>
            <td class="{dep_class}">{dep_str}</td>
        </tr>
        """)

    return "\n".join(rows)


def generate_statistics(records: List[Dict]) -> Dict:
    """Generate statistics from analyzed records."""
    stats = {
        "total_records": len(records),
        "total_variants": 0,
        "total_steps": 0,
        "total_dependencies": 0,
        "dependency_types": {
            "problem": 0,
            "external": 0,
            "steps": 0
        },
        "avg_steps_per_variant": 0,
        "avg_deps_per_step": 0
    }

    variant_count = 0
    for record in records:
        for variant in ["original", "simple", "hard"]:
            if variant not in record:
                continue

            entry = record[variant]
            if "dag_analysis" not in entry or entry["dag_analysis"] is None:
                continue

            variant_count += 1
            dag = entry["dag_analysis"]
            stats["total_steps"] += len(dag)

            for dep in dag:
                depends_on = dep["depends_on"]
                stats["total_dependencies"] += len(depends_on)

                if depends_on == [0]:
                    stats["dependency_types"]["problem"] += 1
                elif "External" in depends_on:
                    stats["dependency_types"]["external"] += 1
                else:
                    stats["dependency_types"]["steps"] += 1

    stats["total_variants"] = variant_count
    if variant_count > 0:
        stats["avg_steps_per_variant"] = stats["total_steps"] / variant_count
    if stats["total_steps"] > 0:
        stats["avg_deps_per_step"] = stats["total_dependencies"] / stats["total_steps"]

    return stats


def generate_html_with_js(stats: Dict, records_json: str) -> str:
    """Generate HTML content with embedded JavaScript (avoiding f-string escaping issues)."""

    css_styles = build_common_css()
    navigation_html = build_navigation_html(stats["total_records"], include_random=True)
    shared_escape_html_js = build_escape_html_js()
    shared_dag_js = build_dag_helpers_js(graph_fn_name="generateDagGraph")
    shared_navigation_js = build_navigation_js(include_random=True)
    shared_tab_js = build_show_tab_js(include_mermaid=True)
    shared_bootstrap_js = build_runtime_bootstrap_js(include_mermaid_init=True)

    # Build HTML using format() instead of f-string to avoid escaping issues
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CoT Dependency Analysis Viewer</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>{css}</style>
</head>
<body>
    <div class="container">
        <h1>🔍 CoT Reasoning Dependency Analysis</h1>
        <p style="color: #666; margin-bottom: 30px;">Chain-of-Thought Dependency DAG Analysis</p>

        <h2>📊 Overall Statistics</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total Records</div>
                <div class="stat-value">{total_records}</div>
            </div>
            <div class="stat-card green">
                <div class="stat-label">Total Variants</div>
                <div class="stat-value">{total_variants}</div>
            </div>
            <div class="stat-card orange">
                <div class="stat-label">Total Reasoning Steps</div>
                <div class="stat-value">{total_steps}</div>
            </div>
            <div class="stat-card blue">
                <div class="stat-label">Average Steps</div>
                <div class="stat-value">{avg_steps:.1f}</div>
            </div>
        </div>

        <h2>🎯 Dependency Type Distribution</h2>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color problem"></div>
                <span>Depends on original problem [0]: {dep_problem}</span>
            </div>
            <div class="legend-item">
                <div class="legend-color external"></div>
                <span>Depends on external knowledge [External]: {dep_external}</span>
            </div>
            <div class="legend-item">
                <div class="legend-color steps"></div>
                <span>Depends on previous steps: {dep_steps}</span>
            </div>
        </div>

        <h2>🔍 Browse Records</h2>
        {navigation_html}

        <div id="recordContainer"></div>
    </div>

    <script>
        const allRecords = {records};
        let currentIndex = 0;
        let mermaidCounter = 0;

        {shared_escape_html_js}
        {shared_dag_js}

        async function renderRecord(index) {{
            const record = allRecords[index];
            const problemId = record.problem_id;
            const sampleIndex = record.sample_index ?? 0;
            const variants = ['original', 'simple', 'hard'].filter(name => record[name]);
            const firstVariant = variants[0];
            const problemType = escapeHtml(record.type || 'Unknown');
            const level = escapeHtml(record.level || 'Unknown');

            let html = `
                <h2>📝 Record ${{index + 1}}: Problem ID ${{problemId}}</h2>
                <div class="record">
                    <div class="record-header">
                        <div>
                            <span class="record-title">Problem ${{problemId}} / Sample ${{sampleIndex}}</span>
                            <span class="badge variant">${{problemType}}</span>
                            <span class="badge variant">${{level}}</span>
                        </div>
                    </div>
                    <div class="tabs">
                        ${{variants.map(name => `<button class="tab ${{name === firstVariant ? 'active' : ''}}" onclick="showTab(event, '${{name}}')">${{name[0].toUpperCase() + name.slice(1)}}</button>`).join('')}}
                    </div>
            `;

            variants.forEach(variantName => {{
                if (record[variantName]) {{
                    const entry = record[variantName];
                    const isActive = variantName === firstVariant ? 'active' : '';

                    html += `<div id="${{variantName}}" class="tab-content ${{isActive}}">`;

                    if (!entry.dag_analysis) {{
                        html += '<p style="color: #e74c3c; padding: 20px;">⚠️ No dependency analysis data for this variant</p>';
                    }} else {{
                        if ('correct' in entry) {{
                            const correctClass = entry.correct ? 'correct' : 'incorrect';
                            const correctText = entry.correct ? '✓ Correct' : '✗ Incorrect';
                            html += `<span class="badge ${{correctClass}}">${{correctText}}</span>`;
                        }}

                        if (entry.dag_metadata) {{
                            const meta = entry.dag_metadata;
                            html += `
                                <div class="metadata">
                                    <div class="metadata-item">
                                        <span class="metadata-label">Model:</span>
                                        <span>${{meta.model || 'N/A'}}</span>
                                    </div>
                                    <div class="metadata-item">
                                        <span class="metadata-label">Processing Time:</span>
                                        <span>${{((meta.processing_time_ms || 0) / 1000).toFixed(2)}}s</span>
                                    </div>
                                    <div class="metadata-item">
                                        <span class="metadata-label">Steps:</span>
                                        <span>${{entry.num_steps || 0}}</span>
                                    </div>
                                </div>
                            `;
                        }}

                        const problemText = entry.problem || '';
                        const displayText = problemText.length > 500 ? problemText.substring(0, 500) + '...' : problemText;
                        html += `
                            <h3>Problem</h3>
                            <div class="problem-box">
                                <div class="problem-text">${{escapeHtml(displayText)}}</div>
                            </div>
                        `;

                        const dagData = entry.dag_analysis_compressed || entry.dag_analysis;
                        const dagGraph = generateDagGraph(dagData);
                        const mermaidId = `mermaid-${{index}}-${{variantName}}-${{mermaidCounter++}}`;
                        html += `
                            <h3>Dependency Graph (DAG) - Compressed View</h3>
                            <div class="mermaid" id="${{mermaidId}}">
${{dagGraph}}
                            </div>
                        `;

                        const depTable = generateDependencyTable(dagData);
                        html += `
                            <h3>Dependency Analysis Details</h3>
                            <table>
                                <thead>
                                    <tr>
                                        <th>Step</th>
                                        <th>Action Type</th>
                                        <th>Analysis</th>
                                        <th>Dependencies</th>
                                    </tr>
                                </thead>
                                <tbody>
${{depTable}}
                                </tbody>
                            </table>
                        `;
                    }}

                    html += '</div>';
                }}
            }});

            html += '</div>';
            document.getElementById('recordContainer').innerHTML = html;
            updateNavigation();

            const activeMermaidElements = document.querySelectorAll('.tab-content.active .mermaid');
            for (const element of activeMermaidElements) {{
                if (!element.hasAttribute('data-processed')) {{
                    try {{
                        await mermaid.run({{ nodes: [element] }});
                    }} catch (err) {{
                        console.error('Mermaid error:', err);
                        element.innerHTML = '<p style="color: red;">Failed to render diagram</p>';
                    }}
                }}
            }}
        }}

        {shared_navigation_js}
        {shared_tab_js}
        {shared_bootstrap_js}
    </script>
</body>
</html>
"""

    return html.format(
        css=css_styles,
        total_records=stats['total_records'],
        total_variants=stats['total_variants'],
        total_steps=stats['total_steps'],
        avg_steps=stats['avg_steps_per_variant'],
        dep_problem=stats['dependency_types']['problem'],
        dep_external=stats['dependency_types']['external'],
        dep_steps=stats['dependency_types']['steps'],
        records=records_json,
        navigation_html=navigation_html,
        shared_escape_html_js=shared_escape_html_js,
        shared_dag_js=shared_dag_js,
        shared_navigation_js=shared_navigation_js,
        shared_tab_js=shared_tab_js,
        shared_bootstrap_js=shared_bootstrap_js
    )


def generate_html_report(
    records: List[Dict],
    output_path: str,
    limit: Optional[int] = None,
    compress: bool = True
):
    """Generate comprehensive HTML report with client-side pagination and rendering."""

    if limit:
        records = records[:limit]

    # Pre-compress DAG analysis for each variant if requested
    if compress:
        print("Applying DAG compression...")
        for record in records:
            for variant in ["original", "simple", "hard"]:
                if variant not in record:
                    continue
                entry = record[variant]
                if "dag_analysis" not in entry or entry["dag_analysis"] is None:
                    continue
                entry["dag_analysis_compressed"] = compress_dag_analysis(entry["dag_analysis"])

    stats = generate_statistics(records)

    # Serialize records to JSON for client-side rendering
    records_json = json.dumps(records, ensure_ascii=False)
    # Prevent embedded JSON from breaking <script> parsing in browsers.
    records_json = (
        records_json
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )

    # Generate HTML with embedded JavaScript
    # Use string concatenation to avoid f-string escaping issues with JavaScript
    html_content = generate_html_with_js(stats, records_json)

    with atomic_text(output_path) as f:
        f.write(html_content)

    print(f"HTML report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize DAG analysis results")
    parser.add_argument("--input", type=str,
                       default="output/qwen-2.5/dag_analysis/analyzed_records.jsonl",
                       help="Input analysis result file")
    parser.add_argument("--output", type=str,
                       default="output/qwen-2.5/dag_analysis/visualization.html",
                       help="Output HTML file")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit the number of displayed records")
    parser.add_argument("--no-compress", action="store_true",
                       help="Disable DAG compression and show the original full graph")
    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return

    # Load records
    print(f"Loading data from: {args.input}")
    records = load_analyzed_records(args.input)
    print(f"Loaded {len(records)} records")

    # Sort records by problem_id
    records = sort_records(records)

    # Generate HTML report
    print(f"Generating visualization report...")
    generate_html_report(records, args.output, args.limit, compress=not args.no_compress)

    print(f"\nDone! Open in browser:")
    print(f"   {Path(args.output).absolute()}")


if __name__ == "__main__":
    main()

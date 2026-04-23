"""Visualization script for DAG analysis results."""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import html
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_analysis.dag_compressor import build_digraph_with_tags, compress_dag_combined
from utils import extract_numeric_id, sort_records
from utils.web_report import (
    build_common_css,
    build_navigation_html,
    build_escape_html_js,
    build_navigation_js,
    build_show_tab_js,
    build_runtime_bootstrap_js,
    build_dag_helpers_js,
)


def compress_dag_analysis(dag_analysis: List[Dict], exclude_external: bool = False) -> List[Dict]:
    """Compress dag_analysis using serial absorption and return simplified structure.

    Args:
        dag_analysis: Original list of dependency objects
        exclude_external: Whether to exclude External nodes

    Returns:
        Compressed dag_analysis list with merged step IDs in labels
    """
    if not dag_analysis or len(dag_analysis) == 0:
        return dag_analysis

    try:
        G = build_digraph_with_tags(dag_analysis, exclude_external=exclude_external)
        G_compressed, stats = compress_dag_combined(G, merge_metadata=True)
    except Exception as e:
        print(f"Warning: Compression failed, using original: {e}")
        return dag_analysis

    # Build compressed dag_analysis from the compressed graph
    compressed = []
    for node in sorted(n for n in G_compressed.nodes if isinstance(n, int) and n > 0):
        attrs = G_compressed.nodes[node]
        tag = attrs.get('macro_action_tag', '')
        absorbed = attrs.get('absorbed_nodes', [])

        # Build merged ID label: e.g. "1,2,3" if node 1 absorbed 2 and 3
        all_ids = sorted([node] + absorbed)
        merged_label = ",".join(str(i) for i in all_ids)

        # Collect dependencies from compressed graph predecessors
        depends_on = []
        for pred in G_compressed.predecessors(node):
            if pred == "External":
                depends_on.append("External")
            else:
                depends_on.append(pred)

        compressed.append({
            "step_id": node,
            "merged_ids": all_ids,
            "merged_label": merged_label,
            "macro_action_tag": tag or "N/A",
            "depends_on": depends_on,
            "analysis": attrs.get('analysis', ''),
        })

    # Check if External node exists in compressed graph
    has_external = "External" in G_compressed.nodes

    return compressed


def extract_dag_from_batch_response(response_data: Dict) -> Optional[List[Dict]]:
    """Extract DAG analysis from batch inference response format.

    Args:
        response_data: Raw response data that may be in batch format

    Returns:
        Parsed DAG analysis list, or None if extraction fails
    """
    try:
        # Check if this is batch inference format
        if "response" in response_data and "body" in response_data["response"]:
            # Extract content from batch response
            content = response_data["response"]["body"]["choices"][0]["message"]["content"]

            # Parse the JSON content (it's a string containing JSON)
            # Remove markdown code blocks if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]  # Remove ```

            # Find the end of the JSON array
            # Look for the closing bracket followed by optional whitespace and ```
            import re
            # Match JSON array and stop at the first complete array
            match = re.search(r'(\[[\s\S]*?\])\s*```?', content)
            if match:
                content = match.group(1)
            else:
                # Try without the ``` at the end
                match = re.search(r'(\[[\s\S]*?\])', content)
                if match:
                    content = match.group(1)
                else:
                    # Check if the array is incomplete (truncated response)
                    if content.strip().startswith('[') and not content.strip().endswith(']'):
                        # Try to fix by adding closing bracket
                        content = content.strip()
                        # Remove any incomplete object at the end
                        last_complete = content.rfind('},')
                        if last_complete > 0:
                            content = content[:last_complete + 1] + '\n]'
                        else:
                            # No complete objects found
                            return None
                    else:
                        # Just try to find the end of array
                        if content.endswith("```"):
                            content = content[:-3]
                        content = content.strip()

            # Parse the JSON using a more robust approach
            # Try to find the JSON array boundaries
            decoder = json.JSONDecoder()
            try:
                dag_analysis, _ = decoder.raw_decode(content)
                return dag_analysis
            except json.JSONDecodeError:
                # Fallback to simple parse
                dag_analysis = json.loads(content)
                return dag_analysis
        else:
            # Already in the correct format
            return response_data
    except (KeyError, json.JSONDecodeError, IndexError):
        # Silently return None for unparseable responses
        return None


def compress_dag_analysis(dag_analysis: List[Dict], exclude_external: bool = False) -> List[Dict]:
    """Compress dag_analysis using serial absorption and return simplified structure.

    Args:
        dag_analysis: Original list of dependency objects
        exclude_external: Whether to exclude External nodes

    Returns:
        Compressed dag_analysis list with merged step IDs in labels
    """
    if not dag_analysis or len(dag_analysis) == 0:
        return dag_analysis

    try:
        G = build_digraph_with_tags(dag_analysis, exclude_external=exclude_external)
        G_compressed, stats = compress_dag_combined(G, merge_metadata=True)
    except Exception as e:
        print(f"Warning: Compression failed, using original: {e}")
        return dag_analysis

    compressed = []
    for node in sorted(n for n in G_compressed.nodes if isinstance(n, int) and n > 0):
        attrs = G_compressed.nodes[node]
        tag = attrs.get('macro_action_tag', '')
        absorbed = attrs.get('absorbed_nodes', [])

        all_ids = sorted([node] + absorbed)
        merged_label = ",".join(str(i) for i in all_ids)

        depends_on = []
        for pred in G_compressed.predecessors(node):
            if pred == "External":
                depends_on.append("External")
            else:
                depends_on.append(pred)

        compressed.append({
            "step_id": node,
            "merged_ids": all_ids,
            "merged_label": merged_label,
            "macro_action_tag": tag or "N/A",
            "depends_on": depends_on,
            "analysis": attrs.get('analysis', ''),
        })

    return compressed


def load_analyzed_records(input_path: str) -> List[Dict]:
    """Load analyzed records from JSONL file and clean batch inference format."""
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                raw_record = json.loads(line)

                if "custom_id" in raw_record and "response" in raw_record:
                    custom_id = raw_record["custom_id"]
                    parts = custom_id.rsplit("_", 1)
                    if len(parts) != 2:
                        print(f"Warning: Invalid custom_id format at line {line_num}: {custom_id}")
                        continue

                    problem_id = parts[0]
                    variant = parts[1]

                    dag_analysis = extract_dag_from_batch_response(raw_record)

                    if dag_analysis is None:
                        print(f"Warning: Failed to extract DAG for {custom_id} at line {line_num}")
                        continue

                    record = None
                    for r in records:
                        if r.get("problem_id") == problem_id:
                            record = r
                            break

                    if record is None:
                        record = {
                            "problem_id": problem_id,
                            "type": "Unknown",
                            "level": "Unknown"
                        }
                        records.append(record)

                    record[variant] = {
                        "dag_analysis": dag_analysis,
                        "problem": f"Problem {problem_id} ({variant} variant)",
                        "steps": [],
                        "num_steps": len(dag_analysis) if dag_analysis else 0
                    }
                else:
                    records.append(raw_record)

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON at line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue

    return records


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
    <title>CoT 依赖分析可视化</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>{css}</style>
</head>
<body>
    <div class="container">
        <h1>🔍 CoT 推理链依赖分析可视化</h1>
        <p style="color: #666; margin-bottom: 30px;">Chain-of-Thought Dependency DAG Analysis</p>

        <h2>📊 总体统计</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">总记录数</div>
                <div class="stat-value">{total_records}</div>
            </div>
            <div class="stat-card green">
                <div class="stat-label">总变体数</div>
                <div class="stat-value">{total_variants}</div>
            </div>
            <div class="stat-card orange">
                <div class="stat-label">总推理步骤</div>
                <div class="stat-value">{total_steps}</div>
            </div>
            <div class="stat-card blue">
                <div class="stat-label">平均步骤数</div>
                <div class="stat-value">{avg_steps:.1f}</div>
            </div>
        </div>

        <h2>🎯 依赖类型分布</h2>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color problem"></div>
                <span>依赖原问题 [0]: {dep_problem}</span>
            </div>
            <div class="legend-item">
                <div class="legend-color external"></div>
                <span>依赖外部知识 [External]: {dep_external}</span>
            </div>
            <div class="legend-item">
                <div class="legend-color steps"></div>
                <span>依赖前序步骤: {dep_steps}</span>
            </div>
        </div>

        <h2>🔍 浏览记录</h2>
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
            const problemType = escapeHtml(record.type || 'Unknown');
            const level = escapeHtml(record.level || 'Unknown');

            let html = `
                <h2>📝 记录 ${{index + 1}}: Problem ID ${{problemId}}</h2>
                <div class="record">
                    <div class="record-header">
                        <div>
                            <span class="record-title">Problem ${{problemId}}</span>
                            <span class="badge variant">${{problemType}}</span>
                            <span class="badge variant">${{level}}</span>
                        </div>
                    </div>
                    <div class="tabs">
                        <button class="tab active" onclick="showTab(event, 'original')">Original</button>
                        <button class="tab" onclick="showTab(event, 'simple')">Simple</button>
                        <button class="tab" onclick="showTab(event, 'hard')">Hard</button>
                    </div>
            `;

            ['original', 'simple', 'hard'].forEach((variantName, idx) => {{
                if (record[variantName]) {{
                    const entry = record[variantName];
                    const isActive = idx === 0 ? 'active' : '';

                    html += `<div id="${{variantName}}" class="tab-content ${{isActive}}">`;

                    if (!entry.dag_analysis) {{
                        html += '<p style="color: #e74c3c; padding: 20px;">⚠️ 此变体没有依赖分析数据</p>';
                    }} else {{
                        if ('correct' in entry) {{
                            const correctClass = entry.correct ? 'correct' : 'incorrect';
                            const correctText = entry.correct ? '✓ 正确' : '✗ 错误';
                            html += `<span class="badge ${{correctClass}}">${{correctText}}</span>`;
                        }}

                        if (entry.dag_metadata) {{
                            const meta = entry.dag_metadata;
                            html += `
                                <div class="metadata">
                                    <div class="metadata-item">
                                        <span class="metadata-label">模型:</span>
                                        <span>${{meta.model || 'N/A'}}</span>
                                    </div>
                                    <div class="metadata-item">
                                        <span class="metadata-label">处理时间:</span>
                                        <span>${{((meta.processing_time_ms || 0) / 1000).toFixed(2)}}s</span>
                                    </div>
                                    <div class="metadata-item">
                                        <span class="metadata-label">步骤数:</span>
                                        <span>${{entry.num_steps || 0}}</span>
                                    </div>
                                </div>
                            `;
                        }}

                        const problemText = entry.problem || '';
                        const displayText = problemText.length > 500 ? problemText.substring(0, 500) + '...' : problemText;
                        html += `
                            <h3>问题描述</h3>
                            <div class="problem-box">
                                <div class="problem-text">${{escapeHtml(displayText)}}</div>
                            </div>
                        `;

                        const dagData = entry.dag_analysis_compressed || entry.dag_analysis;
                        const dagGraph = generateDagGraph(dagData);
                        const mermaidId = `mermaid-${{index}}-${{variantName}}-${{mermaidCounter++}}`;
                        html += `
                            <h3>依赖关系图 (DAG) - 压缩视图</h3>
                            <div class="mermaid" id="${{mermaidId}}">
${{dagGraph}}
                            </div>
                        `;

                        const depTable = generateDependencyTable(dagData);
                        html += `
                            <h3>依赖分析详情</h3>
                            <table>
                                <thead>
                                    <tr>
                                        <th>步骤序号</th>
                                        <th>动作类型</th>
                                        <th>分析说明</th>
                                        <th>依赖项</th>
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

    # Generate HTML with embedded JavaScript
    # Use string concatenation to avoid f-string escaping issues with JavaScript
    html_content = generate_html_with_js(stats, records_json)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="可视化 DAG 分析结果")
    parser.add_argument("--input", type=str,
                       default="output/deepseek/dag_analysis/analyzed_records.jsonl",
                       help="输入的分析结果文件")
    parser.add_argument("--output", type=str,
                       default="output/deepseek/dag_analysis/visualization.html",
                       help="输出的 HTML 文件")
    parser.add_argument("--limit", type=int, default=None,
                       help="限制显示的记录数量")
    parser.add_argument("--no-compress", action="store_true",
                       help="禁用 DAG 压缩，显示原始完整图")
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

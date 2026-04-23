"""Visualize GED similarity examples with percentile-based filtering."""

import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import html
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_analysis.dag_compressor import build_digraph_with_tags, compress_dag_combined
from utils import extract_numeric_id
from utils.web_report import (
    build_common_css,
    build_navigation_html,
    build_escape_html_js,
    build_navigation_js,
    build_show_tab_js,
    build_runtime_bootstrap_js,
    build_dag_helpers_js,
)


def load_ged_problem_ids_by_percentile(
    csv_path: str,
    low_percentile: float = 10.0,
    high_percentile: float = 10.0,
    comparisons: Optional[List[str]] = None
) -> Dict[str, Dict[str, List[str]]]:
    """Load problem IDs by GED percentile for multiple comparisons.

    Args:
        csv_path: Path to similarity_results.csv
        low_percentile: Percentage of lowest GED to include (e.g., 10 = bottom 10%)
        high_percentile: Percentage of highest GED to include (e.g., 10 = top 10%)
        comparisons: Comparison list to filter on

    Returns:
        Dict mapping comparison to {'low': [...], 'high': [...]} problem_ids
    """
    if comparisons is None:
        comparisons = ["original_vs_simple", "original_vs_hard"]

    # Collect all GED values per comparison
    ged_values_by_comparison = {comparison: [] for comparison in comparisons}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            problem_id = row.get('problem_id', '')
            if not problem_id:
                continue

            for comparison in comparisons:
                ged_field = f"{comparison}_ged"
                ged_str = row.get(ged_field, '')
                if ged_str and ged_str != '':
                    try:
                        ged = float(ged_str)
                        ged_values_by_comparison[comparison].append((problem_id, ged))
                    except ValueError:
                        continue

    # Sort and select by percentile
    result = {}
    for comparison in comparisons:
        ged_values = ged_values_by_comparison[comparison]
        if not ged_values:
            result[comparison] = {'low': [], 'high': []}
            continue

        ged_values.sort(key=lambda x: x[1])
        total = len(ged_values)

        low_count = max(1, int(total * low_percentile / 100))
        high_count = max(1, int(total * high_percentile / 100))

        low_ids = [pid for pid, _ in ged_values[:low_count]]
        high_ids = [pid for pid, _ in ged_values[-high_count:]]

        result[comparison] = {'low': low_ids, 'high': high_ids}

    return result


def build_output_path(base_output: str, comparison: str, category: str) -> str:
    """Build output path with comparison and category suffix."""
    output_path = Path(base_output)
    comparison_suffix = comparison.replace("original_vs_", "")
    file_suffix = output_path.suffix or ".html"
    filename = f"{output_path.stem}_{comparison_suffix}_{category}{file_suffix}"
    return str(output_path.with_name(filename))


def load_segmented_records(jsonl_path: str, problem_ids: List[str]) -> Dict[str, Dict]:
    """Load segmented records for specific problem IDs."""
    records = {}
    problem_id_set = set(problem_ids)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            pid = str(record.get('problem_id', ''))
            if pid in problem_id_set:
                records[pid] = record

    return records


def extract_dag_from_batch_response(response_data: Dict) -> Optional[List[Dict]]:
    """Extract DAG analysis from batch inference response format."""
    try:
        content = response_data.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")

        if not content:
            return None

        import re
        content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'\s*```$', '', content, flags=re.MULTILINE)
        content = content.strip()

        if not content:
            return None

        dag_analysis = json.loads(content)
        if isinstance(dag_analysis, list):
            return dag_analysis

        return None
    except Exception:
        return None


def load_dag_analysis(jsonl_path: str, problem_ids: List[str]) -> Dict[str, Dict]:
    """Load DAG analysis from batch format analyzed_records.jsonl."""
    dag_data = {}
    problem_id_set = set(problem_ids)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)

            custom_id = record.get('custom_id', '')
            parts = custom_id.rsplit('_', 1)
            if len(parts) != 2:
                continue

            problem_id, variant = parts
            if problem_id not in problem_id_set:
                continue

            dag_analysis = extract_dag_from_batch_response(record)
            if dag_analysis:
                if problem_id not in dag_data:
                    dag_data[problem_id] = {}
                dag_data[problem_id][variant] = dag_analysis

    return dag_data


def compress_dag_analysis(dag_analysis: List[Dict]) -> List[Dict]:
    """Compress DAG and return simplified structure for visualization."""
    if not dag_analysis:
        return dag_analysis

    try:
        G = build_digraph_with_tags(dag_analysis, exclude_external=False)
        G_compressed, stats = compress_dag_combined(G, merge_metadata=True)
    except Exception as e:
        print(f"Warning: Compression failed: {e}")
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


def generate_html_report(records: List[Dict], output_path: str, comparison: str, category: str):
    """Generate HTML report for GED similarity examples."""

    compared_variant = comparison.replace("original_vs_", "")
    if compared_variant not in ["simple", "hard"]:
        compared_variant = "simple"
    compared_variant_label = compared_variant.capitalize()
    variant_order_json = json.dumps(["original", compared_variant], ensure_ascii=False)

    category_label = "最低" if category == "low" else "最高"

    css_styles = build_common_css()
    navigation_html = build_navigation_html(len(records), total_count_id="totalCount")
    shared_escape_html_js = build_escape_html_js()
    shared_dag_js = build_dag_helpers_js(graph_fn_name="generateDagMermaid")
    shared_navigation_js = build_navigation_js()
    shared_tab_js = build_show_tab_js(include_mermaid=True, include_mathjax=True)
    shared_bootstrap_js = build_runtime_bootstrap_js(include_mermaid_init=True)

    records_json = json.dumps(records, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GED {category_label}相似度示例可视化</title>
    <script>
    MathJax = {{
        tex: {{
            inlineMath: [['$','$'], ['\\\\(','\\\\)']],
            displayMath: [['$$','$$'], ['\\\\[','\\\\]']]
        }}
    }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>{css_styles}</style>
</head>
<body>
    <div class="container">
        <h1>🔍 图编辑距离 (GED) {category_label}相似度示例分析</h1>
        <p style="color: #666; margin-bottom: 30px;">展示 original vs {compared_variant} 变体间 GED {category_label}的问题及其完整分析</p>

        <div class="metadata">
            <div class="metadata-item">
                <span class="metadata-label">总示例数:</span>
                <span id="totalRecords">{len(records)}</span>
            </div>
        </div>

        <h2>🔍 浏览记录</h2>
        {navigation_html}

        <div id="recordContainer"></div>
    </div>

    <script>
        const allRecords = {records_json};
        const variantOrder = {variant_order_json};
        let currentIndex = 0;
        let mermaidCounter = 0;

        {shared_escape_html_js}
        {shared_dag_js}"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        f.write("""

        async function renderRecord(index) {
            const record = allRecords[index];
            const problemId = record.problem_id;
            const problemType = escapeHtml(record.type || 'Unknown');
            const level = escapeHtml(record.level || 'Unknown');

            let html = `
                <div class="record">
                    <div class="record-header">
                        <div>
                            <span class="record-title">Problem ${problemId}</span>
                            <span class="badge variant">${problemType}</span>
                            <span class="badge variant">${level}</span>
                        </div>
                    </div>

                    <div class="tabs">
                        <button class="tab active" onclick="showTab(event, 'original')">Original</button>
                        <button class="tab" onclick="showTab(event, '""")
        f.write(compared_variant)
        f.write("""')">""")
        f.write(compared_variant_label)
        f.write("""</button>
                    </div>
            `;

            variantOrder.forEach((variantName, idx) => {
                const variant = record[variantName];
                const isActive = idx === 0 ? 'active' : '';

                html += `<div id="${variantName}" class="tab-content ${isActive}">`;

                if (!variant) {
                    html += '<p style="color: #e74c3c; padding: 20px;">⚠️ 此变体数据缺失</p>';
                } else {
                    if ('correct' in variant) {
                        const correctClass = variant.correct ? 'correct' : 'incorrect';
                        const correctText = variant.correct ? '✓ 正确' : '✗ 错误';
                        html += `<span class="badge ${correctClass}">${correctText}</span>`;
                    }

                    const numSteps = variant.num_steps || (variant.steps ? variant.steps.length : 0);
                    html += `
                        <div class="metadata">
                            <div class="metadata-item">
                                <span class="metadata-label">步骤数:</span>
                                <span>${numSteps}</span>
                            </div>
                        </div>
                    `;

                    const problemText = variant.problem || '';
                    const displayText = problemText.length > 500 ? problemText.substring(0, 500) + '...' : problemText;
                    html += `
                        <h3>问题描述</h3>
                        <div class="problem-box">
                            <div class="problem-text">${escapeHtml(displayText)}</div>
                        </div>
                    `;

                    const responseText = variant.response || '';
                    const displayResponse = responseText.length > 2000 ? responseText.substring(0, 2000) + '...' : responseText;
                    html += `
                        <h3>模型回答</h3>
                        <div class="response-box">
                            <div class="response-text">${escapeHtml(displayResponse)}</div>
                        </div>
                    `;

                    const dagCompressed = variant.dag_analysis_compressed;
                    if (dagCompressed && dagCompressed.length > 0) {
                        const dagMermaid = generateDagMermaid(dagCompressed);
                        const mermaidId = `mermaid-${index}-${variantName}-${mermaidCounter++}`;
                        html += `
                            <h3>依赖关系图 (DAG) - 压缩视图</h3>
                            <div class="mermaid" id="${mermaidId}">
${dagMermaid}
                            </div>
                        `;

                        const depTable = generateDependencyTable(dagCompressed);
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
${depTable}
                                </tbody>
                            </table>
                        `;
                    } else {
                        html += '<p style="color: #e67e22; padding: 20px;">⚠️ 此变体没有 DAG 分析数据</p>';
                    }
                }

                html += '</div>';
            });

            html += '</div>';
            document.getElementById('recordContainer').innerHTML = html;
            updateNavigation();

            const activeMermaidElements = document.querySelectorAll('.tab-content.active .mermaid');
            for (const element of activeMermaidElements) {
                if (!element.hasAttribute('data-processed')) {
                    try {
                        await mermaid.run({ nodes: [element] });
                    } catch (err) {
                        console.error('Mermaid error:', err);
                        element.innerHTML = '<p style="color: red;">Failed to render diagram</p>';
                    }
                }
            }

            if (window.MathJax) {
                MathJax.typesetPromise([document.getElementById('recordContainer')]).catch((err) => {
                    console.error('MathJax rendering error:', err);
                });
            }
        }

        """)
        f.write(shared_navigation_js)
        f.write("\n        ")
        f.write(shared_tab_js)
        f.write("\n        ")
        f.write(shared_bootstrap_js)
        f.write("""
    </script>
</body>
</html>
""")

    print(f"HTML report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="可视化 GED 相似度示例（百分比过滤）")
    parser.add_argument("--csv", type=str,
                       default="output/qwen/dag_analysis/similarity_results.csv",
                       help="相似度结果 CSV 文件")
    parser.add_argument("--segmented", type=str,
                       default="output/qwen/segmented_records.jsonl",
                       help="切分后的记录文件")
    parser.add_argument("--analyzed", type=str,
                       default="output/qwen/dag_analysis/analyzed_records.jsonl",
                       help="DAG 分析结果文件")
    parser.add_argument("--output", type=str,
                       default="output/qwen/dag_analysis/ged_similarity_visualization.html",
                       help="输出的 HTML 文件前缀")
    parser.add_argument("--low-percentile", type=float, default=10.0,
                       help="最低 GED 百分比（例如 10 表示最低的 10%%）")
    parser.add_argument("--high-percentile", type=float, default=10.0,
                       help="最高 GED 百分比（例如 10 表示最高的 10%%）")
    args = parser.parse_args()

    for path_arg, path_val in [("csv", args.csv), ("segmented", args.segmented), ("analyzed", args.analyzed)]:
        if not Path(path_val).exists():
            print(f"Error: {path_arg} file not found: {path_val}")
            return

    comparisons = ["original_vs_simple", "original_vs_hard"]

    print(f"Loading GED data by percentile (low={args.low_percentile}%, high={args.high_percentile}%)...")
    problem_ids_by_comparison = load_ged_problem_ids_by_percentile(
        args.csv, args.low_percentile, args.high_percentile, comparisons
    )

    for comparison in comparisons:
        low_count = len(problem_ids_by_comparison[comparison]['low'])
        high_count = len(problem_ids_by_comparison[comparison]['high'])
        print(f"{comparison}: {low_count} low GED, {high_count} high GED")

    all_problem_ids = sorted(
        {pid for comp_data in problem_ids_by_comparison.values()
         for ids in comp_data.values() for pid in ids},
        key=extract_numeric_id
    )
    if not all_problem_ids:
        print("No problems found matching criteria.")
        return

    print(f"Loading segmented records...")
    segmented_records = load_segmented_records(args.segmented, all_problem_ids)
    print(f"Loaded {len(segmented_records)} segmented records")

    print(f"Loading DAG analysis...")
    dag_data = load_dag_analysis(args.analyzed, all_problem_ids)
    print(f"Loaded DAG analysis for {len(dag_data)} problems")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    generated_outputs = []

    for comparison in comparisons:
        for category in ['low', 'high']:
            problem_ids = sorted(
                problem_ids_by_comparison[comparison][category],
                key=extract_numeric_id
            )
            if not problem_ids:
                print(f"Skipping {comparison} {category}: no matching problems")
                continue

            print(f"Merging data for {comparison} {category}...")
            merged_records = []
            for pid in problem_ids:
                if pid not in segmented_records:
                    print(f"Warning: Problem {pid} not found in segmented records")
                    continue

                record = segmented_records[pid]
                if pid in dag_data:
                    for variant in ['original', 'simple', 'hard']:
                        if variant in record and variant in dag_data[pid]:
                            dag_analysis = dag_data[pid][variant]
                            record[variant]['dag_analysis_compressed'] = compress_dag_analysis(dag_analysis)
                merged_records.append(record)

            print(f"Merged {len(merged_records)} complete records for {comparison} {category}")

            output_path = build_output_path(args.output, comparison, category)
            print(f"Generating HTML report: {output_path}")
            generate_html_report(merged_records, output_path, comparison, category)
            generated_outputs.append(output_path)

    if generated_outputs:
        print("\nDone! Open in browser:")
        for output_path in generated_outputs:
            print(f"   {Path(output_path).absolute()}")
    else:
        print("No reports generated.")


if __name__ == "__main__":
    main()

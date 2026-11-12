"""Browse GED examples selected by percentiles or an absolute threshold."""

import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.io import atomic_text
from utils.visualization.dag_records import compress_dag_analysis, load_segmented_records, load_dag_analysis
from utils.web_report import (
    build_common_css,
    build_navigation_html,
    build_escape_html_js,
    build_navigation_js,
    build_show_tab_js,
    build_runtime_bootstrap_js,
    build_dag_helpers_js,
)


COMPARISONS = ("original_vs_simple", "original_vs_hard")


def select_ged_problem_ids(csv_path, low_percentile=10.0, high_percentile=10.0, max_ged=None):
    """Select each comparison's percentile tails or an inclusive GED threshold."""
    values = {comparison: [] for comparison in COMPARISONS}
    with open(csv_path, encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            for comparison in COMPARISONS:
                score = row[f'{comparison}_ged']
                if score:
                    values[comparison].append((row['problem_id'], float(score)))

    selected = {}
    for comparison, entries in values.items():
        if max_ged is not None:
            selected[comparison] = {'threshold': [pid for pid, score in entries if score <= max_ged]}
        else:
            entries.sort(key=lambda item: item[1])
            low_count = max(1, int(len(entries) * low_percentile / 100)) if low_percentile else 0
            high_count = max(1, int(len(entries) * high_percentile / 100)) if high_percentile else 0
            selected[comparison] = {
                'low': [pid for pid, _ in entries[:low_count]],
                'high': [pid for pid, _ in entries[-high_count:]] if high_count else [],
            }
    return selected


def build_output_path(base_output: str, comparison: str, category: str) -> str:
    """Build output path with comparison and category suffix."""
    output_path = Path(base_output)
    comparison_suffix = comparison.replace("original_vs_", "")
    file_suffix = output_path.suffix or ".html"
    filename = f"{output_path.stem}_{comparison_suffix}_{category}{file_suffix}"
    return str(output_path.with_name(filename))


def generate_html_report(records: List[Dict], output_path: str, comparison: str, category: str):
    """Generate HTML report for GED similarity examples."""

    compared_variant = comparison.replace("original_vs_", "")
    compared_variant_label = compared_variant.capitalize()
    variant_order_json = json.dumps(["original", compared_variant], ensure_ascii=False)

    category_label = {"low": "lowest", "high": "highest", "threshold": "selected"}[category]

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
    <title>GED {category_label} Similarity Example Viewer</title>
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
        <h1>🔍 Graph Edit Distance (GED) {category_label} Similarity Examples</h1>
        <p style="color: #666; margin-bottom: 30px;">Problems with the {category_label} GED between original and {compared_variant} variants, with complete analysis</p>

        <div class="metadata">
            <div class="metadata-item">
                <span class="metadata-label">Total Examples:</span>
                <span id="totalRecords">{len(records)}</span>
            </div>
        </div>

        <h2>🔍 Browse Records</h2>
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

    with atomic_text(output_path) as f:
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
                    html += '<p style="color: #e74c3c; padding: 20px;">⚠️ Missing data for this variant</p>';
                } else {
                    if ('correct' in variant) {
                        const correctClass = variant.correct ? 'correct' : 'incorrect';
                        const correctText = variant.correct ? '✓ Correct' : '✗ Incorrect';
                        html += `<span class="badge ${correctClass}">${correctText}</span>`;
                    }

                    const numSteps = variant.num_steps || (variant.steps ? variant.steps.length : 0);
                    html += `
                        <div class="metadata">
                            <div class="metadata-item">
                                <span class="metadata-label">Steps:</span>
                                <span>${numSteps}</span>
                            </div>
                        </div>
                    `;

                    const problemText = variant.problem || '';
                    const displayText = problemText.length > 500 ? problemText.substring(0, 500) + '...' : problemText;
                    html += `
                        <h3>Problem</h3>
                        <div class="problem-box">
                            <div class="problem-text">${escapeHtml(displayText)}</div>
                        </div>
                    `;

                    const responseText = variant.response || '';
                    const displayResponse = responseText.length > 2000 ? responseText.substring(0, 2000) + '...' : responseText;
                    html += `
                        <h3>Model Response</h3>
                        <div class="response-box">
                            <div class="response-text">${escapeHtml(displayResponse)}</div>
                        </div>
                    `;

                    const dagCompressed = variant.dag_analysis_compressed;
                    if (dagCompressed && dagCompressed.length > 0) {
                        const dagMermaid = generateDagMermaid(dagCompressed);
                        const mermaidId = `mermaid-${index}-${variantName}-${mermaidCounter++}`;
                        html += `
                            <h3>Dependency Graph (DAG) - Compressed View</h3>
                            <div class="mermaid" id="${mermaidId}">
${dagMermaid}
                            </div>
                        `;

                        const depTable = generateDependencyTable(dagCompressed);
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
${depTable}
                                </tbody>
                            </table>
                        `;
                    } else {
                        html += '<p style="color: #e67e22; padding: 20px;">⚠️ No DAG analysis data for this variant</p>';
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, help='Wide GED table with original_vs_simple_ged / original_vs_hard_ged')
    parser.add_argument('--segmented', required=True)
    parser.add_argument('--analyzed', required=True)
    parser.add_argument('--output', required=True, help='HTML output prefix')
    parser.add_argument('--max-ged', type=float, help='Inclusive threshold; replaces percentile selection')
    parser.add_argument('--low-percentile', type=float, default=10.0)
    parser.add_argument('--high-percentile', type=float, default=10.0)
    args = parser.parse_args()
    if not all(0 <= value <= 100 for value in (args.low_percentile, args.high_percentile)):
        parser.error('Percentiles must be between 0 and 100')

    selection = select_ged_problem_ids(args.csv, args.low_percentile, args.high_percentile, args.max_ged)
    problem_ids = {pid for groups in selection.values() for ids in groups.values() for pid in ids}
    if not problem_ids:
        print('No problems matched the selection.')
        return
    records = load_segmented_records(args.segmented, problem_ids)
    dags = load_dag_analysis(args.analyzed, problem_ids)
    for comparison, groups in selection.items():
        variant = comparison.replace('original_vs_', '')
        for category, ids in groups.items():
            if not ids:
                continue
            selected_records = []
            for pid in sorted(ids, key=int):
                record = records[pid]
                for current in ('original', variant):
                    record[current]['dag_analysis_compressed'] = compress_dag_analysis(dags[pid][current])
                selected_records.append(record)
            path = build_output_path(args.output, comparison, category)
            print(f'{comparison} / {category}: {len(selected_records)} problems')
            generate_html_report(selected_records, path, comparison, category)


if __name__ == "__main__":
    main()

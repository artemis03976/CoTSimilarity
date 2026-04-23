"""HTML rendering script for model generation results."""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import sort_records, load_records
from utils.web_report import (
    build_common_css,
    build_navigation_html,
    build_escape_html_js,
    build_navigation_js,
    build_show_tab_js,
    build_runtime_bootstrap_js,
)


def generate_statistics(records: List[Dict]) -> Dict:
    """Generate statistics from records."""
    stats = {
        "total_records": len(records),
        "total_variants": 0,
        "correct_count": 0,
        "incorrect_count": 0,
        "variant_stats": {
            "original": {"correct": 0, "incorrect": 0},
            "simple": {"correct": 0, "incorrect": 0},
            "hard": {"correct": 0, "incorrect": 0}
        }
    }

    for record in records:
        for variant in ["original", "simple", "hard"]:
            if variant in record:
                stats["total_variants"] += 1
                entry = record[variant]
                sample = entry.get("samples", [{}])[0] if entry.get("samples") else entry
                if sample.get("correct", False):
                    stats["correct_count"] += 1
                    stats["variant_stats"][variant]["correct"] += 1
                else:
                    stats["incorrect_count"] += 1
                    stats["variant_stats"][variant]["incorrect"] += 1

    return stats


def generate_html_report(records: List[Dict], output_path: str):
    """Generate comprehensive HTML report."""

    stats = generate_statistics(records)

    # Serialize records to JSON for client-side rendering
    records_json = json.dumps(records, ensure_ascii=False)
    css_styles = build_common_css()
    navigation_html = build_navigation_html(stats["total_records"], include_random=True)
    shared_escape_html_js = build_escape_html_js()
    shared_navigation_js = build_navigation_js(include_random=True)
    shared_tab_js = build_show_tab_js(include_mathjax=True)
    shared_bootstrap_js = build_runtime_bootstrap_js()

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型生成结果查看器</title>
    <script>
    MathJax = {{
        tex: {{
            inlineMath: [['$','$'], ['\\\\(','\\\\)']],
            displayMath: [['$$','$$'], ['\\\\[','\\\\]']]
        }}
    }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>{css_styles}</style>
</head>
<body>
    <div class="container">
        <h1>📊 模型生成结果查看器</h1>
        <p style="color: #666; margin-bottom: 30px;">Model Generation Results Viewer</p>

        <h2>📈 总体统计</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">总记录数</div>
                <div class="stat-value">{stats['total_records']}</div>
            </div>
            <div class="stat-card blue">
                <div class="stat-label">总变体数</div>
                <div class="stat-value">{stats['total_variants']}</div>
            </div>
            <div class="stat-card green">
                <div class="stat-label">正确数量</div>
                <div class="stat-value">{stats['correct_count']}</div>
            </div>
            <div class="stat-card red">
                <div class="stat-label">错误数量</div>
                <div class="stat-value">{stats['incorrect_count']}</div>
            </div>
        </div>

        <h2>📋 各变体统计</h2>
        <div class="variant-summary">
            <div class="summary-item">
                <span class="summary-label">Original:</span>
                <span class="badge correct">✓ {stats['variant_stats']['original']['correct']}</span>
                <span class="badge incorrect">✗ {stats['variant_stats']['original']['incorrect']}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Simple:</span>
                <span class="badge correct">✓ {stats['variant_stats']['simple']['correct']}</span>
                <span class="badge incorrect">✗ {stats['variant_stats']['simple']['incorrect']}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Hard:</span>
                <span class="badge correct">✓ {stats['variant_stats']['hard']['correct']}</span>
                <span class="badge incorrect">✗ {stats['variant_stats']['hard']['incorrect']}</span>
            </div>
        </div>

        <h2>🔍 浏览记录</h2>
        {navigation_html}

        <div id="recordContainer"></div>
"""

    # Add JavaScript for client-side rendering and navigation
    html_content += f"""
    </div>

    <script>
        // Store all records in JavaScript
        const allRecords = {records_json};
        let currentIndex = 0;

        {shared_escape_html_js}

        function renderRecord(index) {{
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

            // Generate content for each variant
            ['original', 'simple', 'hard'].forEach((variantName, idx) => {{
                if (record[variantName]) {{
                    const entry = record[variantName];
                    const isActive = idx === 0 ? 'active' : '';
                    const sample = (entry.samples && entry.samples.length > 0) ? entry.samples[0] : entry;
                    const correct = sample.correct || false;
                    const correctClass = correct ? 'correct' : 'incorrect';
                    const correctText = correct ? '✓ 正确' : '✗ 错误';
                    const problemText = escapeHtml(entry.problem || '');
                    const groundTruth = escapeHtml(String(entry.ground_truth || ''));
                    const response = escapeHtml(sample.response || '');

                    html += `
                        <div id="${{variantName}}" class="tab-content ${{isActive}}">
                            <span class="badge ${{correctClass}}">${{correctText}}</span>
                            <h3>问题描述</h3>
                            <div class="content-box">
                                <div class="content-text">${{problemText}}</div>
                            </div>
                            <h3>标准答案</h3>
                            <div class="content-box">
                                <div class="content-text">${{groundTruth}}</div>
                            </div>
                            <h3>模型回答</h3>
                            <div class="content-box">
                                <div class="content-text">${{response}}</div>
                            </div>
                        </div>
                    `;
                }}
            }});

            html += '</div>';
            document.getElementById('recordContainer').innerHTML = html;

            // Update navigation state
            updateNavigation();

            // Re-render MathJax
            if (window.MathJax) {{
                MathJax.typesetPromise([document.getElementById('recordContainer')]).catch((err) => {{
                    console.error('MathJax rendering error:', err);
                }});
            }}
        }}

        {shared_navigation_js}
        {shared_tab_js}
        {shared_bootstrap_js}
    </script>
</body>
</html>
"""

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="生成模型结果的 HTML 可视化报告")
    parser.add_argument("--input", type=str,
                       default="output/qwen/dcpr/all_records.jsonl",
                       help="输入的 JSONL 文件")
    parser.add_argument("--output", type=str, default=None,
                       help="输出的 HTML 文件")
    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return

    # Load records
    print(f"Loading data from: {args.input}")
    records = load_records(args.input)
    print(f"Loaded {len(records)} records")

    # Sort records by problem_id
    records = sort_records(records)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Generate output path based on input path
        input_path = Path(args.input)
        output_path = str(input_path.with_suffix('.html'))

    # Generate HTML report
    print(f"Generating HTML report...")
    generate_html_report(records, output_path)

    print(f"\nDone! Open in browser:")
    print(f"   {Path(output_path).absolute()}")


if __name__ == "__main__":
    main()

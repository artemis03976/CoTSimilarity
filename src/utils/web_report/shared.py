from textwrap import dedent


def build_common_css() -> str:
    return dedent(
        """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        h1 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2em;
        }

        h2 {
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }

        h3 {
            color: #555;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }

        .stat-card.green {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }

        .stat-card.red {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        }

        .stat-card.orange {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }

        .stat-card.blue {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }

        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }

        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }

        .navigation {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            gap: 15px;
            flex-wrap: wrap;
        }

        .nav-buttons {
            display: flex;
            gap: 10px;
        }

        .nav-btn {
            padding: 10px 20px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            transition: background 0.3s;
        }

        .nav-btn:hover {
            background: #2980b9;
        }

        .nav-btn:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
        }

        .nav-info {
            font-size: 1.1em;
            font-weight: 600;
            color: #2c3e50;
        }

        .jump-controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .jump-input {
            width: 80px;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 1em;
        }

        .record {
            margin: 30px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }

        .record-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .record-title {
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
        }

        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }

        .badge.correct {
            background: #d4edda;
            color: #155724;
        }

        .badge.incorrect {
            background: #f8d7da;
            color: #721c24;
        }

        .badge.variant {
            background: #d1ecf1;
            color: #0c5460;
        }

        .tabs {
            display: flex;
            gap: 10px;
            margin: 20px 0;
            border-bottom: 2px solid #ddd;
        }

        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 1em;
            color: #666;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }

        .tab:hover {
            color: #3498db;
        }

        .tab.active {
            color: #3498db;
            border-bottom-color: #3498db;
            font-weight: 600;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .content-box {
            background: white;
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
            border: 1px solid #ddd;
            max-height: 500px;
            overflow-y: auto;
        }

        .field-title {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 8px;
            font-size: 1.1em;
        }

        .content-text {
            white-space: pre-wrap;
            color: #555;
            font-size: 0.95em;
            line-height: 1.8;
            font-family: "Courier New", monospace;
        }

        .variant-summary {
            display: flex;
            gap: 15px;
            margin: 15px 0;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 6px;
            font-size: 0.9em;
        }

        .summary-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .summary-label {
            font-weight: 600;
            color: #666;
        }

        .problem-box {
            background: white;
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
            border: 1px solid #ddd;
        }

        .problem-text {
            font-family: "Courier New", monospace;
            white-space: pre-wrap;
            color: #555;
            font-size: 0.9em;
        }

        .response-box {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
            border-left: 3px solid #3498db;
        }

        .response-text {
            white-space: pre-wrap;
            color: #333;
            font-size: 0.95em;
        }

        .mermaid {
            background: white;
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
            border: 1px solid #ddd;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }

        th {
            background: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }

        td {
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }

        tr:hover {
            background: #f8f9fa;
        }

        .step-id {
            font-weight: bold;
            color: #3498db;
            width: 100px;
        }

        .analysis {
            color: #555;
        }

        .dep-problem {
            color: #2980b9;
            font-weight: 600;
        }

        .dep-external {
            color: #e67e22;
            font-weight: 600;
        }

        .dep-steps {
            color: #27ae60;
            font-weight: 600;
        }

        .tag-define {
            background: #e3f2fd;
            color: #1565c0;
            font-weight: 600;
            padding: 4px 8px;
            border-radius: 4px;
        }

        .tag-recall {
            background: #fff3e0;
            color: #e65100;
            font-weight: 600;
            padding: 4px 8px;
            border-radius: 4px;
        }

        .tag-derive {
            background: #f3e5f5;
            color: #6a1b9a;
            font-weight: 600;
            padding: 4px 8px;
            border-radius: 4px;
        }

        .tag-calculate {
            background: #e8f5e9;
            color: #2e7d32;
            font-weight: 600;
            padding: 4px 8px;
            border-radius: 4px;
        }

        .tag-verify {
            background: #fff9c4;
            color: #f57f17;
            font-weight: 600;
            padding: 4px 8px;
            border-radius: 4px;
        }

        .tag-conclude {
            background: #fce4ec;
            color: #c2185b;
            font-weight: 600;
            padding: 4px 8px;
            border-radius: 4px;
        }

        .metadata {
            display: flex;
            gap: 20px;
            margin: 15px 0;
            font-size: 0.9em;
            color: #666;
        }

        .metadata-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .metadata-label {
            font-weight: 600;
        }

        .legend {
            display: flex;
            gap: 20px;
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
            flex-wrap: wrap;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }

        .legend-color.problem {
            background: #2980b9;
        }

        .legend-color.external {
            background: #e67e22;
        }

        .legend-color.steps {
            background: #27ae60;
        }

        #recordContainer {
            min-height: 400px;
        }
        """
    )


def build_navigation_html(total_records: int, include_random: bool = False, total_count_id: str = "totalRecords") -> str:
    random_button = '<button class="nav-btn" onclick="randomRecord()">🎲 随机</button>' if include_random else ""
    return dedent(
        f"""
        <div class="navigation">
            <div class="nav-buttons">
                <button class="nav-btn" id="firstBtn" onclick="goToFirst()">⏮ 首页</button>
                <button class="nav-btn" id="prevBtn" onclick="goToPrev()">◀ 上一个</button>
            </div>
            <div class="nav-info">
                <span id="currentIndex">1</span> / <span id="{total_count_id}">{total_records}</span>
            </div>
            <div class="nav-buttons">
                <button class="nav-btn" id="nextBtn" onclick="goToNext()">下一个 ▶</button>
                <button class="nav-btn" id="lastBtn" onclick="goToLast()">末页 ⏭</button>
            </div>
            <div class="jump-controls">
                <input type="number" class="jump-input" id="jumpInput" min="1" max="{total_records}" placeholder="跳转">
                <button class="nav-btn" onclick="jumpTo()">跳转</button>
                {random_button}
            </div>
        </div>
        """
    )


def build_escape_html_js() -> str:
    return dedent(
        """
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        """
    )


def build_navigation_js(include_random: bool = False) -> str:
    random_js = dedent(
        """
        function randomRecord() {
            currentIndex = Math.floor(Math.random() * allRecords.length);
            renderRecord(currentIndex);
        }
        """
    ) if include_random else ""
    return dedent(
        f"""
        function updateNavigation() {{
            document.getElementById('currentIndex').textContent = currentIndex + 1;
            document.getElementById('firstBtn').disabled = currentIndex === 0;
            document.getElementById('prevBtn').disabled = currentIndex === 0;
            document.getElementById('nextBtn').disabled = currentIndex === allRecords.length - 1;
            document.getElementById('lastBtn').disabled = currentIndex === allRecords.length - 1;
        }}

        function goToFirst() {{ currentIndex = 0; renderRecord(currentIndex); }}
        function goToPrev() {{ if (currentIndex > 0) {{ currentIndex--; renderRecord(currentIndex); }} }}
        function goToNext() {{ if (currentIndex < allRecords.length - 1) {{ currentIndex++; renderRecord(currentIndex); }} }}
        function goToLast() {{ currentIndex = allRecords.length - 1; renderRecord(currentIndex); }}

        function jumpTo() {{
            const input = document.getElementById('jumpInput');
            const targetIndex = parseInt(input.value) - 1;
            if (targetIndex >= 0 && targetIndex < allRecords.length) {{
                currentIndex = targetIndex;
                renderRecord(currentIndex);
                input.value = '';
            }} else {{
                alert('请输入有效的记录编号 (1-' + allRecords.length + ')');
            }}
        }}

        {random_js}
        """
    )


def build_show_tab_js(include_mermaid: bool = False, include_mathjax: bool = False) -> str:
    mermaid_js = dedent(
        """
            const mermaidElements = selectedContent.querySelectorAll('.mermaid');
            for (const element of mermaidElements) {
                if (!element.hasAttribute('data-processed')) {
                    try {
                        await mermaid.run({ nodes: [element] });
                    } catch (err) {
                        console.error('Mermaid error:', err);
                        element.innerHTML = '<p style="color: red;">Failed to render diagram</p>';
                    }
                }
            }
        """
    ) if include_mermaid else ""
    mathjax_js = dedent(
        """
            if (window.MathJax) {
                MathJax.typesetPromise([selectedContent]).catch((err) => {
                    console.error('MathJax rendering error:', err);
                });
            }
        """
    ) if include_mathjax else ""
    return dedent(
        f"""
        async function showTab(event, tabId) {{
            const record = event.target.closest('.record');
            record.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            record.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));

            const selectedContent = document.getElementById(tabId);
            if (selectedContent) selectedContent.classList.add('active');
            event.target.classList.add('active');
        {mermaid_js}{mathjax_js}
        }}
        """
    )


def build_runtime_bootstrap_js(include_mermaid_init: bool = False) -> str:
    mermaid_init = "mermaid.initialize({ startOnLoad: false, theme: 'default', securityLevel: 'loose' });" if include_mermaid_init else ""
    return dedent(
        f"""
        {mermaid_init}
        document.addEventListener('DOMContentLoaded', () => renderRecord(currentIndex));
        document.addEventListener('keydown', e => {{
            if (e.key === 'ArrowLeft') goToPrev();
            else if (e.key === 'ArrowRight') goToNext();
        }});
        """
    )


def build_dag_helpers_js(graph_fn_name: str = "generateDagGraph") -> str:
    return dedent(
        f"""
        function {graph_fn_name}(dagAnalysis) {{
            const lines = ['graph TD'];
            lines.push('    Problem["[0] Problem"]');

            dagAnalysis.forEach(dep => {{
                const stepId = dep.step_id;
                const tag = dep.macro_action_tag || '';
                const mergedLabel = dep.merged_label || String(stepId);
                const label = tag ? `[${{mergedLabel}}] ${{tag}}` : `[${{mergedLabel}}]`;
                lines.push(`    Step${{stepId}}["${{label}}"]`);
            }});

            dagAnalysis.forEach(dep => {{
                const stepId = dep.step_id;
                dep.depends_on.forEach(dependency => {{
                    if (dependency === 0) {{
                        lines.push(`    Problem --> Step${{stepId}}`);
                    }} else if (dependency === 'External') {{
                        if (!lines.some(line => line.includes('External["'))) {{
                            lines.push('    External["[Ext] External"]');
                        }}
                        lines.push(`    External -.-> Step${{stepId}}`);
                    }} else {{
                        lines.push(`    Step${{dependency}} --> Step${{stepId}}`);
                    }}
                }});
            }});

            lines.push('    style Problem fill:#e1f5ff');
            lines.push('    style External fill:#fff4e1');

            return lines.join('\\n');
        }}

        function generateDependencyTable(dagAnalysis) {{
            const rows = [];
            dagAnalysis.forEach(dep => {{
                const mergedLabel = dep.merged_label || String(dep.step_id);
                const tag = dep.macro_action_tag || 'N/A';
                const depStr = dep.depends_on.map(d => `[${{d}}]`).join(', ');

                let depClass;
                if (JSON.stringify(dep.depends_on) === '[0]') {{
                    depClass = 'dep-problem';
                }} else if (dep.depends_on.includes('External')) {{
                    depClass = 'dep-external';
                }} else {{
                    depClass = 'dep-steps';
                }}

                const tagClass = tag !== 'N/A' ? `tag-${{tag.toLowerCase()}}` : '';

                rows.push(`
                    <tr>
                        <td class="step-id">${{mergedLabel}}</td>
                        <td class="${{tagClass}}">${{tag}}</td>
                        <td class="analysis">${{escapeHtml(dep.analysis || '')}}</td>
                        <td class="${{depClass}}">${{depStr}}</td>
                    </tr>
                `);
            }});

            return rows.join('\\n');
        }}
        """
    )

"""
FinanceBench Manual Review Report Generator
===========================================
Exports JSONL evaluation results into a clean, searchable HTML table
and Markdown report for convenient manual review.
"""

import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"


def export_review(jsonl_file: Path):
    if not jsonl_file.exists():
        print(f"[ERROR] File not found: {jsonl_file}")
        return

    records = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    print(f"[INFO] Loaded {len(records)} records from {jsonl_file.name}")

    stem = jsonl_file.stem
    html_out = EVAL_DIR / f"{stem}_review.html"
    md_out = EVAL_DIR / f"{stem}_review.md"

    # 1. Generate Markdown Report
    md_lines = [
        f"# 📋 Báo Cáo Manual Review: {jsonl_file.name}",
        f"- **Tổng số câu đã đánh giá:** {len(records)}",
        f"- **Thời gian xuất file:** {jsonl_file.stat().st_mtime}",
        "",
        "---",
        "",
        "| # | ID | Company | Ev Hit | Phán Quyết | Đáp Án Chuẩn (Gold) | Câu Trả Lời Của Model | Lý Do Judge |",
        "|---|---|---|:---:|:---:|---|---|---|"
    ]

    for i, r in enumerate(records, 1):
        q_id = r.get("financebench_id", "")
        comp = r.get("company", "")
        ev_hit = "✅ True" if r.get("hit_evidence") else "❌ False"
        label = r.get("label", "")
        lbl_badge = f"**{label}**"
        gold = str(r.get("gold_answer", "")).replace("\n", " ").replace("|", "\\|")[:80]
        model = str(r.get("model_answer", "")).replace("\n", " ").replace("|", "\\|")[:120]
        judge_rsn = str(r.get("judge_reasoning", "")).replace("\n", " ").replace("|", "\\|")[:120]

        md_lines.append(f"| {i} | `{q_id}` | {comp} | {ev_hit} | {lbl_badge} | {gold} | {model} | {judge_rsn} |")

    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[SAVED] Markdown review saved to: {md_out}")

    # 2. Generate Interactive HTML Table
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FinanceBench Manual Review - {stem}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #e2e8f0;
            padding: 24px;
            margin: 0;
        }}
        h1 {{
            color: #38bdf8;
            font-size: 24px;
            margin-bottom: 8px;
        }}
        .stats-bar {{
            display: flex;
            gap: 16px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .stat-card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 12px 18px;
        }}
        .stat-val {{
            font-size: 20px;
            font-weight: bold;
            color: #f8fafc;
        }}
        .stat-lbl {{
            font-size: 12px;
            color: #94a3b8;
        }}
        .filter-box {{
            margin-bottom: 16px;
            padding: 10px;
            background: #1e293b;
            border-radius: 6px;
            border: 1px solid #334155;
        }}
        input, select {{
            background: #0f172a;
            border: 1px solid #475569;
            color: #f8fafc;
            padding: 8px 12px;
            border-radius: 4px;
            margin-right: 12px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            background: #1e293b;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px 14px;
            border-bottom: 1px solid #334155;
            text-align: left;
            vertical-align: top;
        }}
        th {{
            background: #0f172a;
            color: #94a3b8;
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        tr:hover {{
            background: #273549;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-correct {{ background: #065f46; color: #34d399; }}
        .badge-incorrect {{ background: #7f1d1d; color: #f87171; }}
        .badge-refusal {{ background: #78350f; color: #fbbf24; }}
        .badge-hit-true {{ background: #1e3a8a; color: #60a5fa; }}
        .badge-hit-false {{ background: #374151; color: #9ca3af; }}
        .text-wrap {{
            max-width: 320px;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 180px;
            overflow-y: auto;
        }}
        .gold-box {{
            color: #a7f3d0;
            background: rgba(16, 185, 129, 0.1);
            padding: 6px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <h1>📊 FinanceBench Manual Review UI</h1>
    <div class="stats-bar">
        <div class="stat-card">
            <div class="stat-val">{len(records)}</div>
            <div class="stat-lbl">Tổng số câu</div>
        </div>
        <div class="stat-card">
            <div class="stat-val" style="color: #34d399;">{sum(1 for r in records if r.get('label') == 'Correct Answer')}</div>
            <div class="stat-lbl">Correct Answer</div>
        </div>
        <div class="stat-card">
            <div class="stat-val" style="color: #fbbf24;">{sum(1 for r in records if r.get('label') == 'Refusal')}</div>
            <div class="stat-lbl">Refusal</div>
        </div>
        <div class="stat-card">
            <div class="stat-val" style="color: #f87171;">{sum(1 for r in records if r.get('label') == 'Incorrect Answer')}</div>
            <div class="stat-lbl">Incorrect Answer</div>
        </div>
        <div class="stat-card">
            <div class="stat-val" style="color: #60a5fa;">{sum(1 for r in records if r.get('hit_evidence'))}</div>
            <div class="stat-lbl">Ev Hit True</div>
        </div>
    </div>

    <div class="filter-box">
        <input type="text" id="searchInput" onkeyup="filterTable()" placeholder="🔍 Tìm kiếm câu hỏi, công ty, id...">
        <select id="labelFilter" onchange="filterTable()">
            <option value="">Tất cả nhãn (All Labels)</option>
            <option value="Correct Answer">Correct Answer</option>
            <option value="Incorrect Answer">Incorrect Answer</option>
            <option value="Refusal">Refusal</option>
        </select>
        <select id="hitFilter" onchange="filterTable()">
            <option value="">Tất cả Ev Hit</option>
            <option value="Hit: True">Hit: True</option>
            <option value="Hit: False">Hit: False</option>
        </select>
    </div>

    <table id="reviewTable">
        <thead>
            <tr>
                <th width="40">#</th>
                <th width="120">ID / Công ty</th>
                <th width="100">Ev Hit & Trang</th>
                <th width="120">Phán quyết</th>
                <th width="240">Câu hỏi</th>
                <th width="220">Đáp án chuẩn (Gold)</th>
                <th>Model Sinh (Space Bunny)</th>
                <th>Lý do của Judge</th>
            </tr>
        </thead>
        <tbody>
    """

    for i, r in enumerate(records, 1):
        q_id = r.get("financebench_id", "")
        comp = r.get("company", "")
        lbl = r.get("label", "")
        lbl_class = "badge-correct" if lbl == "Correct Answer" else ("badge-refusal" if lbl == "Refusal" else "badge-incorrect")
        
        hit = r.get("hit_evidence", False)
        hit_badge = f'<span class="badge { "badge-hit-true" if hit else "badge-hit-false" }">Hit: {hit}</span>'
        ev_pages = r.get("evidence_pages_1indexed", [])
        ret_pages = r.get("retrieved_pages_1indexed", [])
        page_info = f"<br><small style='color: #94a3b8;'>Ev p.{ev_pages}<br>Ret p.{ret_pages}</small>"

        q_txt = r.get("question", "")
        gold = str(r.get("gold_answer", ""))
        model_ans = str(r.get("model_answer", ""))
        judge_rsn = str(r.get("judge_reasoning", ""))

        html_content += f"""
            <tr>
                <td>{i}</td>
                <td><strong>{comp}</strong><br><small style="color: #64748b;">{q_id}</small></td>
                <td>{hit_badge}{page_info}</td>
                <td><span class="badge {lbl_class}">{lbl}</span></td>
                <td><div class="text-wrap">{q_txt}</div></td>
                <td><div class="text-wrap gold-box">{gold}</div></td>
                <td><div class="text-wrap">{model_ans}</div></td>
                <td><div class="text-wrap" style="color: #cbd5e1;">{judge_rsn}</div></td>
            </tr>
        """

    html_content += """
        </tbody>
    </table>

    <script>
        function filterTable() {
            var input = document.getElementById("searchInput").value.toUpperCase();
            var labelFilter = document.getElementById("labelFilter").value;
            var hitFilter = document.getElementById("hitFilter").value;
            var table = document.getElementById("reviewTable");
            var tr = table.getElementsByTagName("tr");

            for (var i = 1; i < tr.length; i++) {
                var rowText = tr[i].textContent || tr[i].innerText;
                var show = true;

                if (input && rowText.toUpperCase().indexOf(input) === -1) {
                    show = false;
                }
                if (labelFilter && rowText.indexOf(labelFilter) === -1) {
                    show = false;
                }
                if (hitFilter && rowText.indexOf(hitFilter) === -1) {
                    show = false;
                }

                tr[i].style.display = show ? "" : "none";
            }
        }
    </script>
</body>
</html>
    """

    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[SAVED] Interactive HTML table saved to: {html_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Manual Review Report")
    parser.add_argument("--file", type=str, default="generation_results_spacebunny_pymupdf_singleStore.jsonl", help="JSONL filename in outputs/evaluation")
    args = parser.parse_args()

    jsonl_path = EVAL_DIR / args.file
    export_review(jsonl_path)

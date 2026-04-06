from __future__ import annotations

from datetime import datetime
from textwrap import wrap
from typing import Iterable


def _latin1(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")


def _pdf_escape(text: str) -> str:
    return _latin1(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _stringify(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _line_chunks(lines: Iterable[str], width: int = 92) -> list[str]:
    chunks: list[str] = []
    for line in lines:
        text = _latin1(str(line)).strip()
        if not text:
            chunks.append("")
            continue
        wrapped = wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
        chunks.extend(wrapped or [""])
    return chunks


def _content_stream(title: str, lines: list[str], page_number: int, total_pages: int) -> bytes:
    title_escaped = _pdf_escape(title)
    body_lines = []
    for line in lines:
        body_lines.append(f"({_pdf_escape(line)}) Tj")
        body_lines.append("T*")
    body_payload = "\n".join(body_lines[:-1]) if body_lines else ""
    footer = _pdf_escape(f"Page {page_number} of {total_pages}")
    stream = (
        "BT\n"
        "/F1 16 Tf\n"
        "72 760 Td\n"
        f"({title_escaped}) Tj\n"
        "ET\n"
        "BT\n"
        "/F1 11 Tf\n"
        "72 736 Td\n"
        "14 TL\n"
        f"{body_payload}\n"
        "ET\n"
        "BT\n"
        "/F1 9 Tf\n"
        "72 36 Td\n"
        f"({footer}) Tj\n"
        "ET"
    )
    return stream.encode("latin-1", "replace")


def _render_pdf(title: str, pages: list[list[str]]) -> bytes:
    page_streams = [
        _content_stream(title, page_lines, index + 1, len(pages))
        for index, page_lines in enumerate(pages)
    ]
    page_ids = [5 + index * 2 for index in range(len(page_streams))]
    max_id = 3 + (len(page_streams) * 2)
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] /Count {len(page_ids)} >>".encode(
            "latin-1"
        ),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }

    for index, page_stream in enumerate(page_streams):
        content_id = 4 + index * 2
        page_id = 5 + index * 2
        objects[content_id] = (
            f"<< /Length {len(page_stream)} >>\nstream\n".encode("latin-1")
            + page_stream
            + b"\nendstream"
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode(
                "latin-1"
            )
        )

    pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = [0] * (max_id + 1)
    for object_id in range(1, max_id + 1):
        offsets[object_id] = len(pdf)
        pdf += f"{object_id} 0 obj\n".encode("latin-1")
        pdf += objects[object_id]
        pdf += b"\nendobj\n"

    xref_offset = len(pdf)
    pdf += f"xref\n0 {max_id + 1}\n".encode("latin-1")
    pdf += b"0000000000 65535 f \n"
    for object_id in range(1, max_id + 1):
        pdf += f"{offsets[object_id]:010d} 00000 n \n".encode("latin-1")
    pdf += (
        f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    return pdf


def build_report_pdf(report: dict[str, object]) -> bytes:
    prediction = dict(report.get("prediction") or {})
    profile = dict(report.get("profile") or {})
    context = dict(report.get("context") or {})
    agp_summary = dict((report.get("agp") or {}).get("summary") or {})
    alert_log = list(report.get("alert_log") or [])
    top_factors = list(prediction.get("top_factors") or [])

    lines = [
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Patient",
        f"  Patient ID: {_stringify(report.get('patient_id'))}",
        f"  Name: {_stringify(profile.get('name'))}",
        f"  Age: {_stringify(profile.get('age'))}",
        f"  Diabetes type: {_stringify(profile.get('diabetes_type'))}",
        f"  Insulin therapy: {_stringify(profile.get('insulin_therapy'))}",
        "",
        "Current status",
        f"  Current glucose: {_stringify(report.get('current_glucose'))} mg/dL",
        f"  15-minute trend: {_stringify(report.get('roc_15'))} mg/dL",
        f"  Prediction status: {_stringify(prediction.get('status'))}",
        f"  Risk level: {_stringify(prediction.get('risk_level'))}",
        f"  Risk score: {_stringify(prediction.get('hypo_probability'))}",
        f"  30-minute forecast: {_stringify(prediction.get('predicted_glucose_30min'))} mg/dL",
        f"  Watch status: {_stringify(prediction.get('watch_status'))}",
        f"  Top reason: {_stringify(prediction.get('top_reason'))}",
        f"  Explanation: {_stringify(prediction.get('explanation'))}",
        "",
        "Recent context",
        f"  Carbs in last hour: {_stringify(context.get('carbs_1h'))} g",
        f"  Carbs in last 2 hours: {_stringify(context.get('carbs_2h'))} g",
        f"  Insulin on board: {_stringify(context.get('insulin_on_board'))} U",
        f"  Activity level: {_stringify(context.get('activity'))}",
        f"  Sleep flag: {_stringify(context.get('sleep_flag'))}",
        f"  Stress score: {_stringify(context.get('stress_score'))}",
        "",
        "AGP summary",
        f"  Mean glucose: {_stringify(agp_summary.get('mean_glucose'))} mg/dL",
        f"  Time in range: {_stringify(agp_summary.get('time_in_range'))}",
        f"  Time below range: {_stringify(agp_summary.get('time_below_range'))}",
        f"  Time above range: {_stringify(agp_summary.get('time_above_range'))}",
        f"  Lowest glucose: {_stringify(agp_summary.get('lowest_glucose'))} mg/dL",
        f"  Highest glucose: {_stringify(agp_summary.get('highest_glucose'))} mg/dL",
        "",
        "Top model factors",
    ]

    if top_factors:
        for factor in top_factors[:5]:
            lines.append(
                f"  - {_stringify(factor.get('feature'))}: {_stringify(factor.get('message'))}"
            )
    else:
        lines.append("  No model factors available for this prediction.")

    lines.extend(["", "Recent alerts"])
    if alert_log:
        for entry in alert_log[-8:]:
            lines.append(
                f"  - {_stringify(entry.get('timestamp'))}: {_stringify(entry.get('risk_level'))} risk, score {_stringify(entry.get('hypo_probability'))}, actual hypo={_stringify(entry.get('actual_hypo'))}"
            )
    else:
        lines.append("  No recent alerts recorded.")

    lines.extend(["", "Recent glucose trace"])
    recent_trace = list(report.get("recent_trace") or [])
    if recent_trace:
        for point in recent_trace[-12:]:
            if isinstance(point, dict):
                lines.append(
                    f"  - {_stringify(point.get('timestamp'))}: {_stringify(point.get('glucose'))} mg/dL"
                )
    else:
        lines.append("  No recent trace available.")

    chunked_lines = _line_chunks(lines)
    lines_per_page = 44
    pages = [
        chunked_lines[index : index + lines_per_page]
        for index in range(0, len(chunked_lines), lines_per_page)
    ] or [["No report data available."]]
    return _render_pdf("GlycoGuard Patient Report", pages)

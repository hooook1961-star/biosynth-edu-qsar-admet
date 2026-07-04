"""Student report generation for BioSynth-EDU.

This module owns report structure and rendering only. Student-facing report
copy lives in ``core.report_text``.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any, Mapping

from core.i18n import disclaimer, limitations, methodology_sections, normalize_language, student_questions
from core.report_text import normalize_report_text, report_labels, score_rows

REPORT_SCHEMA_VERSION = "student_report_v1.0"


def build_student_report(explanation_dict: Mapping[str, Any], lang: str | None = None) -> dict[str, Any]:
    """Build a serializable student-report payload."""
    selected_lang = normalize_language(lang or (explanation_dict.get("language") if isinstance(explanation_dict, Mapping) else None))
    molecule = dict(explanation_dict.get("molecule", {}) or {})
    model_outputs = dict(explanation_dict.get("model_outputs", {}) or {})
    decision = dict(explanation_dict.get("decision_explanation", {}) or {})
    applicability = dict(explanation_dict.get("applicability_domain", {}) or {})
    uncertainty = dict(explanation_dict.get("uncertainty", {}) or {})
    matrix = dict(explanation_dict.get("bbb_pgp_matrix", {}) or {})
    input_disclaimers = dict(explanation_dict.get("disclaimers", {}) or {})

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "language": selected_lang,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "title": report_labels(selected_lang)["title"],
        "molecule": {
            "input_smiles": molecule.get("input_smiles"),
            "canonical_smiles": molecule.get("canonical_smiles"),
            "valid": bool(molecule.get("valid", False)),
            "warnings": _normalise_warnings(molecule.get("warnings", [])),
        },
        "executive_summary": {
            "final_class": decision.get("final_class") or model_outputs.get("final_cns_class"),
            "final_label_ru": decision.get("final_label") or decision.get("final_label_ru") or decision.get("title") or "",
            "summary": decision.get("summary") or "",
            "student_interpretation": decision.get("student_interpretation") or "",
            "uncertainty_level": uncertainty.get("level", "unknown"),
            "uncertainty_message": uncertainty.get("student_message", ""),
        },
        "scores": {
            "bbb_normalized_score": model_outputs.get("bbb_classifier_probability"),
            "gupta_bbb_score": model_outputs.get("bbb_v2_score"),
            "gupta_v1_score": model_outputs.get("bbb_v1_score"),
            "pgp_probability": model_outputs.get("pgp_probability"),
            "pka_pred": model_outputs.get("pka_pred"),
            "clint_risk": model_outputs.get("clint_risk"),
            "catmos_ld50": model_outputs.get("catmos_ld50"),
            "final_cns_class": model_outputs.get("final_cns_class") or decision.get("final_class"),
        },
        "applicability_domain": {
            "level": applicability.get("level", "unknown"),
            "student_message": applicability.get("student_message", ""),
            "reasons": list(applicability.get("reasons") or []),
        },
        "descriptors_table": _build_descriptor_rows(explanation_dict),
        "factor_summary": _build_factor_summary(explanation_dict),
        "bbb_pgp_matrix": {
            "current_cell": matrix.get("current_cell"),
            "current_interpretation": matrix.get("current_interpretation"),
        },
        "stepwise_trace": _build_step_rows(explanation_dict),
        "methodology": methodology_sections(selected_lang),
        "limitations": limitations(selected_lang),
        "student_questions": student_questions(selected_lang),
        "disclaimers": {
            "in_silico": input_disclaimers.get("in_silico") or disclaimer("in_silico", selected_lang),
            "what_if": input_disclaimers.get("what_if") or disclaimer("what_if", selected_lang),
        },
    }
    return normalize_report_text(report, selected_lang)


def render_report_markdown(report: Mapping[str, Any], lang: str | None = None) -> str:
    """Render a report payload as localized Markdown."""
    selected_lang = normalize_language(lang or report.get("language"))
    report = normalize_report_text(report, selected_lang)
    labels = report_labels(selected_lang)
    lines: list[str] = []
    a = lines.append

    a(f"# {report.get('title', labels['title'])}")
    a("")
    a(f"{labels['generated']}: `{report.get('generated_at_utc', 'N/A')}`")
    a("")

    molecule = report.get("molecule", {})
    a(f"## 1. {labels['molecule']}")
    a("")
    a(f"- {labels['input_smiles']}: `{_fmt(molecule.get('input_smiles'))}`")
    a(f"- {labels['canonical_smiles']}: `{_fmt(molecule.get('canonical_smiles'))}`")
    a(f"- {labels['validity']}: {labels['valid'] if molecule.get('valid') else labels['invalid']}")
    warnings = molecule.get("warnings") or []
    if warnings:
        a(f"- {labels['warnings']}:")
        for warning in warnings:
            a(f"  - {_fmt(warning.get('message'))}")
    else:
        a(f"- {labels['warnings']}: {labels['none']}")
    a("")

    summary = report.get("executive_summary", {})
    a(f"## 2. {labels['summary']}")
    a("")
    a(f"**{_fmt(summary.get('final_label_ru'))}**")
    a("")
    a(_fmt(summary.get("summary")))
    if summary.get("student_interpretation"):
        a("")
        a(_fmt(summary.get("student_interpretation")))
    if summary.get("uncertainty_message"):
        a("")
        a(
            f"{labels['uncertainty']}: "
            f"`{_fmt(summary.get('uncertainty_level_label') or summary.get('uncertainty_level'))}`. "
            f"{_fmt(summary.get('uncertainty_message'))}"
        )
    a("")

    a(f"## 3. {labels['scores']}")
    a("")
    a(f"| {labels['indicator']} | {labels['value']} |")
    a("|---|---:|")
    scores = report.get("scores") or {}
    for label, key in score_rows(selected_lang):
        a(f"| {label} | {_fmt(scores.get(key))} |")
    a("")

    a(f"## 4. {labels['descriptors']}")
    rows = report.get("descriptors_table") or []
    if rows:
        a(f"| {labels['descriptor']} | {labels['value']} | {labels['zone']} | {labels['effect']} | {labels['short_expl']} |")
        a("|---|---:|---|---|---|")
        for row in rows:
            a(
                "| "
                + " | ".join(
                    [
                        _fmt(row.get("label")),
                        _fmt(row.get("value_with_unit")),
                        _fmt(row.get("zone_label")),
                        _fmt(row.get("effect_label")),
                        _fmt(row.get("short_explanation")),
                    ]
                )
                + " |"
            )
    else:
        a("N/A")
    a("")

    factors = report.get("factor_summary", {})
    a(f"## 5. {labels['factors']}")
    _append_factor_markdown(lines, labels["positive"], factors.get("positive", []), labels)
    _append_factor_markdown(lines, labels["negative"], factors.get("negative", []), labels)
    _append_factor_markdown(lines, labels["borderline"], factors.get("borderline", []), labels)

    matrix = report.get("bbb_pgp_matrix", {})
    a(f"## 6. {labels['matrix']}")
    a("")
    a(f"- {labels['current']}: `{_fmt(matrix.get('current_cell'))}`")
    a(f"- {labels['interpretation']}: {_fmt(matrix.get('current_interpretation'))}")
    a("")

    a(f"## 7. {labels['steps']}")
    a("")
    for step in report.get("stepwise_trace") or []:
        a(f"### {labels['step']} {step.get('step')}. {_fmt(step.get('title'))}")
        a(f"{labels['status']}: `{_fmt(step.get('status'))}`")
        a("")
        a(_fmt(step.get("message")))
        a("")

    a(f"## 8. {labels['methodology']}")
    a("")
    for section in report.get("methodology") or []:
        a(f"### {_fmt(section.get('title'))}")
        a(_fmt(section.get("text")))
        a("")

    a(f"## 9. {labels['limitations']}")
    a("")
    for item in report.get("limitations") or []:
        a(f"- {_fmt(item)}")
    a("")

    a(f"## 10. {labels['questions']}")
    a("")
    for item in report.get("student_questions") or []:
        a(f"- {_fmt(item)}")
    a("")

    disclaimers = report.get("disclaimers", {})
    a(f"## 11. {labels['disclaimers']}")
    a("")
    a(f"**{labels['in_silico_label']}:** {_fmt(disclaimers.get('in_silico'))}")
    a("")
    a(f"**{labels['what_if_label']}:** {_fmt(disclaimers.get('what_if'))}")
    a("")
    return "\n".join(lines).strip() + "\n"


def render_report_html(report: Mapping[str, Any], lang: str | None = None) -> str:
    """Render a standalone localized HTML report."""
    selected_lang = normalize_language(lang or report.get("language"))
    report = normalize_report_text(report, selected_lang)
    labels = report_labels(selected_lang)
    markdown_report = render_report_markdown(report, selected_lang)
    title = _e(report.get("title", labels["title"]))
    summary = report.get("executive_summary", {})
    molecule = report.get("molecule") or {}
    validity = labels["valid"] if molecule.get("valid") else labels["invalid"]
    rows = report.get("descriptors_table") or []
    factors = report.get("factor_summary") or {}
    matrix = report.get("bbb_pgp_matrix") or {}

    body = [
        f"<h1>{title}</h1>",
        f"<p class='muted'>{_e(labels['generated'])}: <code>{_e(report.get('generated_at_utc'))}</code></p>",
        (
            f"<section><h2>1. {_e(labels['molecule'])}</h2>"
            f"<p><strong>{_e(labels['input_smiles'])}:</strong> <code>{_e(molecule.get('input_smiles'))}</code></p>"
            f"<p><strong>{_e(labels['canonical_smiles'])}:</strong> <code>{_e(molecule.get('canonical_smiles'))}</code></p>"
            f"<p><strong>{_e(labels['validity'])}:</strong> {_e(validity)}</p></section>"
        ),
        (
            f"<section><h2>2. {_e(labels['summary'])}</h2><div class='callout'>"
            f"<strong>{_e(summary.get('final_label_ru'))}</strong><br>"
            f"{_e(summary.get('summary'))}<br>{_e(summary.get('student_interpretation'))}</div></section>"
        ),
        f"<section><h2>3. {_e(labels['scores'])}</h2>{_html_score_table(report, selected_lang)}</section>",
        f"<section><h2>4. {_e(labels['descriptors'])}</h2>{_html_descriptor_table(rows, labels)}</section>",
        (
            f"<section><h2>5. {_e(labels['factors'])}</h2>"
            + _html_factor_group(labels["positive"], factors.get("positive", []), "positive", labels)
            + _html_factor_group(labels["negative"], factors.get("negative", []), "negative", labels)
            + _html_factor_group(labels["borderline"], factors.get("borderline", []), "borderline", labels)
            + "</section>"
        ),
        (
            f"<section><h2>6. {_e(labels['matrix'])}</h2>"
            f"<p><strong>{_e(labels['current'])}:</strong> <code>{_e(matrix.get('current_cell'))}</code></p>"
            f"<p><strong>{_e(labels['interpretation'])}:</strong> {_e(matrix.get('current_interpretation'))}</p></section>"
        ),
        (
            f"<section><h2>8. {_e(labels['methodology'])}</h2>"
            + "".join(f"<h3>{_e(s.get('title'))}</h3><p>{_e(s.get('text'))}</p>" for s in report.get("methodology") or [])
            + "</section>"
        ),
        (
            f"<section><h2>9. {_e(labels['limitations'])}</h2><ul>"
            + "".join(f"<li>{_e(x)}</li>" for x in report.get("limitations") or [])
            + "</ul></section>"
        ),
    ]
    escaped_markdown = _e(markdown_report)
    return f"""<!doctype html>
<html lang="{selected_lang}">
<head><meta charset="utf-8"><title>{title}</title><style>body {{ font-family: Arial, sans-serif; line-height: 1.55; max-width: 1100px; margin: 32px auto; padding: 0 24px; color: #1f2933; }} code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }} table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }} th, td {{ border: 1px solid #d0d7de; padding: 8px; vertical-align: top; }} th {{ background: #f6f8fa; }} section {{ margin: 28px 0; }} .callout {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 14px 16px; border-radius: 6px; }} .muted {{ color: #667085; }} .positive {{ border-left: 4px solid #16a34a; padding-left: 12px; }} .negative {{ border-left: 4px solid #dc2626; padding-left: 12px; }} .borderline {{ border-left: 4px solid #ca8a04; padding-left: 12px; }}</style></head>
<body>{''.join(body)}<details><summary>{_e(labels['markdown'])}</summary><pre>{escaped_markdown}</pre></details></body></html>
"""


def build_markdown_report(explanation_dict: Mapping[str, Any], lang: str | None = None) -> str:
    return render_report_markdown(build_student_report(explanation_dict, lang=lang), lang=lang)


def build_html_report(explanation_dict: Mapping[str, Any], lang: str | None = None) -> str:
    return render_report_html(build_student_report(explanation_dict, lang=lang), lang=lang)


def build_report_filename(explanation_dict: Mapping[str, Any], extension: str = "html") -> str:
    """Build a safe filename for report downloads."""
    molecule = explanation_dict.get("molecule", {}) if isinstance(explanation_dict, Mapping) else {}
    raw = str(molecule.get("canonical_smiles") or molecule.get("input_smiles") or "molecule")
    digest = sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]
    short = re.sub(r"[^A-Za-z0-9_-]+", "_", raw)[:36].strip("_") or "molecule"
    ext = extension.lower().lstrip(".") or "html"
    return f"biosynth_edu_report_{short}_{digest}.{ext}"


def _build_descriptor_rows(explanation_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    descriptors = explanation_dict.get("descriptors", {}) or {}
    for name, item in descriptors.items():
        if not isinstance(item, Mapping):
            continue
        value = item.get("value")
        unit = item.get("unit") or ""
        explanation = item.get("explanation") or ""
        rows.append(
            {
                "name": name,
                "label": item.get("short_label") or item.get("display_name") or name,
                "display_name": item.get("display_name") or name,
                "value": value,
                "unit": unit,
                "value_with_unit": f"{_fmt(value)} {unit}".strip(),
                "zone": item.get("zone", "gray"),
                "zone_label": item.get("zone_label") or item.get("zone", "gray"),
                "effect": item.get("effect"),
                "effect_label": item.get("effect_label") or item.get("effect") or "",
                "short_explanation": _first_sentence(explanation),
                "full_explanation": explanation,
            }
        )
    return rows


def _build_factor_summary(explanation_dict: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    source = explanation_dict.get("factor_summary", {}) or {}
    result: dict[str, list[dict[str, Any]]] = {"positive": [], "negative": [], "borderline": []}
    for group in result:
        for item in source.get(group, []) or []:
            if isinstance(item, Mapping):
                result[group].append(
                    {
                        "name": item.get("name"),
                        "display_name": item.get("display_name") or item.get("name"),
                        "value": item.get("value"),
                        "reason": item.get("reason", ""),
                    }
                )
    return result


def _build_step_rows(explanation_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for step in explanation_dict.get("stepwise_trace", []) or []:
        if isinstance(step, Mapping):
            rows.append(
                {
                    "step": step.get("step"),
                    "title": step.get("title"),
                    "status": step.get("status"),
                    "message": step.get("message"),
                }
            )
    return rows


def _normalise_warnings(warnings: Any) -> list[dict[str, str]]:
    if not warnings:
        return []
    result = []
    for item in warnings:
        if isinstance(item, Mapping):
            result.append(
                {
                    "code": str(item.get("code", "warning")),
                    "severity": str(item.get("severity", "warning")),
                    "message": str(item.get("message", item)),
                }
            )
        else:
            result.append({"code": "warning", "severity": "warning", "message": str(item)})
    return result


def _append_factor_markdown(lines: list[str], title: str, factors: list[Mapping[str, Any]], labels: Mapping[str, str]) -> None:
    lines.append(f"### {title}")
    if not factors:
        lines.append(labels["no_factors"])
        lines.append("")
        return
    for item in factors:
        lines.append(f"- **{_fmt(item.get('display_name') or item.get('name'))}** = `{_fmt(item.get('value'))}` - {_fmt(item.get('reason'))}")
    lines.append("")


def _html_score_table(report: Mapping[str, Any], lang: str) -> str:
    labels = report_labels(lang)
    scores = report.get("scores") or {}
    out = [f"<table><thead><tr><th>{_e(labels['indicator'])}</th><th>{_e(labels['value'])}</th></tr></thead><tbody>"]
    for label, key in score_rows(lang):
        out.append(f"<tr><td>{_e(label)}</td><td>{_e(scores.get(key))}</td></tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _html_descriptor_table(rows: list[Mapping[str, Any]], labels: Mapping[str, str]) -> str:
    if not rows:
        return f"<p>{_e(labels['none'])}</p>"
    out = [
        "<table><thead><tr>"
        f"<th>{_e(labels['descriptor'])}</th>"
        f"<th>{_e(labels['value'])}</th>"
        f"<th>{_e(labels['zone'])}</th>"
        f"<th>{_e(labels['effect'])}</th>"
        f"<th>{_e(labels['short_expl'])}</th>"
        "</tr></thead><tbody>"
    ]
    for row in rows:
        out.append(
            "<tr>"
            f"<td>{_e(row.get('label'))}</td>"
            f"<td>{_e(row.get('value_with_unit'))}</td>"
            f"<td>{_e(row.get('zone_label'))}</td>"
            f"<td>{_e(row.get('effect_label'))}</td>"
            f"<td>{_e(row.get('short_explanation'))}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    return "".join(out)


def _html_factor_group(title: str, factors: list[Mapping[str, Any]], css_class: str, labels: Mapping[str, str]) -> str:
    out = [f"<div class='{css_class}'><h3>{_e(title)}</h3>"]
    if not factors:
        out.append(f"<p>{_e(labels['no_factors'])}</p></div>")
        return "".join(out)
    out.append("<ul>")
    for item in factors:
        out.append(
            f"<li><strong>{_e(item.get('display_name') or item.get('name'))}</strong> = "
            f"<code>{_e(item.get('value'))}</code>. {_e(item.get('reason'))}</li>"
        )
    out.append("</ul></div>")
    return "".join(out)


def _first_sentence(text: Any) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    for separator in [". ", "! ", "? "]:
        if separator in text:
            return text.split(separator)[0].strip() + separator.strip()
    return text


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3g}"
    try:
        numeric = float(value)
        if str(value).strip() and re.fullmatch(r"[-+]?\d+(\.\d+)?", str(value).strip()):
            if abs(numeric) < 1000:
                return f"{numeric:.3g}"
            return f"{numeric:.1f}"
    except (TypeError, ValueError):
        pass
    return str(value).replace("\n", " ").strip()


def _e(value: Any) -> str:
    return html.escape(_fmt(value), quote=True)

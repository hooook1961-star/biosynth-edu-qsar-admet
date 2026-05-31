"""Student report generation for BioSynth-EDU explainable ADMET/BBB mode.

Stage 4 adds a pure backend report layer. It receives the same
``explanation_dict`` that powers the Streamlit explainability tabs and converts
it into a stable report payload, Markdown text and standalone HTML.

The module intentionally does not import Streamlit, RDKit or ML models. It is
safe to test without the web UI and can later be reused for PDF export.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any, Mapping

from core.teaching_templates import IN_SILICO_DISCLAIMER_RU, WHAT_IF_DISCLAIMER_RU
from core.i18n import methodology_sections, limitations as localized_limitations, student_questions, normalize_language, t

REPORT_SCHEMA_VERSION = "student_report_v1.0"

METHODOLOGY_SECTIONS = [
    {
        "title": "1. Проверка структуры",
        "text": (
            "SMILES сначала преобразуется в молекулярный граф. Если RDKit не может "
            "распознать структуру, расчёт дескрипторов и объяснение модели не выполняются."
        ),
    },
    {
        "title": "2. Физико-химические дескрипторы",
        "text": (
            "Для учебного объяснения используются понятные признаки: MW, LogP, TPSA, HBD, HBA, "
            "число вращаемых связей, ароматические кольца, pKa, заряд и P-gp probability."
        ),
    },
    {
        "title": "3. Пассивная BBB-проницаемость",
        "text": (
            "BBB-блок интерпретирует физико-химический профиль молекулы: липофильность, размер, "
            "полярность, ионизацию и способность образовывать водородные связи."
        ),
    },
    {
        "title": "4. P-gp эффлюкс",
        "text": (
            "P-gp рассматривается как отдельный биологический механизм. Он может снижать CNS-доступность "
            "даже тогда, когда пассивная BBB-проницаемость выглядит благоприятной."
        ),
    },
    {
        "title": "5. Rule-based explainability",
        "text": (
            "Объяснение в учебном режиме строится по правилам и типичным диапазонам дескрипторов. "
            "Это делает вывод понятным студенту, но не заменяет экспериментальную валидацию."
        ),
    },
]

LIMITATIONS = [
    "BioSynth-EDU предоставляет in silico-прогноз, а не медицинскую рекомендацию.",
    "Прогноз не является экспериментальным доказательством BBB-проницаемости, токсичности или эффективности.",
    "Вероятности и score моделей следует трактовать как модельные оценки, а не как абсолютную биологическую истину.",
    "What-if режим меняет дескрипторы без изменения структуры, поэтому это педагогическая симуляция, а не прогноз новой молекулы.",
    "Для солей, смесей, полифенолов, гликозидов, природных соединений и очень крупных молекул важно учитывать домен применимости.",
]

STUDENT_QUESTIONS = [
    "Какие два дескриптора сильнее всего поддерживают BBB-проницаемость?",
    "Какой фактор является главным ограничением для CNS-доступности?",
    "Есть ли конфликт между пассивной BBB-проницаемостью и P-gp эффлюксом?",
    "Как изменился бы учебный вывод, если TPSA или P-gp probability стали выше?",
    "Какие предупреждения о домене применимости нужно учесть перед интерпретацией?",
]


def _build_student_report_ru(explanation_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Build a structured student report from ``explanation_dict``.

    The returned payload is intentionally simple and serialisable. It is the
    single source for Markdown and HTML rendering.
    """
    lang = "ru"
    molecule = dict(explanation_dict.get("molecule", {}) or {})
    model_outputs = dict(explanation_dict.get("model_outputs", {}) or {})
    decision = dict(explanation_dict.get("decision_explanation", {}) or {})
    applicability = dict(explanation_dict.get("applicability_domain", {}) or {})
    uncertainty = dict(explanation_dict.get("uncertainty", {}) or {})
    matrix = dict(explanation_dict.get("bbb_pgp_matrix", {}) or {})
    disclaimers = dict(explanation_dict.get("disclaimers", {}) or {})

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "language": lang,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "title": t("report.title", lang),
        "labels": {
            "generated_at": t("report.generated_at", lang),
            "molecule": t("report.molecule", lang),
            "summary": t("report.summary", lang),
            "scores": t("report.scores", lang),
            "descriptors": t("report.descriptors", lang),
            "factors": t("report.factors", lang),
            "matrix": t("section.matrix", lang),
            "steps": t("report.steps", lang),
            "methodology": t("section.methodology", lang),
            "limitations": t("section.limitations", lang),
            "questions": t("report.questions", lang),
            "disclaimers": "Дисклеймеры" if lang == "ru" else ("Ескертпелер" if lang == "kk" else "Disclaimers"),
            "validity": t("label.validity", lang),
            "warnings": "Предупреждения" if lang == "ru" else ("Ескертулер" if lang == "kk" else "Warnings"),
            "none": "нет" if lang == "ru" else ("жоқ" if lang == "kk" else "none"),
            "step": "Шаг" if lang == "ru" else ("Қадам" if lang == "kk" else "Step"),
        },
        "molecule": {
            "input_smiles": molecule.get("input_smiles"),
            "canonical_smiles": molecule.get("canonical_smiles"),
            "valid": bool(molecule.get("valid", False)),
            "warnings": _normalise_warnings(molecule.get("warnings", [])),
        },
        "executive_summary": {
            "final_label_ru": decision.get("final_label") or decision.get("final_label_ru") or decision.get("title") or "Итоговая интерпретация недоступна",
            "summary": decision.get("summary") or "Итоговое резюме недоступно.",
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
            "final_cns_class": model_outputs.get("final_cns_class"),
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
        "methodology": methodology_sections(lang),
        "limitations": localized_limitations(lang),
        "student_questions": student_questions(lang),
        "disclaimers": {
            "in_silico": disclaimers.get("in_silico") or t("msg.in_silico", lang),
            "what_if": disclaimers.get("what_if") or t("msg.what_if_disclaimer", lang),
        },
    }
    return report


def _render_report_markdown_ru(report: Mapping[str, Any]) -> str:
    """Render a student report payload as Markdown."""
    lines: list[str] = []
    append = lines.append

    labels = dict(report.get("labels", {}) or {})
    append(f"# {report.get('title', 'BioSynth-EDU report')}")
    append("")
    append(f"{labels.get('generated_at', 'Дата генерации UTC')}: `{report.get('generated_at_utc', 'N/A')}`")
    append("")

    molecule = report.get("molecule", {})
    append(f"## 1. {labels.get('molecule', 'Молекула')}")
    append("")
    append(f"- Input SMILES: `{_fmt(molecule.get('input_smiles'))}`")
    append(f"- Canonical SMILES: `{_fmt(molecule.get('canonical_smiles'))}`")
    append(f"- {labels.get('validity', 'Валидность')}: {'valid' if molecule.get('valid') else 'invalid'}")
    warnings = molecule.get("warnings") or []
    if warnings:
        append(f"- {labels.get('warnings', 'Предупреждения')}:")
        for warning in warnings:
            append(f"  - {_fmt(warning.get('message'))}")
    else:
        append(f"- {labels.get('warnings', 'Предупреждения')}: {labels.get('none', 'нет')}")
    append("")

    summary = report.get("executive_summary", {})
    append(f"## 2. {labels.get('summary', 'Итоговый учебный вывод')}")
    append("")
    append(f"**{_fmt(summary.get('final_label_ru'))}**")
    append("")
    append(_fmt(summary.get("summary")))
    student_text = summary.get("student_interpretation")
    if student_text:
        append("")
        append(_fmt(student_text))
    if summary.get("uncertainty_message"):
        append("")
        append(f"Уровень неопределённости: `{_fmt(summary.get('uncertainty_level'))}`. {_fmt(summary.get('uncertainty_message'))}")
    append("")

    scores = report.get("scores", {})
    append(f"## 3. {labels.get('scores', 'Основные score')}")
    append("")
    append("| Показатель | Значение |")
    append("|---|---:|")
    for label, key in [
        ("BBB normalized score", "bbb_normalized_score"),
        ("Gupta BBB score", "gupta_bbb_score"),
        ("Gupta V1 score", "gupta_v1_score"),
        ("P-gp probability", "pgp_probability"),
        ("pKa_pred", "pka_pred"),
        ("Clint risk", "clint_risk"),
        ("CATMoS LD50", "catmos_ld50"),
        ("Final CNS class", "final_cns_class"),
    ]:
        append(f"| {label} | {_fmt(scores.get(key))} |")
    append("")

    append(f"## 4. {labels.get('descriptors', 'Дескрипторы и объяснения')}")
    append("")
    rows = report.get("descriptors_table") or []
    if rows:
        append("| Дескриптор | Значение | Зона | Влияние | Короткое объяснение |")
        append("|---|---:|---|---|---|")
        for row in rows:
            append(
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
        append("Дескрипторы недоступны.")
    append("")

    append(f"## 5. {labels.get('factors', 'Факторы за и против')}")
    append("")
    factors = report.get("factor_summary", {})
    _append_factor_markdown(lines, "Поддерживают BBB/CNS", factors.get("positive", []))
    _append_factor_markdown(lines, "Мешают BBB/CNS", factors.get("negative", []))
    _append_factor_markdown(lines, "Пограничные", factors.get("borderline", []))
    append("")

    matrix = report.get("bbb_pgp_matrix", {})
    append(f"## 6. {labels.get('matrix', 'Матрица BBB × P-gp')}")
    append("")
    append(f"- Текущий сценарий: `{_fmt(matrix.get('current_cell'))}`")
    append(f"- Интерпретация: {_fmt(matrix.get('current_interpretation'))}")
    append("")

    append(f"## 7. {labels.get('steps', 'Пошаговое решение модели')}")
    append("")
    for step in report.get("stepwise_trace") or []:
        append(f"### {labels.get('step', 'Шаг')} {step.get('step')}. {_fmt(step.get('title'))}")
        append(f"Статус: `{_fmt(step.get('status'))}`")
        append("")
        append(_fmt(step.get("message")))
        append("")

    append(f"## 8. {labels.get('methodology', 'Методология')}")
    append("")
    for section in report.get("methodology") or []:
        append(f"### {_fmt(section.get('title'))}")
        append(_fmt(section.get("text")))
        append("")

    append(f"## 9. {labels.get('limitations', 'Ограничения')}")
    append("")
    for item in report.get("limitations") or []:
        append(f"- {_fmt(item)}")
    append("")

    append(f"## 10. {labels.get('questions', 'Вопросы для студента')}")
    append("")
    for item in report.get("student_questions") or []:
        append(f"- {_fmt(item)}")
    append("")

    disclaimers = report.get("disclaimers", {})
    append(f"## 11. {labels.get('disclaimers', 'Дисклеймеры')}")
    append("")
    append(f"**In silico:** {_fmt(disclaimers.get('in_silico'))}")
    append("")
    append(f"**What-if:** {_fmt(disclaimers.get('what_if'))}")
    append("")

    return "\n".join(lines).strip() + "\n"


def _render_report_html_ru(report: Mapping[str, Any]) -> str:
    """Render a standalone HTML report."""
    markdown_report = _render_report_markdown_ru(report)
    # We render from structured data rather than relying on a Markdown package.
    title = _e(report.get("title", "BioSynth-EDU report"))
    body_parts: list[str] = []

    body_parts.append(f"<h1>{title}</h1>")
    body_parts.append(f"<p class='muted'>Дата генерации UTC: <code>{_e(report.get('generated_at_utc'))}</code></p>")

    molecule = report.get("molecule", {})
    body_parts.append(f"<section><h2>1. {_e(labels.get('molecule', 'Молекула'))}</h2>")
    body_parts.append("<dl>")
    body_parts.append(f"<dt>Input SMILES</dt><dd><code>{_e(molecule.get('input_smiles'))}</code></dd>")
    body_parts.append(f"<dt>Canonical SMILES</dt><dd><code>{_e(molecule.get('canonical_smiles'))}</code></dd>")
    body_parts.append(f"<dt>Валидность</dt><dd>{'валидна' if molecule.get('valid') else 'невалидна'}</dd>")
    body_parts.append("</dl>")
    warnings = molecule.get("warnings") or []
    if warnings:
        body_parts.append(f"<h3>{_e(labels.get('warnings', 'Предупреждения'))}</h3><ul>")
        for warning in warnings:
            body_parts.append(f"<li>{_e(warning.get('message'))}</li>")
        body_parts.append("</ul>")
    body_parts.append("</section>")

    summary = report.get("executive_summary", {})
    body_parts.append(f"<section><h2>2. {_e(labels.get('summary', 'Итоговый учебный вывод'))}</h2>")
    body_parts.append(f"<div class='callout'><strong>{_e(summary.get('final_label_ru'))}</strong><br>{_e(summary.get('summary'))}<br>{_e(summary.get('student_interpretation'))}</div>")
    if summary.get("uncertainty_message"):
        body_parts.append(f"<p><strong>Неопределённость:</strong> {_e(summary.get('uncertainty_level'))}. {_e(summary.get('uncertainty_message'))}</p>")
    body_parts.append("</section>")

    body_parts.append(f"<section><h2>3. {_e(labels.get('scores', 'Основные score'))}</h2>")
    body_parts.append(_html_key_value_table(report.get("scores", {})))
    body_parts.append("</section>")

    body_parts.append(f"<section><h2>4. {_e(labels.get('descriptors', 'Дескрипторы и объяснения'))}</h2>")
    body_parts.append(_html_descriptor_table(report.get("descriptors_table") or []))
    body_parts.append("</section>")

    factors = report.get("factor_summary", {})
    body_parts.append(f"<section><h2>5. {_e(labels.get('factors', 'Факторы за и против'))}</h2>")
    body_parts.append(_html_factor_group("Поддерживают BBB/CNS", factors.get("positive", []), "positive"))
    body_parts.append(_html_factor_group("Мешают BBB/CNS", factors.get("negative", []), "negative"))
    body_parts.append(_html_factor_group("Пограничные", factors.get("borderline", []), "borderline"))
    body_parts.append("</section>")

    matrix = report.get("bbb_pgp_matrix", {})
    body_parts.append(f"<section><h2>6. {_e(labels.get('matrix', 'Матрица BBB × P-gp'))}</h2>")
    body_parts.append(f"<p><strong>Текущий сценарий:</strong> <code>{_e(matrix.get('current_cell'))}</code></p>")
    body_parts.append(f"<p>{_e(matrix.get('current_interpretation'))}</p>")
    body_parts.append("</section>")

    body_parts.append(f"<section><h2>7. {_e(labels.get('steps', 'Пошаговое решение модели'))}</h2>")
    for step in report.get("stepwise_trace") or []:
        body_parts.append(f"<h3>{_e(labels.get('step', 'Шаг'))} {_e(step.get('step'))}. {_e(step.get('title'))}</h3>")
        body_parts.append(f"<p class='muted'>Статус: <code>{_e(step.get('status'))}</code></p>")
        body_parts.append(f"<p>{_e(step.get('message'))}</p>")
    body_parts.append("</section>")

    body_parts.append(f"<section><h2>8. {_e(labels.get('methodology', 'Методология'))}</h2>")
    for section in report.get("methodology") or []:
        body_parts.append(f"<h3>{_e(section.get('title'))}</h3><p>{_e(section.get('text'))}</p>")
    body_parts.append("</section>")

    body_parts.append("<section><h2>9. Ограничения</h2><ul>")
    for item in report.get("limitations") or []:
        body_parts.append(f"<li>{_e(item)}</li>")
    body_parts.append("</ul></section>")

    body_parts.append("<section><h2>10. Вопросы для студента</h2><ul>")
    for item in report.get("student_questions") or []:
        body_parts.append(f"<li>{_e(item)}</li>")
    body_parts.append("</ul></section>")

    disclaimers = report.get("disclaimers", {})
    body_parts.append(f"<section><h2>11. {_e(labels.get('disclaimers', 'Дисклеймеры'))}</h2>")
    body_parts.append(f"<p><strong>In silico:</strong> {_e(disclaimers.get('in_silico'))}</p>")
    body_parts.append(f"<p><strong>What-if:</strong> {_e(disclaimers.get('what_if'))}</p>")
    body_parts.append("</section>")

    escaped_markdown = _e(markdown_report)
    return f"""<!doctype html>
<html lang="{_e(report.get('language', 'ru'))}">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: Arial, sans-serif; line-height: 1.55; max-width: 1100px; margin: 32px auto; padding: 0 24px; color: #1f2933; }}
code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
th, td {{ border: 1px solid #d0d7de; padding: 8px; vertical-align: top; }}
th {{ background: #f6f8fa; }}
section {{ margin: 28px 0; }}
.callout {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 14px 16px; border-radius: 6px; }}
.muted {{ color: #667085; }}
.positive {{ border-left: 4px solid #16a34a; padding-left: 12px; }}
.negative {{ border-left: 4px solid #dc2626; padding-left: 12px; }}
.borderline {{ border-left: 4px solid #ca8a04; padding-left: 12px; }}
.report-source {{ display: none; white-space: pre-wrap; }}
</style>
</head>
<body>
{''.join(body_parts)}
<details>
<summary>Markdown-версия отчёта</summary>
<pre class="report-source">{escaped_markdown}</pre>
</details>
</body>
</html>
"""


def _build_markdown_report_ru(explanation_dict: Mapping[str, Any]) -> str:
    """Convenience wrapper: explanation_dict -> Markdown."""
    return _render_report_markdown_ru(_build_student_report_ru(explanation_dict))


def _build_html_report_ru(explanation_dict: Mapping[str, Any]) -> str:
    """Convenience wrapper: explanation_dict -> standalone HTML."""
    return _render_report_html_ru(_build_student_report_ru(explanation_dict))


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
    summary = explanation_dict.get("factor_summary", {}) or {}
    result: dict[str, list[dict[str, Any]]] = {"positive": [], "negative": [], "borderline": []}
    for group in result:
        for item in summary.get(group, []) or []:
            if not isinstance(item, Mapping):
                continue
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
        if not isinstance(step, Mapping):
            continue
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


def _append_factor_markdown(lines: list[str], title: str, factors: list[Mapping[str, Any]]) -> None:
    lines.append(f"### {title}")
    if not factors:
        lines.append("Нет явных факторов в этой группе.")
        lines.append("")
        return
    for item in factors:
        lines.append(f"- **{_fmt(item.get('display_name') or item.get('name'))}** = `{_fmt(item.get('value'))}` — {_fmt(item.get('reason'))}")
    lines.append("")


def _html_key_value_table(data: Mapping[str, Any]) -> str:
    rows = []
    for key, value in data.items():
        rows.append(f"<tr><th>{_e(key)}</th><td>{_e(_fmt(value))}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def _html_descriptor_table(rows: list[Mapping[str, Any]]) -> str:
    if not rows:
        return "<p>Дескрипторы недоступны.</p>"
    out = ["<table><thead><tr><th>Дескриптор</th><th>Значение</th><th>Зона</th><th>Влияние</th><th>Объяснение</th></tr></thead><tbody>"]
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


def _html_factor_group(title: str, factors: list[Mapping[str, Any]], css_class: str) -> str:
    out = [f"<div class='{css_class}'><h3>{_e(title)}</h3>"]
    if not factors:
        out.append("<p>Нет явных факторов в этой группе.</p></div>")
        return "".join(out)
    out.append("<ul>")
    for item in factors:
        out.append(
            f"<li><strong>{_e(item.get('display_name') or item.get('name'))}</strong> = "
            f"<code>{_e(_fmt(item.get('value')))}</code>. {_e(item.get('reason'))}</li>"
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


# ---------------------------------------------------------------------------
# Stage 6 multilingual report wrappers
# ---------------------------------------------------------------------------


def build_student_report(explanation_dict: Mapping[str, Any], lang: str | None = None) -> dict[str, Any]:
    """Build a structured student report in ru/kk/en."""
    from core.i18n import normalize_language, methodology_sections, limitations, student_questions, disclaimer

    selected_lang = normalize_language(lang or (explanation_dict.get("language") if isinstance(explanation_dict, Mapping) else None))
    report = _build_student_report_ru(explanation_dict)
    report["language"] = selected_lang
    report["title"] = {
        "ru": "BioSynth-EDU: учебный отчёт Explainable ADMET/BBB",
        "kk": "BioSynth-EDU: Explainable ADMET/BBB оқу есебі",
        "en": "BioSynth-EDU: Explainable ADMET/BBB student report",
    }[selected_lang]
    report["methodology"] = methodology_sections(selected_lang)
    report["limitations"] = limitations(selected_lang)
    report["student_questions"] = student_questions(selected_lang)
    report["disclaimers"] = {
        "in_silico": disclaimer("in_silico", selected_lang),
        "what_if": disclaimer("what_if", selected_lang),
    }
    return report


_REPORT_LABELS = {
    "ru": {
        "generated": "Дата генерации UTC", "molecule": "Молекула", "validity": "Валидность", "valid": "валидна", "invalid": "невалидна", "warnings": "Предупреждения", "none": "нет",
        "summary": "Итоговый учебный вывод", "uncertainty": "Уровень неопределённости", "scores": "Основные score", "indicator": "Показатель", "value": "Значение",
        "descriptors": "Дескрипторы и объяснения", "descriptor": "Дескриптор", "zone": "Зона", "effect": "Влияние", "short_expl": "Короткое объяснение",
        "factors": "Факторы за и против", "positive": "Поддерживают BBB/CNS", "negative": "Мешают BBB/CNS", "borderline": "Пограничные", "no_factors": "Нет явных факторов в этой группе.",
        "matrix": "Матрица BBB × P-gp", "current": "Текущий сценарий", "interpretation": "Интерпретация", "steps": "Пошаговое решение модели", "step": "Шаг", "status": "Статус",
        "methodology": "Методология", "limitations": "Ограничения", "questions": "Вопросы для студента", "disclaimers": "Дисклеймеры", "markdown": "Markdown-версия отчёта",
    },
    "kk": {
        "generated": "UTC генерация уақыты", "molecule": "Молекула", "validity": "Валидтілік", "valid": "валидті", "invalid": "валидті емес", "warnings": "Ескертулер", "none": "жоқ",
        "summary": "Қорытынды оқу тұжырымы", "uncertainty": "Белгісіздік деңгейі", "scores": "Негізгі score", "indicator": "Көрсеткіш", "value": "Мәні",
        "descriptors": "Дескрипторлар және түсіндірмелер", "descriptor": "Дескриптор", "zone": "Аймақ", "effect": "Әсері", "short_expl": "Қысқа түсіндірме",
        "factors": "Оң және теріс факторлар", "positive": "BBB/CNS қолдайды", "negative": "BBB/CNS шектейді", "borderline": "Шекаралық", "no_factors": "Бұл топта айқын фактор жоқ.",
        "matrix": "BBB × P-gp матрицасы", "current": "Ағымдағы сценарий", "interpretation": "Түсіндірме", "steps": "Модель шешімінің қадамдары", "step": "Қадам", "status": "Статус",
        "methodology": "Әдістеме", "limitations": "Шектеулер", "questions": "Студентке сұрақтар", "disclaimers": "Дисклеймерлер", "markdown": "Есептің Markdown нұсқасы",
    },
    "en": {
        "generated": "Generated at UTC", "molecule": "Molecule", "validity": "Validity", "valid": "valid", "invalid": "invalid", "warnings": "Warnings", "none": "none",
        "summary": "Final educational conclusion", "uncertainty": "Uncertainty level", "scores": "Main scores", "indicator": "Indicator", "value": "Value",
        "descriptors": "Descriptors and explanations", "descriptor": "Descriptor", "zone": "Zone", "effect": "Effect", "short_expl": "Short explanation",
        "factors": "Factors for and against", "positive": "Supports BBB/CNS", "negative": "Opposes BBB/CNS", "borderline": "Borderline", "no_factors": "No clear factors in this group.",
        "matrix": "BBB × P-gp matrix", "current": "Current scenario", "interpretation": "Interpretation", "steps": "Stepwise model trace", "step": "Step", "status": "Status",
        "methodology": "Methodology", "limitations": "Limitations", "questions": "Questions for students", "disclaimers": "Disclaimers", "markdown": "Markdown version of the report",
    },
}


def render_report_markdown(report: Mapping[str, Any], lang: str | None = None) -> str:
    """Render a student report payload as localized Markdown."""
    from core.i18n import normalize_language

    selected_lang = normalize_language(lang or report.get("language"))
    L = _REPORT_LABELS[selected_lang]
    lines: list[str] = []
    a = lines.append
    a(f"# {report.get('title', 'BioSynth-EDU report')}")
    a("")
    a(f"{L['generated']}: `{report.get('generated_at_utc', 'N/A')}`")
    a("")
    molecule = report.get("molecule", {})
    a(f"## 1. {L['molecule']}")
    a("")
    a(f"- Input SMILES: `{_fmt(molecule.get('input_smiles'))}`")
    a(f"- Canonical SMILES: `{_fmt(molecule.get('canonical_smiles'))}`")
    a(f"- {L['validity']}: {L['valid'] if molecule.get('valid') else L['invalid']}")
    warnings = molecule.get("warnings") or []
    if warnings:
        a(f"- {L['warnings']}:")
        for warning in warnings:
            a(f"  - {_fmt(warning.get('message'))}")
    else:
        a(f"- {L['warnings']}: {L['none']}")
    a("")
    summary = report.get("executive_summary", {})
    a(f"## 2. {L['summary']}")
    a("")
    a(f"**{_fmt(summary.get('final_label_ru'))}**")
    a("")
    a(_fmt(summary.get("summary")))
    if summary.get("student_interpretation"):
        a("")
        a(_fmt(summary.get("student_interpretation")))
    if summary.get("uncertainty_message"):
        a("")
        a(f"{L['uncertainty']}: `{_fmt(summary.get('uncertainty_level'))}`. {_fmt(summary.get('uncertainty_message'))}")
    a("")
    a(f"## 3. {L['scores']}")
    a("")
    a(f"| {L['indicator']} | {L['value']} |")
    a("|---|---:|")
    for label, key in [("BBB normalized score", "bbb_normalized_score"), ("Gupta BBB score", "gupta_bbb_score"), ("Gupta V1 score", "gupta_v1_score"), ("P-gp probability", "pgp_probability"), ("pKa_pred", "pka_pred"), ("Clint risk", "clint_risk"), ("CATMoS LD50", "catmos_ld50"), ("Final CNS class", "final_cns_class")]:
        a(f"| {label} | {_fmt((report.get('scores') or {}).get(key))} |")
    a("")
    a(f"## 4. {L['descriptors']}")
    rows = report.get("descriptors_table") or []
    if rows:
        a(f"| {L['descriptor']} | {L['value']} | {L['zone']} | {L['effect']} | {L['short_expl']} |")
        a("|---|---:|---|---|---|")
        for row in rows:
            a("| " + " | ".join([_fmt(row.get("label")), _fmt(row.get("value_with_unit")), _fmt(row.get("zone_label")), _fmt(row.get("effect_label")), _fmt(row.get("short_explanation"))]) + " |")
    else:
        a("N/A")
    a("")
    factors = report.get("factor_summary", {})
    a(f"## 5. {L['factors']}")
    _append_factor_markdown_localized(lines, L["positive"], factors.get("positive", []), L)
    _append_factor_markdown_localized(lines, L["negative"], factors.get("negative", []), L)
    _append_factor_markdown_localized(lines, L["borderline"], factors.get("borderline", []), L)
    matrix = report.get("bbb_pgp_matrix", {})
    a(f"## 6. {L['matrix']}")
    a("")
    a(f"- {L['current']}: `{_fmt(matrix.get('current_cell'))}`")
    a(f"- {L['interpretation']}: {_fmt(matrix.get('current_interpretation'))}")
    a("")
    a(f"## 7. {L['steps']}")
    a("")
    for step in report.get("stepwise_trace") or []:
        a(f"### {L['step']} {step.get('step')}. {_fmt(step.get('title'))}")
        a(f"{L['status']}: `{_fmt(step.get('status'))}`")
        a("")
        a(_fmt(step.get("message")))
        a("")
    a(f"## 8. {L['methodology']}")
    a("")
    for section in report.get("methodology") or []:
        a(f"### {_fmt(section.get('title'))}")
        a(_fmt(section.get("text")))
        a("")
    a(f"## 9. {L['limitations']}")
    a("")
    for item in report.get("limitations") or []:
        a(f"- {_fmt(item)}")
    a("")
    a(f"## 10. {L['questions']}")
    a("")
    for item in report.get("student_questions") or []:
        a(f"- {_fmt(item)}")
    a("")
    disclaimers = report.get("disclaimers", {})
    a(f"## 11. {L['disclaimers']}")
    a("")
    a(f"**In silico:** {_fmt(disclaimers.get('in_silico'))}")
    a("")
    a(f"**What-if:** {_fmt(disclaimers.get('what_if'))}")
    a("")
    return "\n".join(lines).strip() + "\n"


def render_report_html(report: Mapping[str, Any], lang: str | None = None) -> str:
    """Render a standalone localized HTML report."""
    from core.i18n import normalize_language

    selected_lang = normalize_language(lang or report.get("language"))
    L = _REPORT_LABELS[selected_lang]
    markdown_report = render_report_markdown(report, selected_lang)
    title = _e(report.get("title", "BioSynth-EDU report"))
    summary = report.get("executive_summary", {})
    rows = report.get("descriptors_table") or []
    body = []
    body.append(f"<h1>{title}</h1>")
    body.append(f"<p class='muted'>{_e(L['generated'])}: <code>{_e(report.get('generated_at_utc'))}</code></p>")
    body.append(f"<section><h2>1. {_e(L['molecule'])}</h2><p><strong>Input SMILES:</strong> <code>{_e((report.get('molecule') or {}).get('input_smiles'))}</code></p><p><strong>Canonical SMILES:</strong> <code>{_e((report.get('molecule') or {}).get('canonical_smiles'))}</code></p></section>")
    body.append(f"<section><h2>2. {_e(L['summary'])}</h2><div class='callout'><strong>{_e(summary.get('final_label_ru'))}</strong><br>{_e(summary.get('summary'))}<br>{_e(summary.get('student_interpretation'))}</div></section>")
    body.append(f"<section><h2>3. {_e(L['scores'])}</h2>{_html_key_value_table(report.get('scores', {}))}</section>")
    body.append(f"<section><h2>4. {_e(L['descriptors'])}</h2>{_html_descriptor_table(rows)}</section>")
    body.append(f"<section><h2>8. {_e(L['methodology'])}</h2>" + "".join(f"<h3>{_e(s.get('title'))}</h3><p>{_e(s.get('text'))}</p>" for s in report.get('methodology') or []) + "</section>")
    body.append(f"<section><h2>9. {_e(L['limitations'])}</h2><ul>" + "".join(f"<li>{_e(x)}</li>" for x in report.get('limitations') or []) + "</ul></section>")
    escaped_markdown = _e(markdown_report)
    return f"""<!doctype html>
<html lang="{selected_lang}">
<head><meta charset="utf-8"><title>{title}</title><style>body {{ font-family: Arial, sans-serif; line-height: 1.55; max-width: 1100px; margin: 32px auto; padding: 0 24px; color: #1f2933; }} code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }} table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }} th, td {{ border: 1px solid #d0d7de; padding: 8px; vertical-align: top; }} th {{ background: #f6f8fa; }} section {{ margin: 28px 0; }} .callout {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 14px 16px; border-radius: 6px; }} .muted {{ color: #667085; }}</style></head>
<body>{''.join(body)}<details><summary>{_e(L['markdown'])}</summary><pre>{escaped_markdown}</pre></details></body></html>
"""


def build_markdown_report(explanation_dict: Mapping[str, Any], lang: str | None = None) -> str:
    return render_report_markdown(build_student_report(explanation_dict, lang=lang), lang=lang)


def build_html_report(explanation_dict: Mapping[str, Any], lang: str | None = None) -> str:
    return render_report_html(build_student_report(explanation_dict, lang=lang), lang=lang)


def _append_factor_markdown_localized(lines: list[str], title: str, factors: list[Mapping[str, Any]], labels: Mapping[str, str]) -> None:
    lines.append(f"### {title}")
    if not factors:
        lines.append(labels["no_factors"])
        lines.append("")
        return
    for item in factors:
        lines.append(f"- **{_fmt(item.get('display_name') or item.get('name'))}** = `{_fmt(item.get('value'))}` — {_fmt(item.get('reason'))}")
    lines.append("")

# ---------------------------------------------------------------------------
# Stage 6.2 HTML report localization override
# ---------------------------------------------------------------------------

def _html_descriptor_table_localized(rows: list[Mapping[str, Any]], labels: Mapping[str, str]) -> str:
    if not rows:
        return f"<p>{_e(labels.get('none', 'N/A'))}</p>"
    out = [
        "<table><thead><tr>"
        f"<th>{_e(labels.get('descriptor', 'Descriptor'))}</th>"
        f"<th>{_e(labels.get('value', 'Value'))}</th>"
        f"<th>{_e(labels.get('zone', 'Zone'))}</th>"
        f"<th>{_e(labels.get('effect', 'Effect'))}</th>"
        f"<th>{_e(labels.get('short_expl', 'Explanation'))}</th>"
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


def _html_factor_group_localized(title: str, factors: list[Mapping[str, Any]], css_class: str, labels: Mapping[str, str]) -> str:
    out = [f"<div class='{css_class}'><h3>{_e(title)}</h3>"]
    if not factors:
        out.append(f"<p>{_e(labels.get('no_factors', 'No clear factors in this group.'))}</p></div>")
        return "".join(out)
    out.append("<ul>")
    for item in factors:
        out.append(
            f"<li><strong>{_e(item.get('display_name') or item.get('name'))}</strong> = "
            f"<code>{_e(_fmt(item.get('value')))}</code>. {_e(item.get('reason'))}</li>"
        )
    out.append("</ul></div>")
    return "".join(out)


def render_report_html(report: Mapping[str, Any], lang: str | None = None) -> str:
    """Render a standalone localized HTML report without Russian fallback headings."""
    from core.i18n import normalize_language

    selected_lang = normalize_language(lang or report.get("language"))
    L = _REPORT_LABELS[selected_lang]
    markdown_report = render_report_markdown(report, selected_lang)
    title = _e(report.get("title", "BioSynth-EDU report"))
    summary = report.get("executive_summary", {})
    rows = report.get("descriptors_table") or []
    factors = report.get("factor_summary") or {}
    matrix = report.get("bbb_pgp_matrix") or {}
    body = []
    body.append(f"<h1>{title}</h1>")
    body.append(f"<p class='muted'>{_e(L['generated'])}: <code>{_e(report.get('generated_at_utc'))}</code></p>")
    molecule = report.get("molecule") or {}
    validity = L["valid"] if molecule.get("valid") else L["invalid"]
    body.append(
        f"<section><h2>1. {_e(L['molecule'])}</h2>"
        f"<p><strong>Input SMILES:</strong> <code>{_e(molecule.get('input_smiles'))}</code></p>"
        f"<p><strong>Canonical SMILES:</strong> <code>{_e(molecule.get('canonical_smiles'))}</code></p>"
        f"<p><strong>{_e(L['validity'])}:</strong> {_e(validity)}</p></section>"
    )
    body.append(f"<section><h2>2. {_e(L['summary'])}</h2><div class='callout'><strong>{_e(summary.get('final_label_ru'))}</strong><br>{_e(summary.get('summary'))}<br>{_e(summary.get('student_interpretation'))}</div></section>")
    body.append(f"<section><h2>3. {_e(L['scores'])}</h2>{_html_key_value_table(report.get('scores', {}))}</section>")
    body.append(f"<section><h2>4. {_e(L['descriptors'])}</h2>{_html_descriptor_table_localized(rows, L)}</section>")
    body.append(f"<section><h2>5. {_e(L['factors'])}</h2>" + _html_factor_group_localized(L['positive'], factors.get('positive', []), 'positive', L) + _html_factor_group_localized(L['negative'], factors.get('negative', []), 'negative', L) + _html_factor_group_localized(L['borderline'], factors.get('borderline', []), 'borderline', L) + "</section>")
    body.append(f"<section><h2>6. {_e(L['matrix'])}</h2><p><strong>{_e(L['current'])}:</strong> <code>{_e(matrix.get('current_cell'))}</code></p><p><strong>{_e(L['interpretation'])}:</strong> {_e(matrix.get('current_interpretation'))}</p></section>")
    body.append(f"<section><h2>8. {_e(L['methodology'])}</h2>" + "".join(f"<h3>{_e(s.get('title'))}</h3><p>{_e(s.get('text'))}</p>" for s in report.get('methodology') or []) + "</section>")
    body.append(f"<section><h2>9. {_e(L['limitations'])}</h2><ul>" + "".join(f"<li>{_e(x)}</li>" for x in report.get('limitations') or []) + "</ul></section>")
    escaped_markdown = _e(markdown_report)
    return f"""<!doctype html>
<html lang="{selected_lang}">
<head><meta charset="utf-8"><title>{title}</title><style>body {{ font-family: Arial, sans-serif; line-height: 1.55; max-width: 1100px; margin: 32px auto; padding: 0 24px; color: #1f2933; }} code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }} table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }} th, td {{ border: 1px solid #d0d7de; padding: 8px; vertical-align: top; }} th {{ background: #f6f8fa; }} section {{ margin: 28px 0; }} .callout {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 14px 16px; border-radius: 6px; }} .muted {{ color: #667085; }}</style></head>
<body>{''.join(body)}<details><summary>{_e(L['markdown'])}</summary><pre>{escaped_markdown}</pre></details></body></html>
"""


def build_html_report(explanation_dict: Mapping[str, Any], lang: str | None = None) -> str:
    return render_report_html(build_student_report(explanation_dict, lang=lang), lang=lang)

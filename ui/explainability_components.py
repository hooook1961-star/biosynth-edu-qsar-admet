"""Streamlit UI components for BioSynth-EDU explainable ADMET/BBB mode.

Internal data keys remain stable; visible labels are resolved through the
text catalogs in ``core.i18n`` and neighboring content modules.
"""

from __future__ import annotations

from typing import Any, Mapping
import hashlib
import json

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from core.i18n import normalize_language, t, zone_badge, language_selectbox_options, language_label_to_code
from core.matrix_text import matrix_current_label, matrix_labels
from core.ml_ui_text import ml_ui_t as _ml_ui_t_802
from core.what_if import simulate_descriptor_change
from core.reporting import build_student_report, render_report_markdown, render_report_html, build_report_filename
import core.batch_explainability as batch_xai

STATUS_RENDERERS = {
    "ok": st.success,
    "warning": st.warning,
    "error": st.error,
    "info": st.info,
}


def _batch_student_table(batch_df: pd.DataFrame, lang: str) -> pd.DataFrame:
    builder = getattr(batch_xai, "build_batch_student_view_dataframe", None)
    if callable(builder):
        return builder(batch_df, lang=lang)
    display_columns = batch_xai.select_batch_display_columns(batch_df)
    return batch_df[display_columns] if display_columns else batch_df


def render_language_selector(default: str = "ru") -> str:
    """Optional helper for app.py sidebars."""
    options = language_selectbox_options()
    default = normalize_language(default)
    index = {"ru": 0, "kk": 1, "en": 2}.get(default, 0)
    label = st.selectbox(t("sidebar.language", default), options, index=index)
    return language_label_to_code(label)


def _collect_what_if_base_descriptors(explanation_dict: Mapping[str, Any]) -> dict[str, Any]:
    what_if_base = explanation_dict.get("what_if_base", {}) or {}
    base = dict(what_if_base.get("base_descriptors") or {})
    descriptors = explanation_dict.get("descriptors", {}) or {}
    for key in ["MW", "LogP", "TPSA", "HBD", "HBA", "pKa_pred", "Pgp_probability"]:
        if key in base:
            continue
        item = descriptors.get(key)
        if isinstance(item, Mapping) and "value" in item:
            base[key] = item.get("value")
    return base


def _what_if_key_prefix(explanation_dict: Mapping[str, Any]) -> str:
    molecule = explanation_dict.get("molecule", {}) or {}
    raw = str(molecule.get("canonical_smiles") or molecule.get("input_smiles") or "molecule")
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]


def _lang(explanation_dict: Mapping[str, Any], lang: str | None = None) -> str:
    return normalize_language(lang or (explanation_dict.get("language") if isinstance(explanation_dict, Mapping) else None) or "ru")


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    try:
        number = float(value)
        return f"{number:.3f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


def _first_sentence(text: Any) -> str:
    value = str(text or "").strip()
    for sep in [". ", "! ", "? "]:
        if sep in value:
            return value.split(sep, 1)[0] + sep.strip()
    return value


def _slider_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = default
    return max(low, min(high, number))


def _slider_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(round(float(value)))
    except Exception:
        number = default
    return max(low, min(high, number))


def _signed_delta(value: Any) -> str | None:
    try:
        number = float(value)
    except Exception:
        return None
    return f"{number:+.3f}".rstrip("0").rstrip(".")


def _progress_value(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, number))

# ---------------------------------------------------------------------------
# Localized rendering helpers
# ---------------------------------------------------------------------------

from core.i18n import (
    normalize_language,
    t,
    zone_label as _i18n_zone_label,
    methodology_sections,
    limitations as _i18n_limitations,
)


def _ui_lang(explanation_dict: Mapping[str, Any] | None = None, lang: str | None = None) -> str:
    if lang:
        return normalize_language(lang)
    if isinstance(explanation_dict, Mapping):
        return normalize_language(explanation_dict.get("language"))
    return "ru"


def _zone_badge(zone: str, lang: str) -> str:
    return _i18n_zone_label(str(zone), lang, badge=True)


def render_explainability_tab(explanation_dict: Mapping[str, Any], lang: str | None = None) -> None:
    lang = _ui_lang(explanation_dict, lang)
    render_decision_summary(explanation_dict, lang=lang)
    st.divider()
    render_molecule_summary(explanation_dict, lang=lang)
    st.divider()
    render_descriptor_table(explanation_dict, lang=lang)
    st.divider()
    render_factor_traffic_light(explanation_dict, lang=lang)
    st.divider()
    render_stepwise_trace(explanation_dict, lang=lang)


def render_decision_summary(explanation_dict: Mapping[str, Any], lang: str | None = None) -> None:
    lang = _ui_lang(explanation_dict, lang)
    decision = explanation_dict.get("decision_explanation", {})
    model_outputs = explanation_dict.get("model_outputs", {})
    uncertainty = explanation_dict.get("uncertainty", {})

    st.markdown(t("section.decision", lang))
    final_class = str(decision.get("final_class", "uncertain_or_borderline"))
    title = decision.get("final_label") or decision.get("final_label_ru") or decision.get("title") or "N/A"
    summary = decision.get("summary", "")
    student_text = decision.get("student_interpretation", "")
    block = f"**{title}**\n\n{summary}\n\n{student_text}"
    if final_class == "likely_cns_active":
        st.success(block)
    elif final_class in {"likely_not_bbb_penetrant", "full_barrier"}:
        st.error(block)
    elif final_class in {"peripheral_action_risk", "uncertain_or_borderline"}:
        st.warning(block)
    else:
        st.info(block)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(t("metric.bbb", lang), _format_value(model_outputs.get("bbb_classifier_probability")))
    with c2:
        st.metric(t("metric.pgp", lang), _format_value(model_outputs.get("pgp_probability")))
    with c3:
        st.metric(t("metric.pka", lang), _format_value(model_outputs.get("pka_pred")))
    if uncertainty:
        message = uncertainty.get("student_message", "")
        if uncertainty.get("level") == "high":
            st.error(message)
        elif uncertainty.get("level") == "medium":
            st.warning(message)
        else:
            st.info(message)


def render_molecule_summary(explanation_dict: Mapping[str, Any], lang: str | None = None) -> None:
    lang = _ui_lang(explanation_dict, lang)
    molecule = explanation_dict.get("molecule", {})
    applicability = explanation_dict.get("applicability_domain", {})
    st.markdown(t("section.molecule", lang))
    col1, col2 = st.columns([1, 2])
    canonical_smiles = molecule.get("canonical_smiles")
    with col1:
        if molecule.get("valid") and canonical_smiles:
            mol = Chem.MolFromSmiles(str(canonical_smiles))
            if mol is not None:
                st.image(Draw.MolToImage(mol, size=(280, 280)), use_container_width=True)
            else:
                st.info(t("msg.structure_unavailable", lang))
        else:
            st.error(t("valid.no", lang))
    with col2:
        st.markdown(f"**{t('label.input_smiles', lang)}**")
        st.code(str(molecule.get("input_smiles") or ""), language="text")
        st.markdown(f"**{t('label.canonical_smiles', lang)}**")
        st.code(str(canonical_smiles or "N/A"), language="text")
        st.markdown(f"**{t('label.validity', lang)}**")
        st.write(t("valid.yes", lang) if molecule.get("valid") else t("valid.no", lang))
        _render_warnings_localized(molecule.get("warnings", []), lang)

    st.markdown(t("applicability.title", lang))
    level = applicability.get("level", "unknown")
    msg = applicability.get("student_message", "")
    if level == "inside":
        st.success(msg)
    elif level == "outside":
        st.error(msg)
    else:
        st.warning(msg)
    reasons = applicability.get("reasons") or []
    if reasons:
        with st.expander(t("warnings.reasons", lang)):
            for reason in reasons:
                st.write(f"- {reason}")


def _render_warnings_localized(warnings: list[Mapping[str, Any]], lang: str) -> None:
    if not warnings:
        st.success(t("warnings.none", lang))
        return
    for warning in warnings:
        severity = str(warning.get("severity", "warning"))
        message = str(warning.get("message", warning))
        if severity == "error":
            st.error(message)
        elif severity == "info":
            st.info(message)
        else:
            st.warning(message)


def render_descriptor_table(explanation_dict: Mapping[str, Any], lang: str | None = None) -> None:
    lang = _ui_lang(explanation_dict, lang)
    st.markdown(t("section.descriptors", lang))
    descriptors = explanation_dict.get("descriptors", {})
    if not descriptors:
        st.warning(t("msg.descriptors_unavailable", lang))
        return
    rows = []
    for name, item in descriptors.items():
        rows.append(
            {
                t("descriptor.column.name", lang): item.get("short_label") or name,
                t("descriptor.column.value", lang): _format_value(item.get("value")),
                t("descriptor.column.unit", lang): item.get("unit", ""),
                t("descriptor.column.zone", lang): _zone_badge(str(item.get("zone", "gray")), lang),
                t("descriptor.column.effect", lang): item.get("effect_label") or item.get("effect") or "",
                t("descriptor.column.meaning", lang): _first_sentence(item.get("explanation", "")),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with st.expander(t("descriptor.details", lang), expanded=False):
        for name, item in descriptors.items():
            badge = _zone_badge(str(item.get("zone", "gray")), lang)
            label = item.get("display_name") or name
            value = _format_value(item.get("value"))
            unit = item.get("unit", "")
            st.markdown(f"#### {badge} {label}: {value} {unit}")
            st.write(item.get("explanation", ""))
            if item.get("threshold_note"):
                st.caption(item.get("threshold_note"))


def render_factor_traffic_light(explanation_dict: Mapping[str, Any], lang: str | None = None) -> None:
    lang = _ui_lang(explanation_dict, lang)
    st.markdown(t("section.factors", lang))
    summary = explanation_dict.get("factor_summary", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        _render_factor_group_localized(t("factors.positive", lang), summary.get("positive", []), "success", lang)
    with col2:
        _render_factor_group_localized(t("factors.negative", lang), summary.get("negative", []), "error", lang)
    with col3:
        _render_factor_group_localized(t("factors.borderline", lang), summary.get("borderline", []), "warning", lang)


def _render_factor_group_localized(title: str, factors: list[Mapping[str, Any]], kind: str, lang: str) -> None:
    st.markdown(f"#### {title}")
    if not factors:
        st.caption(t("factors.empty", lang))
        return
    for factor in factors[:6]:
        text = f"**{factor.get('display_name') or factor.get('name')}** = `{_format_value(factor.get('value'))}`\n\n{factor.get('reason', '')}"
        if kind == "success":
            st.success(text)
        elif kind == "error":
            st.error(text)
        else:
            st.warning(text)


def render_stepwise_trace(explanation_dict: Mapping[str, Any], lang: str | None = None) -> None:
    lang = _ui_lang(explanation_dict, lang)
    st.markdown(t("section.stepwise", lang))
    steps = explanation_dict.get("stepwise_trace", [])
    if not steps:
        st.info(t("msg.stepwise_unavailable", lang))
        return
    for step in steps:
        renderer = STATUS_RENDERERS.get(str(step.get("status", "info")), st.info)
        word = t("label.step", lang)
        renderer(f"**{word} {step.get('step', '')}. {step.get('title', '')}**\n\n{step.get('message', '')}")


def render_bbb_pgp_matrix(explanation_dict: Mapping[str, Any], lang: str | None = None) -> None:
    lang = _ui_lang(explanation_dict, lang)
    matrix = explanation_dict.get("bbb_pgp_matrix", {})
    labels = matrix_labels(lang)
    st.markdown(t("section.matrix", lang))
    if matrix.get("intro_text"):
        st.info(matrix.get("intro_text"))
    current = str(matrix.get("current_cell", "insufficient_data"))
    current_text = matrix.get("current_interpretation", "")
    st.markdown(f"**{labels['current']}:** {matrix_current_label(current, lang)}")
    if current in {"bbb_high_pgp_high", "borderline"}:
        st.warning(current_text)
    else:
        st.info(current_text)
    rows = []
    for key in ["bbb_high_pgp_low", "bbb_high_pgp_high", "bbb_low_pgp_low", "bbb_low_pgp_high"]:
        cell = (matrix.get("cells") or {}).get(key, {})
        rows.append({
            labels["columns"]["bbb"]: cell.get("bbb_label", ""),
            labels["columns"]["pgp"]: cell.get("pgp_label", ""),
            labels["columns"]["scenario"]: cell.get("label", ""),
            labels["columns"]["interpretation"]: cell.get("interpretation", ""),
            labels["columns"]["current"]: "<-" if key == current else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with st.expander(labels["expander"], expanded=True):
        st.write(labels["expander_text"])

def render_what_if_lab(explanation_dict: Mapping[str, Any], lang: str | None = None) -> None:
    lang = _ui_lang(explanation_dict, lang)
    st.markdown(t("section.what_if", lang))
    what_if_base = explanation_dict.get("what_if_base", {})
    st.warning(what_if_base.get("disclaimer", ""))
    base_descriptors = _collect_what_if_base_descriptors(explanation_dict)
    if not base_descriptors:
        st.error(t("what_if.not_enough", lang))
        return
    prefix = _what_if_key_prefix(explanation_dict)
    base_scores = what_if_base.get("base_educational_scores", {})
    if base_scores:
        c1, c2 = st.columns(2)
        with c1:
            st.metric(t("what_if.base_passive_bbb", lang), _format_value(base_scores.get("passive_bbb_score")))
        with c2:
            st.metric(t("what_if.base_cns", lang), _format_value(base_scores.get("cns_score")))
    st.markdown(t("what_if.slider_title", lang))
    col1, col2, col3 = st.columns(3)
    with col1:
        mw = st.slider(t("what_if.slider.mw", lang), 50.0, 800.0, _slider_float(base_descriptors.get("MW"), default=300.0, low=50.0, high=800.0), 1.0, key=f"{prefix}_{lang}_mw")
        logp = st.slider(t("what_if.slider.logp", lang), -2.0, 7.0, _slider_float(base_descriptors.get("LogP"), default=2.5, low=-2.0, high=7.0), 0.1, key=f"{prefix}_{lang}_logp")
    with col2:
        tpsa = st.slider(t("what_if.slider.tpsa", lang), 0.0, 220.0, _slider_float(base_descriptors.get("TPSA"), default=70.0, low=0.0, high=220.0), 1.0, key=f"{prefix}_{lang}_tpsa")
        pka = st.slider(t("what_if.slider.pka", lang), 0.0, 14.0, _slider_float(base_descriptors.get("pKa_pred"), default=7.4, low=0.0, high=14.0), 0.1, key=f"{prefix}_{lang}_pka")
    with col3:
        hbd = st.slider(t("what_if.slider.hbd", lang), 0, 8, _slider_int(base_descriptors.get("HBD"), default=1, low=0, high=8), 1, key=f"{prefix}_{lang}_hbd")
        hba = st.slider(t("what_if.slider.hba", lang), 0, 15, _slider_int(base_descriptors.get("HBA"), default=4, low=0, high=15), 1, key=f"{prefix}_{lang}_hba")
        pgp = st.slider(t("what_if.slider.pgp", lang), 0.0, 1.0, _slider_float(base_descriptors.get("Pgp_probability"), default=0.5, low=0.0, high=1.0), 0.01, key=f"{prefix}_{lang}_pgp")
    simulation = simulate_descriptor_change(base_descriptors, {"MW": mw, "LogP": logp, "TPSA": tpsa, "HBD": hbd, "HBA": hba, "pKa_pred": pka, "Pgp_probability": pgp}, lang=lang)
    _render_what_if_scores_localized(simulation, lang)
    st.divider()
    _render_what_if_components_localized(simulation, lang)
    st.divider()
    _render_what_if_commentary_localized(simulation, lang)


def _render_what_if_scores_localized(simulation: Mapping[str, Any], lang: str) -> None:
    st.markdown(t("what_if.result", lang))
    base = simulation.get("base", {})
    modified = simulation.get("modified", {})
    delta = simulation.get("delta", {})
    c1, c2 = st.columns(2)
    with c1:
        st.metric(t("what_if.passive_bbb", lang), _format_value(modified.get("passive_bbb_score")), delta=_signed_delta(delta.get("passive_bbb_score")))
        st.progress(_progress_value(modified.get("passive_bbb_score")))
        st.caption(f"{t('what_if.base', lang)}: {_format_value(base.get('passive_bbb_score'))}")
    with c2:
        st.metric(t("what_if.educational_cns", lang), _format_value(modified.get("cns_score")), delta=_signed_delta(delta.get("cns_score")))
        st.progress(_progress_value(modified.get("cns_score")))
        st.caption(f"{t('what_if.base', lang)}: {_format_value(base.get('cns_score'))}")


def _render_what_if_components_localized(simulation: Mapping[str, Any], lang: str) -> None:
    st.markdown(t("what_if.changed_factors", lang))
    rows = []
    for item in simulation.get("score_components", []):
        rows.append({t("what_if.col.factor", lang): item.get("display_name") or item.get("name"), t("what_if.col.before", lang): _format_value(item.get("base_value")), t("what_if.col.after", lang): _format_value(item.get("modified_value")), "Δ": _format_value(item.get("value_delta")), t("what_if.col.contribution", lang): _format_value(item.get("weighted_delta"))})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    zones = []
    for item in simulation.get("zone_changes", []):
        zones.append({t("what_if.col.descriptor", lang): item.get("display_name"), t("what_if.col.before", lang): _format_value(item.get("base_value")), t("what_if.col.after", lang): _format_value(item.get("modified_value")), t("what_if.col.from", lang): _zone_badge(item.get("from_zone"), lang), t("what_if.col.to", lang): _zone_badge(item.get("to_zone"), lang)})
    if zones:
        st.dataframe(pd.DataFrame(zones), use_container_width=True, hide_index=True)


def _render_what_if_commentary_localized(simulation: Mapping[str, Any], lang: str) -> None:
    st.markdown(t("what_if.commentary", lang))
    commentary = simulation.get("commentary", {})
    st.info(commentary.get("summary", ""))
    if commentary.get("pgp_note"):
        st.warning(commentary.get("pgp_note"))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(t("what_if.better", lang))
        improvements = commentary.get("improvements", [])
        if improvements:
            for text in improvements:
                st.success(text)
        else:
            st.caption(t("what_if.no_improve", lang))
    with c2:
        st.markdown(t("what_if.worse", lang))
        worsenings = commentary.get("worsenings", [])
        if worsenings:
            for text in worsenings:
                st.error(text)
        else:
            st.caption(t("what_if.no_worse", lang))
    st.caption(commentary.get("teaching_note", ""))


def render_methodology_block(lang: str = "ru") -> None:
    lang = normalize_language(lang)
    st.markdown(t("section.methodology", lang))
    for section in methodology_sections(lang):
        st.markdown(f"#### {section['title']}")
        st.write(section["text"])


def render_limitations_block(explanation_dict: Mapping[str, Any] | None = None, lang: str | None = None) -> None:
    lang = _ui_lang(explanation_dict, lang)
    st.markdown(t("section.limitations", lang))
    disclaimers = (explanation_dict or {}).get("disclaimers", {})
    if disclaimers.get("in_silico"):
        st.warning(disclaimers.get("in_silico"))
    if disclaimers.get("what_if"):
        st.info(disclaimers.get("what_if"))
    for item in _i18n_limitations(lang):
        st.write(f"- {item}")


def render_student_report_tab(explanation_dict: Mapping[str, Any], lang: str | None = None) -> None:
    lang = _ui_lang(explanation_dict, lang)
    st.markdown(t("section.report", lang))
    st.info(t("report.info", lang))
    report = build_student_report(explanation_dict, lang=lang)
    markdown_report = render_report_markdown(report, lang=lang)
    html_report = render_report_html(report, lang=lang)
    scores = report.get("scores", {})
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(t("metric.bbb", lang), _format_value(scores.get("bbb_normalized_score")))
    with c2:
        st.metric(t("metric.pgp", lang), _format_value(scores.get("pgp_probability")))
    with c3:
        st.metric(t("metric.pka", lang), _format_value(scores.get("pka_pred")))
    with st.expander(t("report.preview", lang), expanded=False):
        st.code(markdown_report[:12000], language="markdown")
    import json
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(t("report.download_html", lang), data=html_report.encode("utf-8"), file_name=build_report_filename(explanation_dict, "html"), mime="text/html")
    with c2:
        st.download_button(t("report.download_md", lang), data=markdown_report.encode("utf-8"), file_name=build_report_filename(explanation_dict, "md"), mime="text/markdown")
    with c3:
        st.download_button(t("report.download_json", lang), data=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"), file_name=build_report_filename(explanation_dict, "json"), mime="application/json")


def render_batch_explainability_summary(batch_df: pd.DataFrame, summary: Mapping[str, Any] | None = None, lang: str = "ru") -> None:
    lang = normalize_language(lang)
    st.markdown(t("batch.title", lang))
    st.info(t("batch.info", lang))
    if batch_df is None or batch_df.empty:
        st.warning(t("batch.empty", lang))
        return
    summary = summary or batch_xai.summarize_batch_explanations(batch_df, lang=lang)
    counts = summary.get("final_class_counts") or {}
    priority_counts = summary.get("priority_counts") or {}
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(t("batch.total", lang), _format_value(summary.get("total")))
    with c2:
        st.metric(t("batch.valid", lang), _format_value(summary.get("valid_count")))
    with c3:
        st.metric(t("batch.candidates", lang), _format_value(counts.get("likely_bbb_penetrant_profile", 0)))
    with c4:
        st.metric(t("batch.review_needed", lang), _format_value(priority_counts.get("review", 0) + priority_counts.get("manual_review", 0)))
    c5, c6, c7 = st.columns(3)
    with c5:
        st.metric(t("batch.borderline", lang), _format_value(counts.get("borderline_bbb_profile", 0)))
    with c6:
        st.metric(t("batch.low_bbb", lang), _format_value(counts.get("likely_low_bbb_profile", 0)))
    with c7:
        st.metric(t("batch.invalid", lang), _format_value(summary.get("invalid_count")))
    st.markdown(t("batch.summary", lang))
    st.success(str(summary.get("teaching_summary") or summary.get("teaching_summary_ru") or ""))
    with st.expander(t("batch.table", lang), expanded=True):
        sorted_df = batch_xai.sort_batch_by_explainability_priority(batch_df)
        st.dataframe(_batch_student_table(sorted_df, lang=lang), use_container_width=True)


def render_batch_explainability_result(batch_result: Mapping[str, Any], lang: str = "ru") -> None:
    lang = normalize_language(lang)
    st.markdown(t("batch.title", lang))
    if not batch_result:
        st.warning(t("batch.result_empty", lang))
        return
    rows = pd.DataFrame(batch_result.get("explanation_rows") or [])
    summary = batch_result.get("summary") or (batch_xai.summarize_batch_explanations(rows, lang=lang) if not rows.empty else {})
    render_batch_explainability_summary(rows, summary, lang=lang)


# ---------------------------------------------------------------------------
# Student-facing ML explanation view
# ---------------------------------------------------------------------------
# Student-facing copy lives in core.ml_ui_text; this file keeps only rendering
# and small view-specific formatting helpers.

def _ml_student_group_label_802(row: Mapping[str, Any], lang: str) -> str:
    key = str(row.get("group_key") or "")
    if key == "morgan":
        return _ml_ui_t_802("structural_fragments", lang)
    if key == "maccs":
        return _ml_ui_t_802("structural_keys", lang)
    return str(row.get("group_label") or key or "N/A")


def _ml_value_note_802(row: Mapping[str, Any], lang: str) -> str:
    key = str(row.get("group_key") or "")
    value = _format_value(row.get("active_or_value"))
    if key == "morgan":
        return _ml_ui_t_802("fragments_seen", lang, value=value)
    if key == "maccs":
        return _ml_ui_t_802("keys_seen", lang, value=value)
    return _ml_ui_t_802("scalar_seen", lang, value=value)


def _ml_effect_note_802(row: Mapping[str, Any], method: str, lang: str) -> str:
    if method != "shap":
        return _ml_ui_t_802("effect_importance", lang)
    direction = str(row.get("direction") or "neutral")
    if direction == "positive":
        return _ml_ui_t_802("effect_positive", lang)
    if direction == "negative":
        return _ml_ui_t_802("effect_negative", lang)
    return _ml_ui_t_802("effect_neutral", lang)


def _ml_class_label_802(legacy_name: str, class_value: Any, lang: str) -> str:
    try:
        cls = int(class_value)
    except Exception:
        return str(class_value)
    if legacy_name == "rf_pgp_model":
        return _ml_ui_t_802("class_pgp_1" if cls == 1 else "class_pgp_0", lang)
    if legacy_name == "rf_bbb_model":
        return _ml_ui_t_802("class_bbb_1" if cls == 1 else "class_bbb_0", lang)
    return str(class_value)


def _ml_positive_class_note_802(legacy_name: str, lang: str) -> str:
    if legacy_name == "rf_pgp_model":
        return _ml_ui_t_802("pgp_positive_class", lang)
    if legacy_name == "rf_bbb_model":
        return _ml_ui_t_802("bbb_positive_class", lang)
    return ""


def _ml_model_use_note_802(legacy_name: str, lang: str) -> str:
    if legacy_name == "rf_pgp_model":
        return _ml_ui_t_802("pgp_use", lang)
    if legacy_name == "rf_bbb_model":
        return _ml_ui_t_802("bbb_use", lang)
    return ""


def _ml_method_label_802(method: str, lang: str) -> str:
    if method == "shap":
        return _ml_ui_t_802("method_shap", lang)
    return _ml_ui_t_802("method_fallback", lang)


def render_ml_explainability_tab(explanation_dict: Mapping[str, Any], lang: str | None = None) -> None:
    """Render the student-facing ML explanation."""
    from core.ml_explainability import explain_selected_runtime_models_for_smiles

    lang = _lang(explanation_dict, lang)
    molecule = explanation_dict.get("molecule", {}) or {}
    smiles = molecule.get("canonical_smiles") or molecule.get("input_smiles")

    st.markdown(_ml_ui_t_802("title", lang))
    st.info(_ml_ui_t_802("intro", lang))
    st.caption(_ml_ui_t_802("how_to", lang))

    if not smiles:
        st.warning(_ml_ui_t_802("unavailable", lang))
        return

    with st.spinner(_ml_ui_t_802("loading", lang)):
        ml_data = explain_selected_runtime_models_for_smiles(str(smiles), lang=lang)

    models = ml_data.get("models", {}) or {}
    if not models:
        st.warning(_ml_ui_t_802("unavailable", lang))
        return

    for legacy_name in ["rf_pgp_model", "rf_bbb_model"]:
        item = models.get(legacy_name)
        if not isinstance(item, Mapping):
            continue
        status = item.get("status", "unknown")
        title = item.get("model_label") or legacy_name
        expanded = legacy_name == "rf_pgp_model" and status == "ok"
        with st.expander(f"{title}", expanded=expanded):
            if status != "ok":
                st.warning(item.get("reason") or _ml_ui_t_802("unavailable", lang))
                continue

            st.markdown(f"**{_ml_ui_t_802('student_label', lang)}**")
            note = _ml_model_use_note_802(legacy_name, lang)
            if note:
                st.write(note)
            positive_note = _ml_positive_class_note_802(legacy_name, lang)
            if positive_note:
                st.caption(positive_note)

            prediction = item.get("prediction", {}) or {}
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(_ml_ui_t_802("probability", lang), _format_value(prediction.get("probability")))
            with c2:
                st.markdown(f"**{_ml_ui_t_802('class', lang)}**")
                st.write(_ml_class_label_802(legacy_name, prediction.get("class"), lang))
            with c3:
                st.markdown(f"**{_ml_ui_t_802('method', lang)}**")
                st.write(_ml_method_label_802(str(item.get("method") or ""), lang))

            if str(item.get("method")) != "shap":
                st.warning(_ml_ui_t_802("note_fallback", lang))

            st.markdown(f"#### {_ml_ui_t_802('group_table', lang)}")
            rows = []
            method = str(item.get("method") or "")
            for row in item.get("group_contributions", []) or []:
                rows.append(
                    {
                        _ml_ui_t_802("group", lang): _ml_student_group_label_802(row, lang),
                        _ml_ui_t_802("value", lang): _ml_value_note_802(row, lang),
                        _ml_ui_t_802("effect", lang): _ml_effect_note_802(row, method, lang),
                        _ml_ui_t_802("contribution", lang): _format_value(row.get("contribution" if method == "shap" else "abs_contribution")),
                    }
                )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            with st.expander(_ml_ui_t_802("developer_details", lang), expanded=False):
                st.caption(
                    "Technical view: individual fingerprint bits are shown for debugging and model inspection. "
                    "They are not meant to replace descriptor-based teaching interpretation."
                )
                top_rows = []
                for row in item.get("top_features", []) or []:
                    top_rows.append(
                        {
                            t("ml.column.feature", lang): row.get("feature_label"),
                            t("ml.column.group", lang): row.get("group_label"),
                            t("ml.column.value", lang): _format_value(row.get("value")),
                            t("ml.column.contribution", lang): _format_value(row.get("contribution")),
                        }
                    )
                if top_rows:
                    st.markdown(f"#### {_ml_ui_t_802('top_features', lang)}")
                    st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)
                st.download_button(
                    _ml_ui_t_802("download_json", lang),
                    json.dumps({legacy_name: item}, ensure_ascii=False, indent=2).encode("utf-8"),
                    f"BioSynth_EDU_{legacy_name}_technical_ml_explanation.json",
                    mime="application/json",
                    key=f"download_technical_ml_{legacy_name}_{lang}",
                )

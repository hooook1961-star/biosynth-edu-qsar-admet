"""BioSynth-EDU full student-facing Streamlit app.

- keeps the full educational QSAR/ADMET workflow;
- hides runtime/debug details from normal student mode;
- treats ML/SHAP as an optional advanced explanation, not as the main lesson;
- hides Clint from the main forecast when model selection disables it;
- preserves SMILES, batch text and selected section when the language is changed.
"""

from __future__ import annotations

import io
import os
from typing import Any, Callable

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from core.app_text import tx
from core.i18n import (
    language_label_to_code,
    language_options,
    t,
    warning_message,
)
from core.descriptors import calculate_bbb_descriptors
from core.bbb_calculation import analyze_molecule_cns_profile, generate_admet_visualizations
from core.explainability import build_explanation_dict
from core.explainability_adapter import build_pipeline_result_from_current_app
import core.batch_explainability as batch_xai
from ui.explainability_components import (
    render_batch_explainability_summary,
    render_bbb_pgp_matrix,
    render_explainability_tab,
    render_limitations_block,
    render_methodology_block,
    render_ml_explainability_tab,
    render_student_report_tab,
    render_what_if_lab,
)


DEFAULT_SMILES = "CC1=CC2C3(O1)C=CC(C)(C4C3C(=C)C(=O)O4)C2"
DEFAULT_BATCH_TEXT = "CCN(CC)CC(=O)Nc1c(C)cccc1C\nCCO\ninvalid_smiles"




def init_state() -> None:
    st.session_state.setdefault("lang_label", language_options()[0])
    st.session_state.setdefault("main_mode", "single")
    st.session_state.setdefault("single_section", "forecast")
    st.session_state.setdefault("batch_section", "summary")
    st.session_state.setdefault("single_smiles", DEFAULT_SMILES)
    st.session_state.setdefault("batch_source", "text")
    st.session_state.setdefault("batch_text", DEFAULT_BATCH_TEXT)
    st.session_state.setdefault("batch_input_df", None)
    st.session_state.setdefault("batch_smiles_col_value", None)
    st.session_state.setdefault("batch_keep_mode", False)
    st.session_state.setdefault("developer_mode", False)


def section_selector(
    label: str,
    options: list[str],
    label_func: Callable[[str], str],
    key: str,
    fallback: str | None = None,
) -> str:
    """Stable section selector.

    Streamlit tabs reset on every rerun. Language switching triggers a rerun and
    labels also change, so tabs are replaced by a stable internal section id.
    The UI still behaves like a tab selector, but the selected section survives
    RU/KZ/EN switching.
    """
    if st.session_state.get(key) not in options:
        st.session_state[key] = fallback if fallback in options else options[0]

    return st.radio(label, options=options, format_func=label_func, key=key, horizontal=True)


def _status_key(value: Any) -> str | None:
    raw = str(value).strip()
    lowered = raw.lower()
    mapping = {
        "да": "status.yes",
        "иә": "status.yes",
        "yes": "status.yes",
        "true": "status.yes",
        "1": "status.yes",
        "нет": "status.no",
        "жоқ": "status.no",
        "no": "status.no",
        "false": "status.no",
        "0": "status.no",
        "high": "status.high",
        "высокая": "status.high",
        "высокий": "status.high",
        "жоғары": "status.high",
        "medium": "status.medium",
        "средняя": "status.medium",
        "средний": "status.medium",
        "орташа": "status.medium",
        "low": "status.low",
        "низкая": "status.low",
        "низкий": "status.low",
        "төмен": "status.low",
        "высокий риск": "status.high_risk",
        "high risk": "status.high_risk",
        "жоғары қауіп": "status.high_risk",
        "стабильное": "status.stable",
        "стабильно": "status.stable",
        "stable": "status.stable",
        "тұрақты": "status.stable",
        "ошибка": "status.error",
        "error": "status.error",
        "қате": "status.error",
        "n/a": "status.na",
        "disabled_by_selection": "status.na",
        "fallback_error": "status.error",
        "ok": "status.stable",
        "ok_units_unverified": "status.medium",
        "не субстрат": "status.not_substrate",
        "non-substrate": "status.not_substrate",
        "субстрат емес": "status.not_substrate",
        "субстрат (вымывается)": "status.substrate_efflux",
        "субстрат (активно выводится)": "status.substrate_efflux",
        "субстрат (белсенді шығарылады)": "status.substrate_efflux",
        "substrate (efflux)": "status.substrate_efflux",
    }
    return mapping.get(lowered)


def localize_status(value: Any, lang: str) -> str:
    key = _status_key(value)
    return t(key, lang) if key else str(value)


def localize_pgp_class(class_value: Any, lang: str) -> str:
    try:
        return t("status.yes", lang) if int(class_value) == 1 else t("status.no", lang)
    except Exception:
        return localize_status(class_value, lang)


def value_or_na(value: Any, lang: str) -> str:
    if value is None:
        return tx("common.na", lang)
    if isinstance(value, float):
        return f"{value:.3g}"
    if str(value).lower() in {"none", "nan", "n/a"}:
        return tx("common.na", lang)
    return str(value)


def _main_mode_fallback() -> str:
    if "batch_xai_df" in st.session_state:
        return "batch"
    if isinstance(st.session_state.get("batch_input_df"), pd.DataFrame):
        return "batch"
    return "single"


def _batch_student_table(batch_df: pd.DataFrame, lang: str) -> pd.DataFrame:
    builder = getattr(batch_xai, "build_batch_student_view_dataframe", None)
    if callable(builder):
        return builder(batch_df, lang=lang)
    display_columns = batch_xai.select_batch_display_columns(batch_df)
    return batch_df[display_columns] if display_columns else batch_df


def _batch_interpretation_markdown(lang: str) -> str:
    builder = getattr(batch_xai, "build_screening_interpretation_markdown", None)
    if callable(builder):
        return builder(lang)
    return ""


def descriptor_table(results: dict[str, Any], lang: str) -> pd.DataFrame:
    rows = [
        ("MW", results.get("mw")),
        ("TPSA", results.get("tpsa")),
        ("LogP", results.get("logp")),
        ("H-Donors", results.get("h_donors")),
        ("H-Acceptors", results.get("h_acceptors")),
        ("Aro_R", results.get("aro_r")),
        ("HA", results.get("ha")),
    ]
    if "gupta_components" in results:
        rows.append(("MWHBN", results.get("gupta_components", {}).get("mwhbn_raw")))

    return pd.DataFrame(
        {
            t("forecast.descriptor_col", lang): [name for name, _ in rows],
            t("forecast.value_col", lang): [value_or_na(value, lang) for _, value in rows],
        }
    )


def build_single_molecule_explanation(clean_smiles: str, mol: Chem.Mol, lang: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    desc = calculate_bbb_descriptors(clean_smiles)
    results = analyze_molecule_cns_profile(clean_smiles, desc)
    pipeline_result = build_pipeline_result_from_current_app(
        input_smiles=clean_smiles,
        mol=mol,
        descriptors=desc,
        results=results,
        lang=lang,
    )
    explanation_dict = build_explanation_dict(pipeline_result, lang=lang)
    return desc, results, explanation_dict


def _show_clint_in_forecast(results: dict[str, Any]) -> bool:
    status = str(results.get("clint_model_status") or "").lower()
    value = str(results.get("clint_status") or "").lower()
    if status in {"disabled_by_selection", "not_loaded", "disabled"}:
        return False
    if value in {"n/a", "none", ""}:
        return False
    return True


def render_forecast_tab(mol: Chem.Mol, results: dict[str, Any], lang: str, developer_mode: bool = False) -> None:
    st.info(f"**{tx('forecast.qsar_bridge_title', lang)}.** {tx('forecast.qsar_bridge_text', lang)}")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(t("forecast.structure_title", lang))
        st.image(Draw.MolToImage(mol, size=(320, 320)), use_container_width=True)
        st.markdown(t("forecast.descriptors_title", lang))
        st.table(descriptor_table(results, lang))

    with col2:
        st.markdown(t("forecast.integral_title", lang))
        st.metric(
            t("metric.caco2", lang),
            value_or_na(results.get("caco2_logpapp"), lang),
            delta=localize_status(results.get("caco2_status"), lang),
            help=t("help.caco2", lang),
        )
        st.metric(
            t("metric.bbb_gupta", lang),
            value_or_na(results.get("gupta_bbb_score"), lang),
            help=t("help.bbb_gupta", lang),
        )
        if results.get("bbb_rf_probability") is not None:
            st.metric(
                tx("metric.bbb_rf", lang),
                value_or_na(results.get("bbb_rf_probability"), lang),
                help=tx("help.bbb_rf", lang),
            )
        st.metric(
            t("metric.pgp_class", lang),
            localize_pgp_class(results.get("pgp_substrate_class"), lang),
            help=t("help.pgp_class", lang),
        )
        st.metric(
            t("metric.pgp_probability", lang),
            value_or_na(results.get("pgp_probability"), lang),
            help=t("help.pgp_probability", lang),
        )
        st.metric(
            t("metric.pka", lang),
            value_or_na(results.get("pKa_predicted"), lang),
            help=t("help.pka", lang),
        )
        if _show_clint_in_forecast(results):
            st.metric(
                t("metric.clint", lang),
                localize_status(results.get("clint_status"), lang),
                help=t("help.clint", lang),
            )
        st.metric(
            tx("metric.catmos_score", lang),
            value_or_na(results.get("catmos_score", results.get("catmos_ld50")), lang),
            help=tx("help.catmos_score", lang),
        )

    st.markdown(t("forecast.visualization_title", lang))
    c_vis1, c_vis2 = st.columns(2)
    try:
        visualizations = generate_admet_visualizations(mol)
        with c_vis1:
            st.image(visualizations["radar"], use_container_width=True)
        with c_vis2:
            st.image(visualizations["atom_weights"], use_container_width=True)
    except Exception:
        st.warning("Визуализацию не удалось построить для этой молекулы.")

    if developer_mode:
        with st.expander(tx("forecast.model_status_title", lang), expanded=False):
            st.caption(tx("forecast.model_status_caption", lang))
            qa_warnings = results.get("qa_warnings") or []
            if qa_warnings:
                for warning in qa_warnings:
                    st.warning(str(warning))
            runtime_info = {
                "runtime_model_selection_path": results.get("runtime_model_selection_path"),
                "runtime_primary_map": results.get("runtime_primary_map"),
                "gupta_formula_version": results.get("gupta_formula_version"),
                "model_statuses": results.get("model_statuses"),
                "model_errors": results.get("model_errors"),
            }
            st.json(runtime_info)


def render_single_analysis(lang: str, developer_mode: bool = False) -> None:
    st.subheader(t("single.input_header", lang))
    st.info(f"**{tx('single.learning_note_title', lang)}.** {tx('single.learning_note', lang)}")

    input_smiles = st.text_input(
        t("single.smiles_label", lang),
        key="single_smiles",
    )

    if not input_smiles:
        return

    clean_smiles = input_smiles.strip()
    mol = Chem.MolFromSmiles(clean_smiles)

    if mol is None:
        st.error(warning_message("invalid_smiles", lang))
        return

    try:
        _, results, explanation_dict = build_single_molecule_explanation(clean_smiles, mol, lang)
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        return

    if results.get("error"):
        st.error(results["error"])
        return

    section_labels = {
        "forecast": tx("section.forecast", lang),
        "explain": tx("section.explain", lang),
        "ml": tx("section.ml", lang),
        "what_if": tx("section.what_if", lang),
        "report": tx("section.report", lang),
        "matrix": tx("section.matrix", lang),
        "methodology": tx("section.methodology", lang),
        "limitations": tx("section.limitations", lang),
    }
    section = section_selector(
        tx("nav.single_section", lang),
        list(section_labels.keys()),
        lambda item: section_labels[item],
        key="single_section",
    )

    if section == "forecast":
        render_forecast_tab(mol, results, lang, developer_mode=developer_mode)
    elif section == "explain":
        render_explainability_tab(explanation_dict, lang=lang)
    elif section == "ml":
        render_ml_explainability_tab(explanation_dict, lang=lang)
    elif section == "what_if":
        render_what_if_lab(explanation_dict, lang=lang)
    elif section == "report":
        render_student_report_tab(explanation_dict, lang=lang)
    elif section == "matrix":
        render_bbb_pgp_matrix(explanation_dict, lang=lang)
    elif section == "methodology":
        render_methodology_block(lang=lang)
    elif section == "limitations":
        render_limitations_block(explanation_dict, lang=lang)


def render_batch_screening(lang: str) -> None:
    st.subheader(t("batch.page_title", lang))
    st.info(tx("batch.learning_intro", lang))

    source = st.radio(
        t("batch.source_label", lang),
        options=["file", "text"],
        format_func=lambda value: t("batch.source_file", lang) if value == "file" else t("batch.source_text", lang),
        key="batch_source",
        horizontal=True,
    )

    df = None
    smiles_col = None

    if source == "file":
        f = st.file_uploader(t("batch.file_label", lang), key="batch_file_upload")
        if f:
            df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
            st.session_state["batch_input_df"] = df
            st.session_state["batch_keep_mode"] = True
            smiles_col = st.selectbox(t("batch.smiles_col_label", lang), df.columns, key="batch_smiles_col")
            st.session_state["batch_smiles_col_value"] = smiles_col
        elif isinstance(st.session_state.get("batch_input_df"), pd.DataFrame):
            df = st.session_state["batch_input_df"]
            st.session_state["batch_keep_mode"] = True
            smiles_col = st.session_state.get("batch_smiles_col_value")
            if smiles_col not in list(df.columns):
                smiles_col = df.columns[0]
            smiles_col = st.selectbox(
                t("batch.smiles_col_label", lang),
                df.columns,
                index=list(df.columns).index(smiles_col),
                key="batch_smiles_col_restored",
            )
            st.session_state["batch_smiles_col_value"] = smiles_col
    else:
        text = st.text_area(
            t("batch.text_area_label", lang),
            key="batch_text",
        )
        if text:
            df = pd.DataFrame({"SMILES": [s.strip() for s in text.split("\n") if s.strip()]})
            st.session_state["batch_input_df"] = df
            st.session_state["batch_keep_mode"] = True
            smiles_col = "SMILES"
            st.session_state["batch_smiles_col_value"] = smiles_col

    if df is not None:
        st.caption(t("batch.loaded_rows", lang, n=len(df)))

    if df is not None and st.button(t("batch.run_button", lang), key="batch_run_button"):
        results_rows: list[dict[str, Any]] = []
        bar = st.progress(0)
        status = st.empty()

        for i, row in df.iterrows():
            progress = (i + 1) / len(df)
            bar.progress(progress)
            status.caption(t("batch.progress", lang, progress=f"{progress:.0%}"))

            smi = str(row[smiles_col]).strip()
            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                results_rows.append({"SMILES": smi, "error": t("batch.invalid_smiles_error", lang)})
                continue

            try:
                d = calculate_bbb_descriptors(smi)
                res = analyze_molecule_cns_profile(smi, d)
                results_rows.append({**{"SMILES": smi}, **d, **res})
            except Exception as exc:
                results_rows.append(
                    {
                        "SMILES": smi,
                        "error": t(
                            "batch.calculation_error",
                            lang,
                            error_type=type(exc).__name__,
                            error=exc,
                        ),
                    }
                )

        batch_df = batch_xai.build_batch_export_dataframe(
            results_rows,
            smiles_key="SMILES",
            include_long_text=False,
            lang=lang,
        )
        batch_summary = batch_xai.summarize_batch_explanations(batch_df, lang=lang)

        st.session_state["batch_xai_df"] = batch_df
        st.session_state["batch_xai_summary"] = batch_summary
        st.session_state["batch_keep_mode"] = True
        status.success(t("batch.done", lang))
        st.caption(tx("batch.result_stored", lang))

    if "batch_xai_df" in st.session_state:
        batch_df = st.session_state["batch_xai_df"]
        batch_summary = st.session_state.get("batch_xai_summary") or batch_xai.summarize_batch_explanations(batch_df, lang=lang)

        section_labels = {
            "summary": tx("batch.section.summary", lang),
            "table": tx("batch.section.table", lang),
            "export": tx("batch.section.export", lang),
        }
        section = section_selector(
            tx("nav.batch_section", lang),
            list(section_labels.keys()),
            lambda item: section_labels[item],
            key="batch_section",
        )

        if section == "summary":
            render_batch_explainability_summary(batch_df, batch_summary, lang=lang)
            with st.expander(t("batch.interpretation_title", lang), expanded=False):
                st.markdown(_batch_interpretation_markdown(lang))
        elif section == "table":
            student_df = _batch_student_table(batch_df, lang=lang)
            st.dataframe(student_df, use_container_width=True)
            with st.expander(t("batch.show_all_columns", lang), expanded=False):
                display_columns = batch_xai.select_batch_display_columns(batch_df)
                st.dataframe(batch_df[display_columns] if display_columns else batch_df, use_container_width=True)
        elif section == "export":
            sheets = batch_xai.build_batch_excel_sheets(batch_df, batch_summary)
            interpretation_md = _batch_interpretation_markdown(lang)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                for sheet_name, sheet_df in sheets.items():
                    sheet_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button(
                    t("batch.download_excel", lang),
                    buffer.getvalue(),
                    "screening_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with c2:
                st.download_button(
                    t("batch.download_csv", lang),
                    batch_df.to_csv(index=False).encode("utf-8"),
                    "screening_results.csv",
                    mime="text/csv",
                )
            with c3:
                st.download_button(
                    t("batch.download_interpretation", lang),
                    interpretation_md.encode("utf-8"),
                    f"screening_interpretation_{lang}.md",
                    mime="text/markdown",
                )


def main() -> None:
    st.set_page_config(page_title="BioSynth-EDU | ADMET/QSAR", layout="wide")
    init_state()

    with st.sidebar:
        st.image("https://img.icons8.com/fluent/96/000000/molecule.png", width=60)
        lang_label = st.selectbox(t("sidebar.language", "ru"), language_options(), key="lang_label")
        lang = language_label_to_code(lang_label)
        st.title("BioSynth-EDU")
        st.markdown(tx("sidebar.stage", lang))
        st.markdown(t("sidebar.subtitle", lang))
        st.markdown("---")
        st.info(t("sidebar.info", lang))
        developer_mode = st.checkbox(
            tx("sidebar.developer_mode", lang),
            help=tx("sidebar.developer_help", lang),
            key="developer_mode",
        )
        if developer_mode:
            with st.expander("Runtime", expanded=False):
                st.caption(tx("sidebar.model_selection_hint", lang))
                st.code(os.environ.get("BIOSYNTH_MODEL_SELECTION_PATH", "auto: models/v2_experiment/model_selection.json"))

    st.title(t("app.title", lang))
    if st.session_state.get("batch_keep_mode") and _main_mode_fallback() == "batch":
        st.session_state["main_mode"] = "batch"
        st.session_state["batch_keep_mode"] = False

    mode_labels = {
        "single": t("tabs.single", lang),
        "batch": t("tabs.batch", lang),
    }
    mode = section_selector(
        tx("nav.main_mode", lang),
        ["single", "batch"],
        lambda item: mode_labels[item],
        key="main_mode",
        fallback=_main_mode_fallback(),
    )

    if mode == "single":
        render_single_analysis(lang, developer_mode=developer_mode)
    else:
        render_batch_screening(lang)


if __name__ == "__main__":
    main()



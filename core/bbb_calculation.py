"""Runtime ADMET/BBB inference core for BioSynth-EDU.

Stage 7.5 runtime integration summary
-----------------------
1. Gupta BBB score uses the corrected ``p_mwhbn`` term via ``core.gupta_bbb``.
2. pKa, P-gp, Caco-2, CATMoS, optional BBB RF and disabled Clint handling are
   driven by ``models/v2_experiment/model_selection.json`` when available.
3. Runtime loading falls back to legacy ``models/rf_*.joblib`` when no model
   selection artifact exists.
4. Stage 7.4 thresholds are applied at inference time for selected classifiers.
5. CATMoS remains score-only when its target units are not confirmed.

The public functions from the earlier app are preserved:
``predict_local_pka``, ``predict_pgp_efflux``, ``predict_clint``,
``predict_catmos``, ``predict_caco2``, ``generate_admet_visualizations`` and
``analyze_molecule_cns_profile``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Draw import rdMolDraw2D

from core.features import (
    build_bbb_rf_features,
    build_caco2_features,
    build_catmos_features,
    build_clint_features,
    build_pgp_features,
    build_pka_features,
    get_positive_class_probability,
    safe_float,
    validate_feature_vector,
)
from core.gupta_bbb import (
    BBB_GUPTA_THRESHOLD,
    DEFAULT_PKA,
    calculate_gupta_bbb_components,
    calculate_gupta_bbb_score as _calculate_gupta_bbb_score,
    calculate_gupta_bbb_score_legacy as _calculate_gupta_bbb_score_legacy,
)

from core.runtime_models import (
    build_runtime_status_summary,
    get_runtime_context,
    get_runtime_entry,
    get_runtime_load_status_by_short_label,
    get_runtime_model,
    get_runtime_threshold,
    is_runtime_disabled,
)

logger = logging.getLogger(__name__)

# Stage 7.5: runtime context is selection-aware. If a Stage 7.4
# ``model_selection.json`` artifact is available, selected v2 models are loaded
# from that artifact. Otherwise the loader falls back to legacy model paths.
RUNTIME_CONTEXT = get_runtime_context()
MODEL_LOAD_STATUS: Dict[str, Dict[str, Any]] = get_runtime_load_status_by_short_label(RUNTIME_CONTEXT)

pka_model = get_runtime_model("rf_pka_model", RUNTIME_CONTEXT)
pgp_model = get_runtime_model("rf_pgp_model", RUNTIME_CONTEXT)
clint_model = get_runtime_model("rf_clint_model", RUNTIME_CONTEXT)
catmos_model = get_runtime_model("rf_catmos_model", RUNTIME_CONTEXT)
caco2_model = get_runtime_model("rf_caco2_model", RUNTIME_CONTEXT)
bbb_rf_model = get_runtime_model("rf_bbb_model", RUNTIME_CONTEXT)


def _error_string(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _prediction_dict(
    *,
    value: Any = None,
    probability: Optional[float] = None,
    pred_class: Optional[int] = None,
    label: Optional[str] = None,
    status: str,
    source: str,
    error: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "value": value,
        "probability": probability,
        "class": pred_class,
        "label": label,
        "status": status,
        "source": source,
        "error": error,
    }
    if extra:
        data.update(dict(extra))
    return data


def _reshape_for_model(features: np.ndarray, model: Any, feature_kind: str) -> np.ndarray:
    expected_dim = getattr(model, "n_features_in_", None)
    arr = validate_feature_vector(features, expected_dim=expected_dim, name=feature_kind)
    return arr.reshape(1, -1)


def _safe_round(value: Any, digits: int = 2, fallback: Any = "N/A") -> Any:
    try:
        x = float(value)
    except Exception:
        return fallback
    if not np.isfinite(x):
        return fallback
    return round(x, digits)


def _safe_probability(value: Any, default: float = 0.0) -> float:
    x = safe_float(value, default=default)
    return float(max(0.0, min(1.0, x)))


def _model_status_summary(*items: Tuple[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for name, result in items:
        summary[name] = {
            "status": str(result.get("status", "unknown")),
            "source": str(result.get("source", "unknown")),
            "error": result.get("error"),
            "runtime_source": result.get("runtime_source"),
            "selection_status": result.get("selection_status"),
            "runtime_role": result.get("runtime_role"),
            "runtime_threshold": result.get("runtime_threshold"),
            "feature_kind": result.get("feature_kind"),
            "task_type": result.get("task_type"),
        }
    return summary


def _model_errors(statuses: Mapping[str, Mapping[str, Any]]) -> Dict[str, str]:
    errors: Dict[str, str] = {}
    for name, item in statuses.items():
        error = item.get("error")
        if error:
            errors[name] = str(error)
    return errors


def _runtime_entry(legacy_name: str) -> Dict[str, Any]:
    return get_runtime_entry(legacy_name, RUNTIME_CONTEXT)


def _runtime_threshold(legacy_name: str) -> Optional[float]:
    return get_runtime_threshold(legacy_name, RUNTIME_CONTEXT)


def _runtime_disabled(legacy_name: str) -> bool:
    return is_runtime_disabled(legacy_name, RUNTIME_CONTEXT)


def _runtime_extra(legacy_name: str) -> Dict[str, Any]:
    entry = _runtime_entry(legacy_name)
    return {
        "runtime_source": entry.get("source"),
        "selection_status": entry.get("selection_status"),
        "runtime_role": entry.get("runtime_role"),
        "runtime_threshold": entry.get("threshold"),
        "feature_kind": entry.get("feature_kind"),
        "task_type": entry.get("task_type"),
    }


def _class_from_probability_or_model(model: Any, x: np.ndarray, probability: float, legacy_name: str) -> int:
    threshold = _runtime_threshold(legacy_name)
    if threshold is not None:
        return int(float(probability) >= float(threshold))
    return int(model.predict(x)[0])


# ---------------------------------------------------------------------------
# Gupta BBB compatibility wrappers
# ---------------------------------------------------------------------------


def calculate_gupta_bbb_score(descriptors: Mapping[str, Any], pka: Optional[float] = None) -> float:
    """Corrected Gupta BBB score, preserving the old import path."""
    return _calculate_gupta_bbb_score(descriptors, pka=pka)


def calculate_gupta_bbb_score_legacy(descriptors: Mapping[str, Any], pka: Optional[float] = None) -> float:
    """Legacy Gupta score that used raw MWHBN. Kept for audit comparisons."""
    return _calculate_gupta_bbb_score_legacy(descriptors, pka=pka)


# ---------------------------------------------------------------------------
# Detailed inference helpers
# ---------------------------------------------------------------------------


def predict_local_pka_detailed(smiles: str, mol: Chem.Mol | None = None) -> Dict[str, Any]:
    """Predict pKa with explicit status and safe fallback metadata."""
    if pka_model is None:
        return _prediction_dict(
            value=DEFAULT_PKA,
            label="fallback pKa",
            status="fallback_model_missing",
            source="default",
            error=MODEL_LOAD_STATUS.get("pka", {}).get("error"),
            extra=_runtime_extra("rf_pka_model"),
        )

    if mol is None:
        mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return _prediction_dict(
            value=DEFAULT_PKA,
            label="fallback pKa",
            status="fallback_invalid_mol",
            source="default",
            error="Invalid SMILES",
            extra=_runtime_extra("rf_pka_model"),
        )

    try:
        x = _reshape_for_model(build_pka_features(mol), pka_model, "hybrid_pka")
        predicted_val = pka_model.predict(x)[0]
        value = _safe_round(predicted_val, 2, fallback=DEFAULT_PKA)
        return _prediction_dict(value=value, label="pKa", status="ok", source="model", extra=_runtime_extra("rf_pka_model"))
    except Exception as exc:
        logger.exception("pKa prediction failed; fallback pKa will be used")
        return _prediction_dict(
            value=DEFAULT_PKA,
            label="fallback pKa",
            status="fallback_error",
            source="default",
            error=_error_string(exc),
            extra=_runtime_extra("rf_pka_model"),
        )


def predict_local_pka(smiles: str, mol: Chem.Mol | None = None) -> float:
    """Backward-compatible pKa value only."""
    return float(predict_local_pka_detailed(smiles, mol).get("value", DEFAULT_PKA))


def predict_pgp_efflux_detailed(mol: Chem.Mol) -> Dict[str, Any]:
    """Predict P-gp substrate class with explicit status."""
    if pgp_model is None:
        return _prediction_dict(
            probability=0.5,
            pred_class=0,
            label="Не субстрат",
            status="fallback_model_missing",
            source="default",
            error=MODEL_LOAD_STATUS.get("pgp", {}).get("error"),
            extra=_runtime_extra("rf_pgp_model"),
        )

    try:
        x = _reshape_for_model(build_pgp_features(mol), pgp_model, "hybrid_pgp")
        pred_prob = get_positive_class_probability(pgp_model, x, positive_label=1)
        prob = round(_safe_probability(pred_prob, 0.5), 3)
        pred_class = _class_from_probability_or_model(pgp_model, x, prob, "rf_pgp_model")
        label = "Субстрат (Вымывается)" if pred_class == 1 else "Не субстрат"
        return _prediction_dict(
            probability=prob,
            pred_class=pred_class,
            label=label,
            status="ok",
            source="model",
            extra=_runtime_extra("rf_pgp_model"),
        )
    except Exception as exc:
        logger.exception("P-gp prediction failed; neutral fallback will be used")
        return _prediction_dict(
            probability=0.5,
            pred_class=0,
            label="Не субстрат",
            status="fallback_error",
            source="default",
            error=_error_string(exc),
            extra=_runtime_extra("rf_pgp_model"),
        )


def predict_pgp_efflux(mol: Chem.Mol) -> Tuple[int, float]:
    """Backward-compatible P-gp result: class, probability."""
    result = predict_pgp_efflux_detailed(mol)
    return int(result.get("class") or 0), round(float(result.get("probability") or 0.5), 3)


def predict_clint_detailed(mol: Chem.Mol) -> Dict[str, Any]:
    """Predict metabolic clearance risk as a classifier output."""
    if _runtime_disabled("rf_clint_model"):
        return _prediction_dict(
            probability=0.0,
            pred_class=None,
            label="N/A",
            status="disabled_by_selection",
            source="selection",
            error=None,
            extra=_runtime_extra("rf_clint_model"),
        )

    if clint_model is None:
        return _prediction_dict(
            probability=0.0,
            pred_class=None,
            label="N/A",
            status="model_missing",
            source="unavailable",
            error=MODEL_LOAD_STATUS.get("clint", {}).get("error"),
            extra=_runtime_extra("rf_clint_model"),
        )

    try:
        x = _reshape_for_model(build_clint_features(mol), clint_model, "morgan_2048")
        prob = get_positive_class_probability(clint_model, x, positive_label=1)
        prob_value = round(_safe_probability(prob), 2)
        pred_class = _class_from_probability_or_model(clint_model, x, prob_value, "rf_clint_model")
        label = "Высокий риск" if pred_class == 1 else "Стабильное"
        return _prediction_dict(
            probability=prob_value,
            pred_class=pred_class,
            label=label,
            status="ok",
            source="model",
            extra=_runtime_extra("rf_clint_model"),
        )
    except Exception as exc:
        logger.exception("Clint prediction failed")
        return _prediction_dict(
            probability=0.0,
            pred_class=None,
            label="Ошибка",
            status="error",
            source="model",
            error=_error_string(exc),
            extra=_runtime_extra("rf_clint_model"),
        )


def predict_clint(mol: Chem.Mol) -> Tuple[str, float]:
    """Backward-compatible Clint result: status label, probability."""
    result = predict_clint_detailed(mol)
    return str(result.get("label") or "N/A"), round(float(result.get("probability") or 0.0), 2)


def predict_catmos_detailed(mol: Chem.Mol) -> Dict[str, Any]:
    """Predict CATMoS target value with explicit unit caution.

    Stage 7.1 intentionally labels the value as ``CATMoS score`` because the
    training target scale must be confirmed before displaying it as mg/kg.
    """
    if catmos_model is None:
        return _prediction_dict(
            value=None,
            label="N/A",
            status="model_missing",
            source="unavailable",
            error=MODEL_LOAD_STATUS.get("catmos", {}).get("error"),
            extra={**_runtime_extra("rf_catmos_model"), "unit_status": "unknown"},
        )

    try:
        x = _reshape_for_model(build_catmos_features(mol), catmos_model, "morgan_2048")
        pred = catmos_model.predict(x)[0]
        value = _safe_round(pred, 2, fallback=None)
        return _prediction_dict(
            value=value,
            label="CATMoS score",
            status="ok_units_unverified",
            source="model",
            extra={**_runtime_extra("rf_catmos_model"), "unit_status": "requires_validation"},
        )
    except Exception as exc:
        logger.exception("CATMoS prediction failed")
        return _prediction_dict(
            value=None,
            label="Ошибка",
            status="error",
            source="model",
            error=_error_string(exc),
            extra={**_runtime_extra("rf_catmos_model"), "unit_status": "unknown"},
        )


def predict_catmos(mol: Chem.Mol) -> Any:
    """Backward-compatible CATMoS value only."""
    result = predict_catmos_detailed(mol)
    value = result.get("value")
    return value if value is not None else result.get("label", "N/A")


def predict_caco2_detailed(mol: Chem.Mol) -> Dict[str, Any]:
    """Predict Caco-2 LogPapp with explicit status."""
    if caco2_model is None:
        return _prediction_dict(
            value=None,
            label="N/A",
            status="model_missing",
            source="unavailable",
            error=MODEL_LOAD_STATUS.get("caco2", {}).get("error"),
            extra=_runtime_extra("rf_caco2_model"),
        )

    try:
        x = _reshape_for_model(build_caco2_features(mol), caco2_model, "hybrid_caco2")
        log_papp = _safe_round(caco2_model.predict(x)[0], 2, fallback=None)
        if log_papp is None:
            raise ValueError("Caco-2 model returned a non-finite value")
        status_label = "High" if float(log_papp) > -5.0 else ("Medium" if float(log_papp) > -6.0 else "Low")
        return _prediction_dict(value=log_papp, label=status_label, status="ok", source="model", extra=_runtime_extra("rf_caco2_model"))
    except Exception as exc:
        logger.exception("Caco-2 prediction failed")
        return _prediction_dict(
            value=None,
            label="N/A",
            status="error",
            source="model",
            error=_error_string(exc),
            extra=_runtime_extra("rf_caco2_model"),
        )


def predict_caco2(mol: Chem.Mol) -> Tuple[Any, str]:
    """Backward-compatible Caco-2 result: LogPapp value, category."""
    result = predict_caco2_detailed(mol)
    return result.get("value", "N/A"), str(result.get("label") or "N/A")


def predict_bbb_rf_detailed(mol: Chem.Mol) -> Dict[str, Any]:
    """Optional RF BBB model prediction. The current app still uses Gupta by default."""
    if bbb_rf_model is None:
        return _prediction_dict(
            probability=None,
            pred_class=None,
            label="N/A",
            status="model_missing",
            source="unavailable",
            error=MODEL_LOAD_STATUS.get("bbb_rf", {}).get("error"),
            extra=_runtime_extra("rf_bbb_model"),
        )

    try:
        x = _reshape_for_model(build_bbb_rf_features(mol), bbb_rf_model, "morgan_2048")
        prob = get_positive_class_probability(bbb_rf_model, x, positive_label=1)
        prob_value = round(_safe_probability(prob), 3)
        pred_class = _class_from_probability_or_model(bbb_rf_model, x, prob_value, "rf_bbb_model")
        label = "BBB+" if pred_class == 1 else "BBB-"
        return _prediction_dict(
            probability=prob_value,
            pred_class=pred_class,
            label=label,
            status="ok_not_primary",
            source="model",
            extra={**_runtime_extra("rf_bbb_model"), "note": "Current app primary BBB score is corrected Gupta, not RF probability."},
        )
    except Exception as exc:
        logger.exception("BBB RF prediction failed")
        return _prediction_dict(
            probability=None,
            pred_class=None,
            label="Ошибка",
            status="error",
            source="model",
            error=_error_string(exc),
            extra=_runtime_extra("rf_bbb_model"),
        )


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def generate_admet_visualizations(mol: Chem.Mol, output_dir: str = "models/") -> None:
    """Generate the same radar and Gasteiger maps as the earlier app.

    This function remains a UI helper. It is intentionally separated from model
    predictions: if visual generation fails, the molecular inference result is
    still returned by ``analyze_molecule_cns_profile``.
    """
    os.makedirs(output_dir, exist_ok=True)

    mw = safe_float(Descriptors.MolWt(mol))
    logp = safe_float(Descriptors.MolLogP(mol))
    tpsa = safe_float(Descriptors.TPSA(mol))
    rotb = safe_float(Descriptors.NumRotatableBonds(mol))
    h_donors = safe_float(Descriptors.NumHDonors(mol))
    h_acceptors = safe_float(Descriptors.NumHAcceptors(mol))

    scores = [
        mw / 500.0,
        logp / 5.0,
        tpsa / 140.0,
        rotb / 10.0,
        h_donors / 5.0,
        h_acceptors / 10.0,
    ]
    scores = [min(max(x, 0.0) * 1.5, 1.5) for x in scores]
    categories = ["MW", "LogP", "TPSA", "Rot. Bonds", "H-Donors", "H-Acceptors"]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    scores_plot = scores + scores[:1]
    angles_plot = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ideal_zone = [1.0] * 6 + [1.0]
    ax.plot(angles_plot, ideal_zone, linestyle="--", linewidth=1, label="Ideal Range")
    ax.fill(angles_plot, ideal_zone, alpha=0.1)
    ax.plot(angles_plot, scores_plot, linewidth=2, label="Molecule")
    ax.fill(angles_plot, scores_plot, alpha=0.3)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles, categories, size=9)
    plt.yticks([0.5, 1.0, 1.5], ["0.5", "1.0", "1.5"], size=7)
    plt.ylim(0, 1.5)
    plt.title("Bioavailability Radar (Rule of 5)", size=12, y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.savefig(os.path.join(output_dir, "admet_radar.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    weights = []
    for atom in mol.GetAtoms():
        weights.append(safe_float(atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0))

    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
    SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, colorMap="coolwarm", draw2d=drawer, contourLines=4)
    drawer.FinishDrawing()
    with open(os.path.join(output_dir, "atom_weights.png"), "wb") as fh:
        fh.write(drawer.GetDrawingText())


# ---------------------------------------------------------------------------
# Main public pipeline
# ---------------------------------------------------------------------------


def analyze_molecule_cns_profile(smiles: str, descriptors: Mapping[str, Any]) -> Dict[str, Any]:
    """Run corrected BBB/ADMET inference for one molecule.

    The returned dictionary keeps all previous keys used by app.py and adds
    explicit QA/status metadata. The BBB score is the corrected Gupta score.
    """
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return {
            "error": "Некорректная структура SMILES",
            "model_statuses": {},
            "model_errors": {"input": "Invalid SMILES"},
        }

    mw = safe_float(Descriptors.MolWt(mol))
    tpsa = safe_float(Descriptors.TPSA(mol))
    logp = safe_float(Descriptors.MolLogP(mol))
    h_donors = int(Descriptors.NumHDonors(mol))
    h_acceptors = int(Descriptors.NumHAcceptors(mol))
    aro_r = int(descriptors.get("Aro_R", Descriptors.NumAromaticRings(mol)))
    ha = int(descriptors.get("HA", Descriptors.HeavyAtomCount(mol)))

    pka_result = predict_local_pka_detailed(smiles, mol)
    pka = float(pka_result.get("value", DEFAULT_PKA))

    gupta_corrected = calculate_gupta_bbb_components(descriptors, pka=pka, corrected=True)
    gupta_legacy = calculate_gupta_bbb_components(descriptors, pka=pka, corrected=False)
    bbb_score = float(gupta_corrected["score"])
    bbb_status = "Высокая" if bbb_score >= BBB_GUPTA_THRESHOLD else "Низкая"

    pgp_result = predict_pgp_efflux_detailed(mol)
    clint_result = predict_clint_detailed(mol)
    catmos_result = predict_catmos_detailed(mol)
    caco2_result = predict_caco2_detailed(mol)
    bbb_rf_result = predict_bbb_rf_detailed(mol)

    pgp_class = int(pgp_result.get("class") or 0)
    if bbb_score >= BBB_GUPTA_THRESHOLD and pgp_class == 0:
        cns_summary = "Активно в ЦНС (Проникает и остается в мозге)"
    elif bbb_score >= BBB_GUPTA_THRESHOLD and pgp_class == 1:
        cns_summary = "Периферическое действие (Проницаемость ГЭБ высокая, но вымывается белком P-gp)"
    else:
        cns_summary = "Не активно в ЦНС (Низкая пассивная диффузия через ГЭБ)"

    try:
        generate_admet_visualizations(mol)
        vis_status = "Успешно сгенерированы"
    except Exception as exc:
        logger.exception("ADMET visualization generation failed")
        vis_status = f"Ошибка визуализации: {_error_string(exc)}"

    model_statuses = _model_status_summary(
        ("pka", pka_result),
        ("pgp", pgp_result),
        ("clint", clint_result),
        ("catmos", catmos_result),
        ("caco2", caco2_result),
        ("bbb_rf", bbb_rf_result),
    )

    qa_warnings = []
    if pka_result.get("source") == "default":
        qa_warnings.append("pKa model fallback was used; Gupta score depends on fallback pKa.")
    if catmos_result.get("unit_status") == "requires_validation":
        qa_warnings.append("CATMoS target units require validation before displaying as mg/kg.")
    if bbb_rf_result.get("status") in {"model_missing", "error"}:
        qa_warnings.append("RF BBB model is not used as the primary BBB score; corrected Gupta score is primary.")
    if clint_result.get("status") == "disabled_by_selection":
        qa_warnings.append("Clint model is disabled by Stage 7.4 model selection because validation was weak.")

    runtime_summary = build_runtime_status_summary(RUNTIME_CONTEXT)

    return {
        "smiles": smiles,
        "mw": round(float(mw), 2),
        "tpsa": round(float(tpsa), 2),
        "logp": round(float(logp), 2),
        "h_donors": int(h_donors),
        "h_acceptors": int(h_acceptors),
        "aro_r": int(aro_r),
        "ha": int(ha),
        "pKa_predicted": round(float(pka), 2),
        "pka_source": pka_result.get("source"),
        "pka_model_status": pka_result.get("status"),
        "pka_error": pka_result.get("error"),
        "gupta_bbb_score": round(float(bbb_score), 2),
        "gupta_corrected_score": gupta_corrected["score"],
        "gupta_legacy_score": gupta_legacy["score"],
        "gupta_formula_version": gupta_corrected["formula_version"],
        "gupta_components": gupta_corrected,
        "gupta_legacy_components": gupta_legacy,
        "passive_permeability": bbb_status,
        "bbb_rf_probability": bbb_rf_result.get("probability"),
        "bbb_rf_class": bbb_rf_result.get("class"),
        "bbb_rf_model_status": bbb_rf_result.get("status"),
        "pgp_substrate_class": int(pgp_result.get("class") or 0),
        "pgp_probability": round(float(pgp_result.get("probability") or 0.5), 2),
        "pgp_status": pgp_result.get("label") or "Не субстрат",
        "pgp_model_status": pgp_result.get("status"),
        "pgp_error": pgp_result.get("error"),
        "clint_status": clint_result.get("label") or "N/A",
        "clint_probability": round(float(clint_result.get("probability") or 0.0), 2),
        "clint_model_status": clint_result.get("status"),
        "clint_error": clint_result.get("error"),
        "catmos_ld50": catmos_result.get("value") if catmos_result.get("value") is not None else "N/A",
        "catmos_score": catmos_result.get("value") if catmos_result.get("value") is not None else "N/A",
        "catmos_unit_status": catmos_result.get("unit_status"),
        "catmos_model_status": catmos_result.get("status"),
        "catmos_error": catmos_result.get("error"),
        "cns_interpretation": cns_summary,
        "caco2_logpapp": caco2_result.get("value") if caco2_result.get("value") is not None else "N/A",
        "caco2_status": caco2_result.get("label") or "N/A",
        "caco2_model_status": caco2_result.get("status"),
        "caco2_error": caco2_result.get("error"),
        "visualizations": vis_status,
        "model_statuses": model_statuses,
        "model_errors": _model_errors(model_statuses),
        "qa_warnings": qa_warnings,
        "runtime_model_selection": runtime_summary,
        "runtime_model_selection_path": runtime_summary.get("selection_path"),
        "runtime_primary_map": runtime_summary.get("primary_runtime", {}),
    }

"""Adapter layer between the current BioSynth-EDU app result format and explanation_dict.

The current app uses a flat ``results`` dict returned by
``analyze_molecule_cns_profile``. The explainability layer expects a stable
pipeline-like contract with molecule/descriptors/model_outputs sections.
This adapter keeps the existing ML code untouched and creates that contract.
"""

from __future__ import annotations

from typing import Any, Mapping

from rdkit import Chem
from rdkit.Chem import Descriptors

from core.applicability_domain import assess_applicability_domain
from core.i18n import normalize_language


def normalize_gupta_score_to_educational_bbb_score(gupta_score: Any) -> float | None:
    """Map the existing Gupta BBB score to a 0..1 educational score.

    The current app interprets ``gupta_bbb_score >= 3.0`` as high passive BBB
    permeability. The explainability matrix expects a probability-like 0..1
    score. This function maps 3.0 to 0.70, so the old threshold aligns with the
    current ``high BBB-estimate`` threshold.

    This is not a calibrated biological probability. It is only a UI bridge for
    educational interpretation until a real BBB classifier probability is added.
    """
    try:
        score = float(gupta_score)
    except (TypeError, ValueError):
        return None

    if score < 0:
        return 0.0

    if score < 3.0:
        return round(max(0.0, min(0.69, (score / 3.0) * 0.69)), 3)

    # Values above the current high threshold gradually approach 0.99.
    return round(max(0.70, min(0.99, 0.70 + ((score - 3.0) / 2.0) * 0.29)), 3)


def build_pipeline_result_from_current_app(
    input_smiles: str,
    mol: Chem.Mol | None,
    descriptors: Mapping[str, Any] | None,
    results: Mapping[str, Any] | None,
    lang: str = "ru",
) -> dict[str, Any]:
    """Build an explainability-compatible pipeline_result from flat app data."""
    lang = normalize_language(lang)
    descriptors = descriptors or {}
    results = results or {}

    if mol is None:
        mol = Chem.MolFromSmiles(str(input_smiles).strip())

    valid = mol is not None and not bool(results.get("error"))
    canonical_smiles = Chem.MolToSmiles(mol) if mol is not None else None

    rdkit_descriptors = _calculate_extra_rdkit_descriptors(mol) if mol is not None else {}

    gupta_raw = results.get("gupta_bbb_score")
    bbb_educational_score = normalize_gupta_score_to_educational_bbb_score(gupta_raw)

    descriptor_block = {
        "MW": _first_present(results, descriptors, rdkit_descriptors, keys=["mw", "MW", "MolWt"]),
        "LogP": _first_present(results, descriptors, rdkit_descriptors, keys=["logp", "LogP", "MolLogP"]),
        "TPSA": _first_present(results, descriptors, rdkit_descriptors, keys=["tpsa", "TPSA"]),
        "HBD": _first_present(results, descriptors, rdkit_descriptors, keys=["h_donors", "HBD", "NumHDonors"]),
        "HBA": _first_present(results, descriptors, rdkit_descriptors, keys=["h_acceptors", "HBA", "NumHAcceptors"]),
        "RotatableBonds": _first_present(results, descriptors, rdkit_descriptors, keys=["rotatable_bonds", "RotatableBonds", "NumRotatableBonds"]),
        "AromaticRings": _first_present(results, descriptors, rdkit_descriptors, keys=["aro_r", "Aro_R", "AromaticRings"]),
        "FormalCharge": rdkit_descriptors.get("FormalCharge"),
        "GasteigerMin": rdkit_descriptors.get("GasteigerMin"),
        "GasteigerMax": rdkit_descriptors.get("GasteigerMax"),
        "GasteigerAbsMax": rdkit_descriptors.get("GasteigerAbsMax"),
        "pKa_pred": results.get("pKa_predicted"),
        "Pgp_probability": results.get("pgp_probability"),
        "Clint_risk": results.get("clint_status"),
        "CATMoS_LD50": results.get("catmos_ld50"),
    }

    if bbb_educational_score is not None:
        descriptor_block["BBB_probability"] = bbb_educational_score

    descriptor_block = {key: value for key, value in descriptor_block.items() if value is not None}

    ad_results = {
        **results,
        "bbb_classifier_probability": bbb_educational_score,
        "pgp_probability": results.get("pgp_probability"),
        "gupta_bbb_score": gupta_raw,
    }
    applicability = assess_applicability_domain(
        mol=mol,
        descriptors={**rdkit_descriptors, **descriptor_block},
        results=ad_results,
        lang=lang,
    )
    warnings = applicability.get("warnings", [])

    model_statuses = results.get("model_statuses") if isinstance(results.get("model_statuses"), Mapping) else {}
    model_errors = results.get("model_errors") if isinstance(results.get("model_errors"), Mapping) else {}
    qa_warnings = results.get("qa_warnings") if isinstance(results.get("qa_warnings"), list) else []
    runtime_model_selection = results.get("runtime_model_selection") if isinstance(results.get("runtime_model_selection"), Mapping) else {}
    runtime_primary_map = results.get("runtime_primary_map") if isinstance(results.get("runtime_primary_map"), Mapping) else {}

    return {
        "language": lang,
        "molecule": {
            "input_smiles": input_smiles,
            "canonical_smiles": canonical_smiles,
            "valid": valid,
            "rdkit_parse_ok": valid,
            "warnings": warnings,
        },
        "applicability_domain": applicability,
        "descriptors": descriptor_block,
        "model_outputs": {
            "bbb_v1_score": _safe_float(results.get("gupta_legacy_score")),
            "bbb_v2_score": _safe_float(gupta_raw),
            "bbb_classifier_probability": bbb_educational_score,
            "pgp_probability": _safe_float(results.get("pgp_probability")),
            "pka_pred": _safe_float(results.get("pKa_predicted")),
            "clint_risk": results.get("clint_status"),
            "catmos_ld50": results.get("catmos_ld50"),
            "gupta_components": results.get("gupta_components"),
            "gupta_legacy_components": results.get("gupta_legacy_components"),
            "gupta_formula_version": results.get("gupta_formula_version"),
            "bbb_rf_probability": results.get("bbb_rf_probability"),
            "bbb_rf_class": results.get("bbb_rf_class"),
            "model_statuses": model_statuses,
            "model_errors": model_errors,
            "qa_warnings": qa_warnings,
            "runtime_model_selection": runtime_model_selection,
            "runtime_model_selection_path": results.get("runtime_model_selection_path"),
            "runtime_primary_map": runtime_primary_map,
        },
        "model_statuses": model_statuses,
        "model_errors": model_errors,
        "qa_warnings": qa_warnings,
        "runtime_model_selection": runtime_model_selection,
        "runtime_model_selection_path": results.get("runtime_model_selection_path"),
        "runtime_primary_map": runtime_primary_map,
        "raw_current_app_results": dict(results),
        "notes": {
            "bbb_score_mapping": (
                "gupta_bbb_score is stored as bbb_v2_score. "
                "bbb_classifier_probability is an educational normalized score, not a calibrated probability."
            )
        },
    }


def _calculate_extra_rdkit_descriptors(mol: Chem.Mol) -> dict[str, Any]:
    values: dict[str, Any] = {
        "MW": round(float(Descriptors.MolWt(mol)), 2),
        "LogP": round(float(Descriptors.MolLogP(mol)), 2),
        "TPSA": round(float(Descriptors.TPSA(mol)), 2),
        "HBD": int(Descriptors.NumHDonors(mol)),
        "HBA": int(Descriptors.NumHAcceptors(mol)),
        "RotatableBonds": int(Descriptors.NumRotatableBonds(mol)),
        "AromaticRings": int(Descriptors.NumAromaticRings(mol)),
        "FormalCharge": int(Chem.GetFormalCharge(mol)),
        "HeavyAtomCount": int(Descriptors.HeavyAtomCount(mol)),
    }

    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            if atom.HasProp("_GasteigerCharge"):
                raw = atom.GetProp("_GasteigerCharge")
                if raw not in {"nan", "-nan", ""}:
                    charges.append(float(raw))
        if charges:
            values["GasteigerMin"] = round(float(min(charges)), 3)
            values["GasteigerMax"] = round(float(max(charges)), 3)
            values["GasteigerAbsMax"] = round(float(max(abs(x) for x in charges)), 3)
    except Exception:
        values["GasteigerMin"] = None
        values["GasteigerMax"] = None
        values["GasteigerAbsMax"] = None

    return values


def _build_molecule_warnings(
    mol: Chem.Mol | None,
    input_smiles: str,
    results: Mapping[str, Any],
    rdkit_descriptors: Mapping[str, Any],
) -> list[dict[str, str]]:
    if mol is None or results.get("error"):
        return [
            {
                "code": "invalid_smiles",
                "severity": "error",
                "message": "SMILES \u043d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u043a\u043e\u0440\u0440\u0435\u043a\u0442\u043d\u043e \u0440\u0430\u0441\u043f\u043e\u0437\u043d\u0430\u0442\u044c.",
            }
        ]

    warnings: list[dict[str, str]] = []

    if "." in str(input_smiles):
        warnings.append(
            {
                "code": "multiple_fragments",
                "severity": "warning",
                "message": "SMILES \u0441\u043e\u0434\u0435\u0440\u0436\u0438\u0442 \u043d\u0435\u0441\u043a\u043e\u043b\u044c\u043a\u043e \u0444\u0440\u0430\u0433\u043c\u0435\u043d\u0442\u043e\u0432; \u044d\u0442\u043e \u043c\u043e\u0436\u0435\u0442 \u0443\u043a\u0430\u0437\u044b\u0432\u0430\u0442\u044c \u043d\u0430 \u0441\u043e\u043b\u044c \u0438\u043b\u0438 \u0441\u043c\u0435\u0441\u044c.",
            }
        )

    has_carbon = any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
    if not has_carbon:
        warnings.append(
            {
                "code": "no_carbon",
                "severity": "warning",
                "message": "\u041c\u043e\u043b\u0435\u043a\u0443\u043b\u0430 \u043d\u0435 \u0441\u043e\u0434\u0435\u0440\u0436\u0438\u0442 \u0443\u0433\u043b\u0435\u0440\u043e\u0434; \u0434\u043b\u044f \u043d\u0435\u043e\u0440\u0433\u0430\u043d\u0438\u0447\u0435\u0441\u043a\u0438\u0445 \u0441\u0442\u0440\u0443\u043a\u0442\u0443\u0440 ADMET-\u043c\u043e\u0434\u0435\u043b\u0438 \u043c\u043e\u0433\u0443\u0442 \u0431\u044b\u0442\u044c \u043d\u0435\u043d\u0430\u0434\u0451\u0436\u043d\u044b.",
            }
        )

    mw = _safe_float(rdkit_descriptors.get("MW"))
    tpsa = _safe_float(rdkit_descriptors.get("TPSA"))
    hbd = _safe_float(rdkit_descriptors.get("HBD"))
    hba = _safe_float(rdkit_descriptors.get("HBA"))
    formal_charge = _safe_float(rdkit_descriptors.get("FormalCharge"))

    if mw is not None and mw > 700:
        warnings.append(
            {
                "code": "very_high_mw",
                "severity": "warning",
                "message": "\u041e\u0447\u0435\u043d\u044c \u0431\u043e\u043b\u044c\u0448\u0430\u044f \u043c\u043e\u043b\u0435\u043a\u0443\u043b\u044f\u0440\u043d\u0430\u044f \u043c\u0430\u0441\u0441\u0430: \u0432\u043e\u0437\u043c\u043e\u0436\u0435\u043d \u0432\u044b\u0445\u043e\u0434 \u0437\u0430 \u0434\u043e\u043c\u0435\u043d \u043c\u043e\u0434\u0435\u043b\u0438.",
            }
        )

    if tpsa is not None and tpsa > 140:
        warnings.append(
            {
                "code": "very_high_tpsa",
                "severity": "warning",
                "message": "\u041e\u0447\u0435\u043d\u044c \u0432\u044b\u0441\u043e\u043a\u0430\u044f TPSA: \u0432\u043e\u0437\u043c\u043e\u0436\u0435\u043d \u0432\u044b\u0445\u043e\u0434 \u0437\u0430 \u0442\u0438\u043f\u0438\u0447\u043d\u044b\u0439 BBB-\u0434\u043e\u043c\u0435\u043d, \u043e\u0441\u043e\u0431\u0435\u043d\u043d\u043e \u0434\u043b\u044f \u0433\u043b\u0438\u043a\u043e\u0437\u0438\u0434\u043e\u0432 \u0438 \u043f\u043e\u043b\u0438\u0444\u0435\u043d\u043e\u043b\u043e\u0432.",
            }
        )

    if hbd is not None and hbd >= 6:
        warnings.append(
            {
                "code": "many_hbd",
                "severity": "warning",
                "message": "\u0412\u044b\u0441\u043e\u043a\u043e\u0435 \u0447\u0438\u0441\u043b\u043e HBD: \u043f\u0430\u0441\u0441\u0438\u0432\u043d\u0430\u044f BBB-\u0434\u0438\u0444\u0444\u0443\u0437\u0438\u044f \u043c\u043e\u0436\u0435\u0442 \u0431\u044b\u0442\u044c \u0441\u0438\u043b\u044c\u043d\u043e \u043e\u0433\u0440\u0430\u043d\u0438\u0447\u0435\u043d\u0430.",
            }
        )

    if hba is not None and hba >= 12:
        warnings.append(
            {
                "code": "many_hba",
                "severity": "warning",
                "message": "\u0412\u044b\u0441\u043e\u043a\u043e\u0435 \u0447\u0438\u0441\u043b\u043e HBA: \u043f\u0440\u043e\u0444\u0438\u043b\u044c \u043c\u043e\u0436\u0435\u0442 \u0431\u044b\u0442\u044c \u0441\u043b\u0438\u0448\u043a\u043e\u043c \u043f\u043e\u043b\u044f\u0440\u043d\u044b\u043c \u0434\u043b\u044f BBB.",
            }
        )

    if formal_charge is not None and abs(formal_charge) >= 1:
        warnings.append(
            {
                "code": "formal_charge",
                "severity": "warning",
                "message": "\u0424\u043e\u0440\u043c\u0430\u043b\u044c\u043d\u044b\u0439 \u0437\u0430\u0440\u044f\u0434 \u043e\u0442\u043b\u0438\u0447\u0435\u043d \u043e\u0442 0; \u043f\u0430\u0441\u0441\u0438\u0432\u043d\u0443\u044e BBB-\u0438\u043d\u0442\u0435\u0440\u043f\u0440\u0435\u0442\u0430\u0446\u0438\u044e \u043d\u0443\u0436\u043d\u043e \u0442\u0440\u0430\u043a\u0442\u043e\u0432\u0430\u0442\u044c \u043e\u0441\u0442\u043e\u0440\u043e\u0436\u043d\u043e.",
            }
        )

    return warnings


def _first_present(*sources: Mapping[str, Any], keys: list[str]) -> Any:
    for key in keys:
        for source in sources:
            if key in source and source[key] is not None:
                return source[key]
    return None


def _safe_float(value: Any) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None

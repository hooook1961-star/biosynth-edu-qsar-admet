"""Applicability-domain heuristics for BioSynth-EDU.

The module is intentionally rule-based and educational. It does not claim to be
a statistical applicability-domain model; it exposes transparent flags that can
be shown to students and reviewers.
"""

from __future__ import annotations

from typing import Any, Mapping

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except Exception:  # pragma: no cover - allows import in environments without RDKit
    Chem = None
    Descriptors = None

from core.i18n import normalize_language, warning_message, applicability_message

WarningDict = dict[str, Any]


def assess_applicability_domain(
    mol: Any | None = None,
    descriptors: Mapping[str, Any] | None = None,
    results: Mapping[str, Any] | None = None,
    *,
    input_smiles: str | None = None,
    lang: str = "ru",
) -> dict[str, Any]:
    """Return an educational applicability-domain assessment.

    Parameters
    ----------
    mol:
        Optional RDKit molecule. If unavailable, descriptor-only checks are used.
    descriptors:
        RDKit/current-app descriptor dictionary.
    results:
        Optional model results. Used to detect BBB/P-gp conflicts and errors.
    lang:
        ``ru``, ``kk`` or ``en``. Internal flag codes remain stable.
    """
    lang = normalize_language(lang)
    descriptors = dict(descriptors or {})
    results = dict(results or {})

    flags: list[str] = []
    severities: dict[str, str] = {}

    if mol is None or results.get("error"):
        flags.append("invalid_smiles")
        severities["invalid_smiles"] = "error"
        return _build_result(flags, severities, level="outside", lang=lang)

    if _fragment_count(mol) > 1:
        flags.append("multiple_fragments")
        severities["multiple_fragments"] = "warning"
        flags.append("salt_or_mixture")
        severities["salt_or_mixture"] = "warning"

    if _carbon_count(mol) == 0:
        flags.append("no_carbon_or_inorganic")
        severities["no_carbon_or_inorganic"] = "warning"

    mw = _first_float(descriptors, ["MW", "mw", "MolWt", "molecular_weight"])
    logp = _first_float(descriptors, ["LogP", "logp", "MolLogP"])
    tpsa = _first_float(descriptors, ["TPSA", "tpsa"])
    hbd = _first_float(descriptors, ["HBD", "h_donors", "NumHDonors"])
    hba = _first_float(descriptors, ["HBA", "h_acceptors", "NumHAcceptors"])
    formal_charge = _first_float(descriptors, ["FormalCharge", "formal_charge"])

    if mw is None and Descriptors is not None:
        mw = float(Descriptors.MolWt(mol))
    if logp is None and Descriptors is not None:
        logp = float(Descriptors.MolLogP(mol))
    if tpsa is None and Descriptors is not None:
        tpsa = float(Descriptors.TPSA(mol))
    if hbd is None and Descriptors is not None:
        hbd = float(Descriptors.NumHDonors(mol))
    if hba is None and Descriptors is not None:
        hba = float(Descriptors.NumHAcceptors(mol))
    if formal_charge is None and Chem is not None:
        formal_charge = float(Chem.GetFormalCharge(mol))

    if mw is not None and mw > 700:
        flags.append("very_large_molecule")
        severities["very_large_molecule"] = "warning"
    if tpsa is not None and tpsa > 140:
        flags.append("very_high_tpsa")
        severities["very_high_tpsa"] = "warning"
    if hbd is not None and hbd >= 6:
        flags.append("many_hbd")
        severities["many_hbd"] = "warning"
    if hba is not None and hba >= 12:
        flags.append("many_hba")
        severities["many_hba"] = "warning"
    if formal_charge is not None and abs(formal_charge) >= 2:
        flags.append("high_abs_formal_charge")
        severities["high_abs_formal_charge"] = "warning"
    elif formal_charge is not None and abs(formal_charge) >= 1:
        flags.append("formal_charge_nonzero")
        severities["formal_charge_nonzero"] = "warning"
    if logp is not None and logp < -1.0:
        flags.append("extreme_logp_low")
        severities["extreme_logp_low"] = "warning"
    if logp is not None and logp > 6.0:
        flags.append("extreme_logp_high")
        severities["extreme_logp_high"] = "warning"

    if detect_polyphenol_like_structure(mol):
        flags.append("polyphenol_like")
        severities["polyphenol_like"] = "warning"
    if detect_glycoside_like_structure(mol):
        flags.append("glycoside_like")
        severities["glycoside_like"] = "warning"

    bbb = _first_float(results, ["bbb_classifier_probability", "BBB_probability", "bbb_probability"])
    pgp = _first_float(results, ["pgp_probability", "Pgp_probability"])
    # Current-app rows may have raw Gupta score instead of normalised score.
    gupta = _first_float(results, ["gupta_bbb_score", "bbb_v2_score"])
    if bbb is None and gupta is not None:
        bbb = 0.70 if gupta >= 3.0 else 0.35
    if bbb is not None and pgp is not None and bbb >= 0.70 and pgp >= 0.65:
        flags.append("bbb_pgp_conflict")
        severities["bbb_pgp_conflict"] = "info"

    flags = _dedupe(flags)
    if any(severities.get(flag) == "error" for flag in flags):
        level = "outside"
    elif any(flag in {"very_large_molecule", "no_carbon_or_inorganic"} for flag in flags):
        level = "outside"
    elif flags:
        level = "caution"
    else:
        level = "inside"
    return _build_result(flags, severities, level=level, lang=lang)


def detect_multiple_fragments(mol: Any) -> bool:
    return _fragment_count(mol) > 1


def detect_salt_or_mixture(mol: Any) -> bool:
    return detect_multiple_fragments(mol)


def detect_inorganic_or_no_carbon(mol: Any) -> bool:
    return _carbon_count(mol) == 0


def detect_polyphenol_like_structure(mol: Any) -> bool:
    if Chem is None or mol is None:
        return False
    # Phenolic OH: aromatic carbon attached to neutral hydroxyl oxygen.
    patt = Chem.MolFromSmarts("[c][OX2H]")
    if patt is None:
        return False
    return len(mol.GetSubstructMatches(patt)) >= 3


def detect_glycoside_like_structure(mol: Any) -> bool:
    if Chem is None or mol is None:
        return False
    # Educational heuristic: many oxygens plus a saturated O-containing ring.
    oxygen_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
    if oxygen_count < 5:
        return False
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if 5 <= len(ring) <= 6:
            ring_atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
            if any(atom.GetAtomicNum() == 8 for atom in ring_atoms) and not all(atom.GetIsAromatic() for atom in ring_atoms):
                return True
    return False


def generate_applicability_warnings(flags: list[str], *, lang: str = "ru", severities: Mapping[str, str] | None = None) -> list[WarningDict]:
    lang = normalize_language(lang)
    severities = dict(severities or {})
    warnings = []
    for flag in _dedupe(flags):
        severity = severities.get(flag, "warning")
        warnings.append(
            {
                "code": flag,
                "severity": severity,
                "message": warning_message(flag, lang),
                "student_message": warning_message(flag, lang),
                "recommendation": _recommendation(lang),
            }
        )
    return warnings


def _build_result(flags: list[str], severities: Mapping[str, str], *, level: str, lang: str) -> dict[str, Any]:
    warnings = generate_applicability_warnings(flags, lang=lang, severities=severities)
    return {
        "level": level,
        "score": _score_from_level(level, flags),
        "flags": _dedupe(flags),
        "reasons": [warning["message"] for warning in warnings],
        "warnings": warnings,
        "student_message": applicability_message("invalid" if "invalid_smiles" in flags else level, lang),
    }


def _score_from_level(level: str, flags: list[str]) -> float:
    if level == "inside":
        return 1.0
    if level == "outside":
        return 0.2
    return max(0.35, 0.85 - 0.08 * len(flags))


def _fragment_count(mol: Any) -> int:
    if Chem is None or mol is None:
        return 0
    try:
        return len(Chem.GetMolFrags(mol))
    except Exception:
        return 0


def _carbon_count(mol: Any) -> int:
    if mol is None:
        return 0
    try:
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    except Exception:
        return 0


def _first_float(mapping: Mapping[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key in mapping:
            value = _safe_float(mapping.get(key))
            if value is not None:
                return value
    # case-insensitive fallback
    lower = {str(k).lower(): v for k, v in mapping.items()}
    for key in keys:
        value = _safe_float(lower.get(str(key).lower()))
        if value is not None:
            return value
    return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        number = float(value)
        if number != number:
            return None
        return number
    except (TypeError, ValueError):
        return None


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _recommendation(lang: str) -> str:
    return {
        "ru": "Интерпретируйте прогноз как учебную гипотезу и проверьте структуру/экспериментальные данные.",
        "kk": "Болжамды оқу гипотезасы ретінде қарастырып, құрылымды және эксперименттік деректерді тексеріңіз.",
        "en": "Treat the prediction as an educational hypothesis and verify the structure or experimental evidence.",
    }[lang]

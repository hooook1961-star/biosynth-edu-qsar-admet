"""Auditable Gupta-style BBB score helpers for BioSynth-EDU.

Stage 7.0 fixes the consistency issue where p_mwhbn was calculated but raw MWHBN
was used in the final score.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from rdkit import Chem
from rdkit.Chem import Lipinski
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors

from .features import safe_float

DEFAULT_PKA = 8.81
BBB_GUPTA_THRESHOLD = 3.0


def calculate_bbb_descriptors_precise(smiles: str) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mw = float(rdMolDescriptors.CalcExactMolWt(mol, onlyHeavy=False))
    nhba = int(rdMolDescriptors.CalcNumHBA(mol))
    nhbd = int(rdMolDescriptors.CalcNumHBD(mol))
    hbn = int(nhba + nhbd)
    mwhbn = float(hbn / (mw ** 0.5)) if mw > 0 else 0.0
    return {
        "SMILES": smiles,
        "MW": mw,
        "MW_display": round(mw, 2),
        "HBA": nhba,
        "HBD": nhbd,
        "HBN": hbn,
        "MWHBN": mwhbn,
        "MWHBN_display": round(mwhbn, 2),
        "HA": int(Lipinski.HeavyAtomCount(mol)),
        "Aro_R": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
    }


def _get_descriptor_value(descriptors: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in descriptors and descriptors[key] is not None:
            return safe_float(descriptors[key], default=default)
    return float(default)


def _resolve_mwhbn(descriptors: Mapping[str, Any]) -> float:
    if "HBN" in descriptors and "MW" in descriptors:
        mw = _get_descriptor_value(descriptors, "MW")
        hbn = _get_descriptor_value(descriptors, "HBN")
        if mw > 0:
            return float(hbn / (mw ** 0.5))
    return _get_descriptor_value(descriptors, "MWHBN")


def gupta_probability_components(aro_r: int, ha: float, mwhbn: float, tpsa: float, pka: float) -> Dict[str, float]:
    p_aro_r = {0: 0.336376, 1: 0.816016, 2: 1.0, 3: 0.691115, 4: 0.199399}.get(int(aro_r), 0.0)
    p_ha = (0.0000443 * (ha ** 3) - 0.004556 * (ha ** 2) + 0.12775 * ha - 0.463) / 0.624231 if 5 < ha <= 45 else 0.0
    p_mwhbn = (26.733 * (mwhbn ** 3) - 31.495 * (mwhbn ** 2) + 9.5202 * mwhbn - 0.1358) / 0.72258 if 0.05 < mwhbn <= 0.45 else 0.0
    p_tpsa = (-0.0067 * tpsa + 0.9598) / 0.9598 if 0 < tpsa <= 120 else 0.0
    p_pka = (0.00045068 * (pka ** 4) - 0.016331 * (pka ** 3) + 0.18618 * (pka ** 2) - 0.71043 * pka + 0.8579) / 0.597488 if 3 < pka <= 11 else 0.0
    return {"p_aro_r": float(p_aro_r), "p_ha": float(p_ha), "p_mwhbn": float(p_mwhbn), "p_tpsa": float(p_tpsa), "p_pka": float(p_pka)}


def calculate_gupta_bbb_components(descriptors: Mapping[str, Any], pka: Optional[float] = None, corrected: bool = True) -> Dict[str, Any]:
    aro_r = int(_get_descriptor_value(descriptors, "Aro_R", "AromaticRings", default=0.0))
    ha = _get_descriptor_value(descriptors, "HA", "HeavyAtomCount", default=0.0)
    tpsa = _get_descriptor_value(descriptors, "TPSA", default=0.0)
    mwhbn = _resolve_mwhbn(descriptors)
    pka_used = safe_float(DEFAULT_PKA if pka is None else pka, default=DEFAULT_PKA)
    components = gupta_probability_components(aro_r=aro_r, ha=ha, mwhbn=mwhbn, tpsa=tpsa, pka=pka_used)
    mwhbn_term = components["p_mwhbn"] if corrected else mwhbn
    score = components["p_aro_r"] + components["p_ha"] + 1.5 * mwhbn_term + 2.0 * components["p_tpsa"] + 0.5 * components["p_pka"]
    return {
        "score": round(float(score), 2),
        "score_raw": float(score),
        "status": "Высокая" if score >= BBB_GUPTA_THRESHOLD else "Низкая",
        "threshold": BBB_GUPTA_THRESHOLD,
        "corrected": bool(corrected),
        "formula_version": "gupta_fixed_v2_use_p_mwhbn" if corrected else "gupta_legacy_raw_mwhbn",
        "pka_used": float(pka_used),
        "aro_r": int(aro_r),
        "ha": float(ha),
        "tpsa": float(tpsa),
        "mwhbn_raw": float(mwhbn),
        **components,
        "formula_note": "corrected: uses p_mwhbn" if corrected else "legacy: uses raw MWHBN",
    }


def calculate_gupta_bbb_score(descriptors: Mapping[str, Any], pka: Optional[float] = None) -> float:
    return float(calculate_gupta_bbb_components(descriptors, pka=pka, corrected=True)["score"])


def calculate_gupta_bbb_score_legacy(descriptors: Mapping[str, Any], pka: Optional[float] = None) -> float:
    return float(calculate_gupta_bbb_components(descriptors, pka=pka, corrected=False)["score"])


def compare_gupta_legacy_corrected(descriptors: Mapping[str, Any], pka: Optional[float] = None) -> Dict[str, Any]:
    legacy = calculate_gupta_bbb_components(descriptors, pka=pka, corrected=False)
    corrected = calculate_gupta_bbb_components(descriptors, pka=pka, corrected=True)
    delta = corrected["score_raw"] - legacy["score_raw"]
    return {
        "legacy_score": legacy["score"],
        "corrected_score": corrected["score"],
        "delta": round(float(delta), 4),
        "legacy": legacy,
        "corrected": corrected,
        "message": "Corrected Gupta score uses p_mwhbn instead of raw MWHBN.",
    }


def compare_gupta_legacy_vs_fixed(descriptors: Mapping[str, Any], pka: Optional[float] = None) -> Dict[str, Any]:
    comparison = compare_gupta_legacy_corrected(descriptors, pka=pka)
    delta = float(comparison["delta"])
    return {
        "legacy_score_raw_mwhbn": comparison["legacy_score"],
        "fixed_score": comparison["corrected_score"],
        "delta_fixed_minus_legacy": delta,
        "severity": "major" if abs(delta) >= 1.0 else ("moderate" if abs(delta) >= 0.5 else "minor"),
        "components": comparison["corrected"],
        "message": comparison["message"],
    }

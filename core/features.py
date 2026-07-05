"""Shared feature builders and numerical safeguards for BioSynth-EDU models.

Training scripts and inference code use the same vector order and the same
NaN/inf handling.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit.Chem import rdFingerprintGenerator

DEFAULT_MORGAN_RADIUS = 2
DEFAULT_MORGAN_BITS = 2048

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=DEFAULT_MORGAN_RADIUS,
    fpSize=DEFAULT_MORGAN_BITS,
)


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    length: int
    blocks: Tuple[str, ...]
    description: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


FEATURE_SCHEMAS: Dict[str, FeatureSchema] = {
    "morgan_2048": FeatureSchema(
        "morgan_2048",
        2048,
        ("MorganFP radius=2 nBits=2048",),
        "Morgan fingerprint only; used by BBB RF, Clint and CATMoS scripts.",
    ),
    "hybrid_pka": FeatureSchema(
        "hybrid_pka",
        2221,
        (
            "MorganFP radius=2 nBits=2048",
            "MACCS keys including bit 0; RDKit length usually 167",
            "MW, TPSA, HBD, HBA, GasteigerMax, GasteigerMin",
        ),
        "Hybrid pKa vector used by train_pka_sdf.py.",
    ),
    "hybrid_pgp": FeatureSchema(
        "hybrid_pgp",
        2222,
        (
            "MorganFP radius=2 nBits=2048",
            "MACCS keys including bit 0; RDKit length usually 167",
            "MW, TPSA, HBD, HBA, GasteigerMax, GasteigerMin, LogP",
        ),
        "Hybrid P-gp vector used by train_pgp.py.",
    ),
    "hybrid_caco2": FeatureSchema(
        "hybrid_caco2",
        2222,
        (
            "MorganFP radius=2 nBits=2048",
            "MACCS keys including bit 0; RDKit length usually 167",
            "MW, TPSA, HBD, HBA, GasteigerMax, GasteigerMin, LogP",
        ),
        "Hybrid Caco-2 vector used by train_caco2.py.",
    ),
}

SCHEMA_ALIASES: Dict[str, str] = {
    "pka_hybrid": "hybrid_pka",
    "pgp_hybrid": "hybrid_pgp",
    "caco2_hybrid": "hybrid_caco2",
    "bbb_morgan": "morgan_2048",
    "clint_morgan": "morgan_2048",
    "catmos_morgan": "morgan_2048",
}

MODEL_FEATURE_KIND: Dict[str, str] = {
    "rf_bbb_model": "morgan_2048",
    "rf_clint_model": "morgan_2048",
    "rf_catmos_model": "morgan_2048",
    "rf_pka_model": "hybrid_pka",
    "rf_pgp_model": "hybrid_pgp",
    "rf_caco2_model": "hybrid_caco2",
}


def _normalize_schema_name(name: str) -> str:
    return SCHEMA_ALIASES.get(name, name)


def get_feature_schema(name: str) -> FeatureSchema:
    normalized = _normalize_schema_name(name)
    if normalized not in FEATURE_SCHEMAS:
        raise KeyError(f"Unknown feature schema: {name}")
    return FEATURE_SCHEMAS[normalized]


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        x = float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return float(x)


def mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def build_morgan_fp(mol: Chem.Mol, dtype=np.int8) -> np.ndarray:
    return morgan_gen.GetFingerprintAsNumPy(mol).astype(dtype)


def build_maccs_keys(mol: Chem.Mol, drop_bit0: bool = False, dtype=np.int8) -> np.ndarray:
    fp = GenMACCSKeys(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=dtype)
    DataStructs.ConvertToNumpyArray(fp, arr)
    if drop_bit0:
        arr = arr[1:]
    return arr.astype(dtype)


def compute_gasteiger_charges(mol: Chem.Mol) -> List[float]:
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    charges: List[float] = []
    for atom in mol.GetAtoms():
        if atom.HasProp("_GasteigerCharge"):
            charges.append(safe_float(atom.GetProp("_GasteigerCharge"), default=0.0))
    return charges


def gasteiger_charge_summary(mol: Chem.Mol) -> Tuple[float, float]:
    charges = compute_gasteiger_charges(mol)
    if not charges:
        return 0.0, 0.0
    return float(max(charges)), float(min(charges))


def physchem_common(mol: Chem.Mol, include_logp: bool = False) -> np.ndarray:
    max_charge, min_charge = gasteiger_charge_summary(mol)
    values: List[float] = [
        safe_float(Descriptors.MolWt(mol)),
        safe_float(Descriptors.TPSA(mol)),
        safe_float(Descriptors.NumHDonors(mol)),
        safe_float(Descriptors.NumHAcceptors(mol)),
        safe_float(max_charge),
        safe_float(min_charge),
    ]
    if include_logp:
        values.append(safe_float(Descriptors.MolLogP(mol)))
    return np.asarray(values, dtype=np.float32)


def validate_feature_vector(
    features: np.ndarray,
    expected_dim: Optional[int] = None,
    *,
    name: str = "feature vector",
    sanitize: bool = True,
) -> np.ndarray:
    arr = np.asarray(features)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if sanitize:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(arr.astype(np.float64, copy=False)).all():
        raise ValueError(f"{name} contains non-finite values")
    if expected_dim is not None and len(arr) != int(expected_dim):
        raise ValueError(f"{name} has length {len(arr)}, expected {int(expected_dim)}")
    return arr


def build_feature_vector(mol: Chem.Mol, kind: str, *, validate: bool = True) -> np.ndarray:
    kind = _normalize_schema_name(kind)
    if kind not in FEATURE_SCHEMAS:
        raise KeyError(f"Unknown feature schema: {kind}")
    if kind == "morgan_2048":
        features = build_morgan_fp(mol)
    elif kind == "hybrid_pka":
        features = np.concatenate([build_morgan_fp(mol), build_maccs_keys(mol), physchem_common(mol, include_logp=False)])
    elif kind in {"hybrid_pgp", "hybrid_caco2"}:
        features = np.concatenate([build_morgan_fp(mol), build_maccs_keys(mol), physchem_common(mol, include_logp=True)])
    else:  # pragma: no cover
        raise KeyError(kind)
    schema = FEATURE_SCHEMAS[kind]
    return validate_feature_vector(features, expected_dim=schema.length if validate else None, name=schema.name)


def build_feature_vector_from_smiles(smiles: str, kind: str, *, validate: bool = True) -> np.ndarray:
    return build_feature_vector(mol_from_smiles(smiles), kind, validate=validate)


def build_pka_features(mol: Chem.Mol, *, validate: bool = True) -> np.ndarray:
    return build_feature_vector(mol, "hybrid_pka", validate=validate)


def build_pgp_features(mol: Chem.Mol, *, validate: bool = True) -> np.ndarray:
    return build_feature_vector(mol, "hybrid_pgp", validate=validate)


def build_caco2_features(mol: Chem.Mol, *, validate: bool = True) -> np.ndarray:
    return build_feature_vector(mol, "hybrid_caco2", validate=validate)


def build_clint_features(mol: Chem.Mol, *, validate: bool = True) -> np.ndarray:
    return build_feature_vector(mol, "morgan_2048", validate=validate)


def build_catmos_features(mol: Chem.Mol, *, validate: bool = True) -> np.ndarray:
    return build_feature_vector(mol, "morgan_2048", validate=validate)


def build_bbb_rf_features(mol: Chem.Mol, *, validate: bool = True) -> np.ndarray:
    return build_feature_vector(mol, "morgan_2048", validate=validate)


def get_positive_class_probability(model, x: np.ndarray, positive_label=1) -> float:
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model does not expose predict_proba")
    proba = model.predict_proba(np.asarray(x).reshape(1, -1))[0]
    classes = list(getattr(model, "classes_", []))
    if positive_label not in classes:
        raise ValueError(f"Positive label {positive_label!r} not found in model.classes_: {classes!r}")
    return float(proba[classes.index(positive_label)])


def inspect_features_for_smiles(smiles: str) -> Dict[str, object]:
    mol = mol_from_smiles(smiles)
    canonical = Chem.MolToSmiles(mol, canonical=True)
    report: Dict[str, object] = {"input_smiles": smiles, "canonical_smiles": canonical}
    for alias, schema_name in {
        "morgan_2048": "morgan_2048",
        "pka_hybrid": "hybrid_pka",
        "pgp_hybrid": "hybrid_pgp",
        "caco2_hybrid": "hybrid_caco2",
        "clint_morgan": "morgan_2048",
        "catmos_morgan": "morgan_2048",
        "bbb_morgan": "morgan_2048",
    }.items():
        try:
            arr = build_feature_vector(mol, schema_name)
            report[alias] = {"schema": schema_name, "length": int(len(arr)), "finite": bool(np.isfinite(arr.astype(float)).all())}
        except Exception as exc:
            report[alias] = {"error": f"{type(exc).__name__}: {exc}"}
    return report


def feature_debug_summary(smiles: str) -> Dict[str, object]:
    return inspect_features_for_smiles(smiles)


def feature_schema_report_for_smiles(smiles: str) -> Dict[str, object]:
    mol = mol_from_smiles(smiles)
    morgan = build_morgan_fp(mol)
    maccs_keep = build_maccs_keys(mol, drop_bit0=False)
    maccs_drop = build_maccs_keys(mol, drop_bit0=True)
    result: Dict[str, object] = {
        "smiles": smiles,
        "morgan_length": int(len(morgan)),
        "maccs_length_keep_bit0": int(len(maccs_keep)),
        "maccs_length_drop_bit0": int(len(maccs_drop)),
        "schemas": {name: schema.to_dict() for name, schema in FEATURE_SCHEMAS.items()},
    }
    for kind in FEATURE_SCHEMAS:
        try:
            result[f"{kind}_length"] = int(len(build_feature_vector(mol, kind)))
        except Exception as exc:
            result[f"{kind}_error"] = f"{type(exc).__name__}: {exc}"
    return result

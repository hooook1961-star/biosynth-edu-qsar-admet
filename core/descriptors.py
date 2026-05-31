"""RDKit descriptor helpers for BioSynth-EDU.

Stage 7.1 keeps the old public function name ``calculate_bbb_descriptors``
but avoids rounding the MWHBN value before it enters the Gupta polynomial.
Rounded values are provided separately for display.
"""

from __future__ import annotations

from typing import Any, Dict

from rdkit import Chem
from rdkit.Chem import Lipinski
import rdkit.Chem.rdMolDescriptors as Descriptor


def calculate_bbb_descriptors(smiles: str) -> Dict[str, Any]:
    """Validate SMILES and calculate descriptors used by the Gupta BBB score.

    The previous implementation rounded ``MWHBN`` to two decimals before using
    it in a cubic polynomial. Stage 7.1 returns the precise value under
    ``MWHBN`` and keeps display-only rounded values under ``*_display`` keys.
    """
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        raise ValueError(f"Невалидный или некорректный SMILES: {smiles}")

    mw = float(Descriptor.CalcExactMolWt(mol, onlyHeavy=False))
    nhba = int(Descriptor.CalcNumHBA(mol))
    nhbd = int(Descriptor.CalcNumHBD(mol))
    hbn = int(nhba + nhbd)
    mwhbn = float(hbn / (mw ** 0.5)) if mw > 0 else 0.0
    ha = int(Lipinski.HeavyAtomCount(mol))
    aro_r = int(Descriptor.CalcNumAromaticRings(mol))
    tpsa = float(Descriptor.CalcTPSA(mol))

    return {
        "SMILES": smiles,
        "MW": mw,
        "MW_display": round(mw, 2),
        "HBA": nhba,
        "HBD": nhbd,
        "HBN": hbn,
        "MWHBN": mwhbn,
        "MWHBN_display": round(mwhbn, 2),
        "HA": ha,
        "Aro_R": aro_r,
        "TPSA": tpsa,
        "TPSA_display": round(tpsa, 2),
    }

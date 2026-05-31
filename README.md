# BioSynth-EDU QSAR/ADMET

Standalone Streamlit application for the advanced explainable QSAR/ADMET module of BioSynth-EDU.

## Main features

- SMILES-based molecular analysis.
- RDKit descriptors.
- Corrected Gupta BBB score.
- P-gp RF model.
- pKa helper model.
- Caco-2 helper model.
- BBB RF supplementary model.
- Explainable ADMET decision breakdown.
- What-if educational simulator.
- BBB × P-gp matrix.
- Student report export.
- Batch screening.
- RU / KZ / EN interface.

## Cloud-light build

This repository may intentionally exclude `rf_catmos_model_v2.joblib` because the current artifact is very large. CATMoS is disabled in `model_selection.json` for cloud deployment and can be re-enabled in local/full deployments.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

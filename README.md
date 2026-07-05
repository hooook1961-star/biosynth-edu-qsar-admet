# BioSynth-EDU QSAR/ADMET

BioSynth-EDU QSAR/ADMET is a standalone Streamlit application for educational molecular analysis. The application accepts SMILES strings, calculates molecular descriptors, runs local prediction models, and builds a student-facing explanation of ADMET and central nervous system exposure factors.

The project is intended for teaching and research-oriented demonstration. It provides computational estimates, not medical advice and not experimental proof of permeability, toxicity, or efficacy.

## Main Capabilities

- Single-molecule analysis from SMILES.
- Batch screening from text, CSV, or Excel input.
- RDKit descriptor calculation.
- Modified Gupta formula for blood-brain barrier passage assessment.
- Local pKa, P-gp, Caco-2, BBB RF, and CATMoS model support.
- Educational matrix: blood-brain barrier passage x P-gp.
- Rule-based student explanation of positive, negative, and borderline factors.
- What-if simulator for descriptor-level teaching scenarios.
- Student report export in Markdown, HTML, and JSON.
- Interface text in Russian, Kazakh, and English.

## Scientific and Educational Notes

The Gupta indicator in this application is based on Gupta, Lee, Barden & Weaver, Journal of Medicinal Chemistry, 2019. The implementation uses a modified version: when predicted pKa is available, it is used instead of the fixed pKa value 8.81.

The What-if module changes descriptor values only. It does not modify the chemical structure and should be read as an educational simulation.

## Project Structure

- `app.py` - Streamlit entry point and screen orchestration.
- `core/bbb_calculation.py` - main ADMET and CNS prediction pipeline.
- `core/gupta_bbb.py` - Gupta-style blood-brain barrier score calculation.
- `core/features.py` - feature generation for local machine-learning models.
- `core/runtime_models.py` - model loading and runtime model status.
- `core/explainability.py` - educational interpretation and decision logic.
- `core/explainability_adapter.py` - adapter between calculation results and explanation data.
- `core/what_if.py` - descriptor-level educational simulation.
- `core/reporting.py` and `core/report_text.py` - student report generation and report copy.
- `core/i18n.py`, `core/app_text.py`, `core/text_catalog.py`, `core/matrix_text.py`, `core/ml_ui_text.py` - multilingual text catalogs.
- `ui/explainability_components.py` - Streamlit rendering components for explanation tabs.
- `models/v2_experiment/` - model artifacts, validation metadata, feature schemas, and model-selection files.
- `docs/content-map.md` - map of active student-facing text sources.
- `docs/architecture.md` - architecture overview.

## Cloud-Light Build

This repository may intentionally exclude `rf_catmos_model_v2.joblib` because the current artifact is large. CATMoS can remain score-only or disabled depending on `model_selection.json`.

## Local Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

- Predictions are computational estimates and require experimental validation.
- Model outputs depend on available local model artifacts.
- The modified Gupta calculation should be cited as a modified implementation, not as the original formula without changes.
- Applicability warnings are educational heuristics, not a formal statistical domain-of-applicability model.

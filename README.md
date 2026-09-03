# MLPLHIV — Machine learning prediction of depression among people living with HIV, Kadjebi District, Ghana

Analysis code, derived results and publication figures for the manuscript:

> **Comparative machine learning for prediction of depression among people living with HIV in a rural Ghanaian antiretroviral therapy programme: development, internal validation and SHAP-based interpretation.**
> Gmayinaam VU, Oduro G, Dzissem PE, Tettey CO, Anafo AZ, Adedia D.
> *[Journal]*, [year]. DOI: [to be added on acceptance]

Reporting follows **TRIPOD+AI** (Collins et al., *BMJ* 2024;385:e078378).

---

## Study in brief

| | |
|---|---|
| Design | Facility-based cross-sectional study |
| Setting | Mary Theresa Hospital (Papase) and Kadjebi Health Centre, Oti Region, Ghana |
| Period | February 2024 |
| Participants | 264 adults aged ≥18 years receiving antiretroviral therapy |
| Outcome | Probable depression, PHQ-9 ≥ 10 (nine standard items, range 0–27) |
| Predictors | 52 encoded features; 33 after hierarchical collinearity reduction |
| Algorithms | 10 supervised classifiers |
| Internal validation | Stratified 80:20 split + repeated stratified 5-fold CV (×4 repeats) |
| Interpretation | Permutation importance, SHAP (TreeExplainer / LinearExplainer), decision-curve analysis |

**Headline result.** Random Forest achieved cross-validated AUC 0.893 (SD 0.041); held-out accuracy 77.4%, sensitivity 80.0%, specificity 75.8%, κ 0.536, Brier 0.137. The top seven algorithms — including penalised logistic regression — were statistically indistinguishable.

---

## Repository layout

```
code/       analysis scripts, in execution order (see below)
figures/    all publication figures, 300 dpi PNG
results/    derived result tables (CSV/JSON) — aggregate only, no individual records
docs/       reporting checklists and supporting documentation
data/       NOT INCLUDED — see "Data availability" below
```

### Execution order

| Step | Script | Produces |
|---|---|---|
| 1 | `code/run.py` | Feature matrix, train/test splits, SMOTE resampling |
| 2 | `code/train2.py` | Ten tuned models on the full 52-feature set |
| 3 | `code/cluster.py` | Collinearity assessment, Ward clustering, VIF, retained feature set |
| 4 | `code/train_red.py` | Ten tuned models on the reduced 33-feature set |
| 5 | `code/consensus.py` | SHAP values for the top five models, cross-model rank consensus |
| 6 | `code/calib.py` | Out-of-fold calibration for the top five models |
| 7 | `code/predprob.py` | Predicted-probability distributions |
| 8 | `code/depend.py` | SHAP dependence plots across models |
| 9 | `code/figs*.py`, `fig4b.py`, `fig11b.py`, `fig12b.py`, `fig14c.py` | Publication figures |
| 10 | `code/review_checks.py` | Safety and burden analyses reported in the review |

### Environment

Python 3.11. Install with:

```bash
pip install -r requirements.txt
```

Key packages: scikit-learn, xgboost, lightgbm, catboost, imbalanced-learn, shap, statsmodels, pandas, numpy, matplotlib.

All random seeds are fixed (42 for splitting and model fitting, 7 for repeated cross-validation), so results are reproducible given the same input data.

---

## Data availability

**Individual participant data are deliberately not included in this repository.**

The dataset comprises HIV status, depressive symptom scores including responses on suicidal ideation, and detailed sociodemographic information for 264 identifiable-in-principle individuals in two facilities serving a single rural district. Deposit of such data in a third-party code-hosting service is not covered by the consent participants gave, and is not appropriate without explicit ethics committee authorisation.

De-identified individual participant data, the data dictionary and the study instruments are available from the corresponding author (vugmayinaam@uhas.edu.gh) on reasonable request, subject to approval by [NAME OF ETHICS COMMITTEE] and completion of a data transfer agreement.

Aggregate results sufficient to reproduce every table and figure in the manuscript are provided in `results/`.

---

## Ethics

Ethical approval: [NAME OF ETHICS COMMITTEE], approval number [XXX/XX/XXXX]. Administrative clearance was obtained from the Kadjebi District Health Directorate and from both participating facilities. All participants gave written informed consent.

---

## Citation

If you use this code, please cite the manuscript above.

## Licence

Code released under the MIT Licence (see `LICENSE`). The manuscript, figures and derived results remain © the authors.

## Contact

Vincent Uwumboriyhie Gmayinaam — Institute of Health Research, University of Health and Allied Sciences, PMB 31, Ho, Volta Region, Ghana — vugmayinaam@uhas.edu.gh

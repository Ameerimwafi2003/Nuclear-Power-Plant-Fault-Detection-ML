# Nuclear Power Plant Fault Detection Using Machine Learning

**MENG-4040-0NA | Machine Learning course project | November 2026**

**Team:** Mohammed Zaid Quadri · Ameer Imwafi · Ahmed Junaid  
**Contributions:** shared equally across the project.

An **educational research prototype** that classifies numerical measurements from simulated nuclear power plant scenarios as *Normal Operation (0)* or *Fault Scenario (1)*. The project compares Random Forest, XGBoost, and Isolation Forest and provides an interactive [Streamlit](https://streamlit.io/) demonstration.

> **Safety notice:** This system has not been validated for operating nuclear plants. Predictions are educational model outputs, not plant safety indications or engineering advice.

## Features

- Simulated nuclear operating data with 96 numeric sensor inputs and scenario metadata
- Exploratory data analysis and sensor comparison in Jupyter
- Baseline, class-balancing, and anomaly-detection models
- Scenario-based 4-fold tests, feature ablation, and exploratory statistical comparisons
- Interactive scenario/time selection, model inference, and stress-test examples
- Model artifact and all 96 expected input feature names included

## Quick start (Windows PowerShell)

```powershell
# From inside the extracted project directory
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/smoke_test.py
python -m streamlit run app.py
```

If PowerShell blocks activation, you can run `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m streamlit run app.py` instead. Open the local URL Streamlit prints (usually `http://localhost:8501`).

### Notebooks

```powershell
python -m notebook
```

Open `01_Dataset_Exploration.ipynb` for raw-data analysis or `02_Model_Development.ipynb` for the modeling experiments. **The modeling notebook is sequential**: run cells from top to bottom. It reads `prepared_data/npp_prepared.csv` and writes graphs/tables under `results/`.

**For notebook 1, download the original source dataset as described below** and place the extracted folder beneath this project before running its dataset-search cells.

## Dataset and preprocessing

Original dataset: [SFStefenon/NPPmalfunctions](https://github.com/SFStefenon/NPPmalfunctions), generated using PCTRAN simulation. The original dataset is **not** included in this repository; refer to its source for attribution and applicable reuse terms.

To rebuild prepared CSVs from a downloaded dataset:

```powershell
python scripts/prepare_data.py --dataset-root "NPPmalfunctions-main/NPPmalfunctions-main/Malfunctions"
```

The CSVs in `prepared_data/` are included to make the app runnable without downloading the original spreadsheets. We excluded the scenario **13 – Reserved for future** from binary classification. The exploratory dataset drops time-zero readings for fault scenarios, then a revised experimental dataset omits scenario IDs **5, 8, and 14**, whose recorded sensor values were found to match normal-operation values at aligned simulation times.

**Crucial label caveat:** labels represent the source *simulation scenario*. A label of 1 is not independent proof that a fault was physically observable at each time step. We retained the originally labeled scenarios for stress tests and discussion.

## Repository contents

```text
.
├── README.md
├── requirements.txt
├── app.py
├── 01_Dataset_Exploration.ipynb
├── 02_Model_Development.ipynb
├── models/
│   ├── feature_names.json
│   └── npp_xgboost_model.joblib
├── prepared_data/
│   ├── npp_prepared.csv
│   └── npp_revised.csv
├── results/                  # Experiment summaries, graphs, and tables
├── scripts/
│   ├── prepare_data.py
│   └── smoke_test.py
└── docs/
    └── methodology_and_limitations.md
```

## Selected results (exploratory, NOT independent deployment performance)

| Experiment | Result |
|---|---|
| Full exploratory 80/20 split: original Random Forest | 49.78% balanced accuracy |
| Full exploratory 80/20 split: undersampled RF | 92.15% balanced accuracy |
| Full exploratory 80/20 split: weighted XGBoost | 92.15% balanced accuracy |
| Full exploratory 80/20 split: Isolation Forest | 78.97% balanced accuracy |
| Revised chronological validation, 8 XGBoost configurations | 100% balanced accuracy; limited discriminative value |
| 4-fold fault-scenario split (RF) | 45.35% mean balanced accuracy |
| 4-fold fault-scenario split (XGBoost) | 82.87% mean balanced accuracy |
| Feature ablation: top 15 vs all 96, same scenario-CV setup | Both 82.87% mean balanced accuracy |
| Exploratory one-sided Wilcoxon, four fold-pairs | p = 0.0625 (not significant at 0.05) |

**Interpretation:** performance is substantially sensitive to how the data are split. The cross-validation folds reuse the **same later normal-operation segment** (there is only one normal run). They therefore do **not** constitute four fully independent tests. Three labeled fault scenarios are indistinguishable from normal operation with these features. Avoid treating 100% within-simulation validation results or predictions from the demonstration model as generalization estimates.

The `app.py` demonstration model was trained on **all revised records**, and the app also loads the original exploratory `npp_prepared.csv` so the excluded scenarios appear for stress testing. App predictions on training records confirm functionality, **not** independent accuracy. The shown *fault score* is a model score, not a calibrated real-world probability of an accident.

## Reproducibility notes

- Python 3.11 recommended; XGBoost **2.0.3** and Streamlit **1.32.0** match the development screenshots.
- Input feature order is stored in `models/feature_names.json`; the dashboard reads it before prediction.
- Run the two notebooks from the project root (their path handling uses `Path.cwd()`).
- The source model artifact is bundled for convenience and should be loaded **only if you trust this repository** (pickle/joblib deserialization risk).
- The experiment notebook and figure-generation cells were developed interactively and are included in their original form; this package has been statically checked, but a completely fresh-install end-to-end execution has not been independently confirmed.
- For research rigor, independent normal runs, more distinct fault runs, calibration, and a locked final test set are still needed.

## Ethical considerations

Missed faults and false alarms have asymmetric consequences. This prototype is limited to simulated observations and is **not** intended for monitoring or automatically controlling physical equipment. Responsible deployment would need independent data, scenario diversity, validation under drift, operator supervision, and formal nuclear-sector safety assurance.

## References

- Stefenon, S. F. [NPPmalfunctions dataset](https://github.com/SFStefenon/NPPmalfunctions).
- Breiman, L. (2001). [Random Forests](https://doi.org/10.1023/A:1010933404324).
- Chen, T., & Guestrin, C. (2016). [XGBoost](https://doi.org/10.1145/2939672.2939785).
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). [Isolation Forest](https://doi.org/10.1109/ICDM.2008.17).

## Authorship

Mohammed Zaid Quadri, Ameer Imwafi, and Ahmed Junaid contributed equally to the project. This repository is a course-project research demonstration; its contents do not imply approval from a nuclear operator.

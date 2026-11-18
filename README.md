# Predicting Gas Turbine Wear from Sensor Data

This is a machine learning pipeline that predicts the wear of a naval gas turbine's compressor and turbine using 14 sensor readings from a propulsion plant simulator. It compares five regression models and checks them with two kinds of validation, random cross-validation and leave-one-throttle-out.

## Key findings

- On a random 80/20 split, a random forest predicts wear with a test RMSE of 0.00087 (compressor) and 0.00065 (turbine), about 94% and 91% lower than a baseline that always guesses the average.
- Those scores do not hold up on new operating conditions. When a whole throttle setting is held out, the tree-based models do worse than the baseline on average.

| | Compressor RMSE | Turbine RMSE |
| :--- | ---: | ---: |
| Baseline (mean) | 0.01467 | 0.00755 |
| Random forest, random split | 0.00087 | 0.00065 |
| Baseline, unseen throttle (mean of 9 folds) | 0.01472 | 0.00750 |
| Random forest, unseen throttle (mean of 9 folds) | 0.02028 | 0.01072 |

The full analysis, figures, and limitations are in the [write-up](reports/writeup.md).

## Project structure

```
.
|-- run_pipeline.py          # Runs the full pipeline (optionally with validation)
|-- requirements.txt
|-- data/
|   `-- raw/                 # data.csv (used by the pipeline and exploration notebook)
|-- notebooks/
|   |-- 01_explore.ipynb             # Exploratory analysis
|   `-- 02_validation_results.ipynb  # Validation plots from the saved results
|-- reports/
|   |-- figures/
|   |-- cv_held_out_throttle.csv
|   |-- cv_random_folds.csv
|   |-- model_results.csv
|   `-- writeup.md
`-- src/
    |-- clean.py             # Checks for missing values, drops constant columns
    |-- evaluate.py          # Test scores, comparison tables, and figures
    |-- features.py          # 80/20 train/test split
    |-- load_data.py         # Reads the raw data and names the columns
    |-- train.py             # Defines and trains the five models
    `-- validate.py          # 5-fold and leave-one-throttle-out cross-validation
```

## Setup

Python Version 3.14.7
```
python -m venv .venv
source .venv/Scripts/activate       # Windows (Git Bash)
# source .venv/bin/activate         # macOS/Linux
# .venv\Scripts\Activate.ps1        # Windows (PowerShell)
pip install -r requirements.txt
pip install jupyter                 # Needed to run the notebooks

# Download the data first (see the Data section) and save it as data/raw/data.csv
```


## Usage

Run everything from the project root.

Train the models, evaluate them on the test set, and save the results and figures:

```
python run_pipeline.py
```

Run the pipeline with cross-validation (random folds and leave-one-throttle-out):

```
python run_pipeline.py --validate
```

Outputs go to `reports/`, including the results tables as CSV files. The figures are saved in `reports/figures/`.

## Models

All models use default settings with no hyperparameter tuning, and both wear targets are predicted together.

1. Baseline: always predicts the training average
2. Linear regression
3. Decision tree
4. Random forest (100 trees)
5. XGBoost

## Data

The dataset is the Condition Based Maintenance of Naval Propulsion Plants dataset from the UCI Machine Learning Repository (Coraddu et al., 2014). It has 11,934 rows from a simulator of a frigate's gas turbine propulsion plant with 16 sensor columns and 2 targets (compressor and turbine decay state, where 1.00 is healthy and lower is more worn).

UCI lists the dataset under CC BY 4.0. The dataset's own README asks that publications cite the paper below and states that commercial use is prohibited. This project is for educational use only.

The pipeline and exploration notebook read a CSV copy of the dataset from Kaggle (https://www.kaggle.com/datasets/thedevastator/improving-naval-vessel-condition-through-machine/data). Download it and save it as data/raw/data.csv. Its values and row order match UCI's data.txt, and it only adds a header row and a leading index column.

## Limitations

The data is simulated, and the rows form a complete grid, so random splits give optimistic scores. See the write-up for the full list.

## Reference

Coraddu, A., Oneto, L., Ghio, A., Savio, S., Anguita, D., & Figari, M. (2014). Condition Based Maintenance of Naval Propulsion Plants [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K31K

Coraddu, A., Oneto, L., Ghio, A., Savio, S., Anguita, D., & Figari, M. (2016). Machine learning approaches for improving condition-based maintenance of naval propulsion plants. Proceedings of the Institution of Mechanical Engineers, Part M: Journal of Engineering for the Maritime Environment, 230(1), 136-153. https://doi.org/10.1177/1475090214540874

# Predict Quantity Sold

This project predicts the `Quantity_Sold_(kilo)` value for supermarket sales data using a machine learning pipeline.

## Project Files

- `predict-the-quantitysold.py` - Main Python script that loads data, preprocesses it, trains models, evaluates performance, and generates predictions.
- `labeled_data.csv` - Labeled dataset containing features and the target `Quantity_Sold_(kilo)`.
- `unlabeled_data.csv` - Dataset without target values, used for final predictions.
- `pyproject.toml` / `uv.lock` - Project metadata and dependency lock file for `uv`.

## What it does

The script:

1. Loads labeled and unlabeled CSV data.
2. Cleans and preprocesses features.
3. Creates new features like price difference, profit ratio, and loss-adjusted margin.
4. Encodes categorical values.
5. Trains Linear Regression and Random Forest regression models.
6. Evaluates model performance on validation data.
7. Predicts `Quantity_Sold_(kilo)` for the unlabeled dataset and exports `submission1.csv`.

## Setup

### Using `uv`

From the project root:

```bash
cd /Users/emilyjoy/Documents/Predict
uv sync
```

### Alternative: pip

If you are not using `uv`:

```bash
/usr/local/bin/python3 -m pip install -r requirements.txt
```

## Run the script

### Run normally

```bash
cd /Users/emilyjoy/Documents/Predict
/usr/local/bin/python3 predict-the-quantitysold.py
```

### Run with `uv`

```bash
cd /Users/emilyjoy/Documents/Predict
uv run python predict-the-quantitysold.py
```

## Run interactively in VS Code

1. Open `predict-the-quantitysold.py` in VS Code.
2. Use the `# %%` cell markers to run cells in the Interactive Window.
3. Make sure the VS Code kernel/interpreter is the `uv` environment or the Python interpreter that has `pandas`, `numpy`, and `scikit-learn` installed.

## Notes

- Do not try to import the script itself with a package-style name like `import predict-the-quantitysold`. Python module names cannot contain `-`.
- If you want to make this a reusable package, create a proper package folder and update `pyproject.toml` accordingly.

## Output
- `submission.csv` - Generated predictions for the unlabeled data.

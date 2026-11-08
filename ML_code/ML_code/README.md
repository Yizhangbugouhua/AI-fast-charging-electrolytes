# Formulation modelling and candidate screening

This module constructs formulation sequences, generates SMI-TED features, trains an MLP regressor and screens candidate electrolytes.

## Entry points

| Script | Function |
| --- | --- |
| `preprocessing/Unify.py` | Composition-unit conversion helper; exports `Unified_mole_ratio` |
| `preprocessing/replace.py` | Map component abbreviations to SMILES in a prepared six-slot table |
| `preprocessing/smiles_integrate.py` | Generate sequence/condition/target CSV files |
| `encode_fastcharge_embeddings.py` | Encode sequences and save X/y arrays |
| `check_npy.py` | Inspect array shapes, missing values and target distributions |
| `train_mlp.py` | Train/evaluate the MLP and save prediction artefacts |
| `generate_candidates.py` | Sample candidate compositions and construct their sequences |
| `predict_5c.py` | Predict at fixed 25 °C and 5C |
| `predict_default.py` | Predict using row-specific conditions or default 25 °C and 4C |

Run commands below from the repository root. Each stage requires its documented inputs.

## Environment

`requirements.txt` lists dependencies used by the workflow and supplied SMI-TED loader. It pins scikit-learn to 1.7.2, the version recorded in the bundled scaler; the remaining original ML package versions have not been established. It is not a locked reproduction environment. The compatible `fast_transformers` implementation must also be installed; its original package source and version have not yet been specified in this revision. See [external/README.md](../external/README.md).

```bash
python -m pip install -r ML_code/requirements.txt
```

## 1. Prepare formulation sequences

The primary input is an Excel table with four solvent/additive slots, two salt slots and their molar fractions. Required columns and units are described in [data/README.md](../data/README.md). Component abbreviations are mapped using a property workbook containing `Short Name` and `SMILES`.

```bash
export ELECTROLYTE_PROPERTIES_XLSX=/absolute/path/to/electrolyte_properties.xlsx
export FORMULATION_SLOTS_XLSX=/absolute/path/to/formulations_six_slots.xlsx
python ML_code/preprocessing/replace.py
python ML_code/preprocessing/smiles_integrate.py
```

The output is `outputs/preprocessing/data_smited_fastcharge.csv`, with `seqA,T_C,Rate_C,y_retention` columns. The `*_mol_pct` input columns contain 0–1 **molar fractions** despite their names; the sequence writer multiplies them by 100 to produce molar percentages. Empty slots use `O<sep>0`.

`Unify.py` is an optional unit-conversion helper. It adds a ratio-string column and does not create the six-slot input table. Its mass/volume bases and supported unit combinations must match the source definitions. In particular, it does not distinguish molality `m` from molarity `M`; mixed bare-ratio/M inputs and multiple wt% additives need separate checking. Missing or unsupported components can be skipped by its existing branches. Normalisation to one alone does not establish that the complete composition was retained.

## 2. Encode training features

Prepare the encoder assets described in [external/README.md](../external/README.md):

```bash
export SMITED_DIR=/absolute/path/to/smi_ted_light
export FORMULATION_SEQUENCE_CSV="$PWD/outputs/preprocessing/data_smited_fastcharge.csv"
python ML_code/encode_fastcharge_embeddings.py
python ML_code/check_npy.py
```

Feature order is `[SMI-TED embedding, T_C, Rate_C]`. A 768-dimensional embedding yields 770 input features. X/y are saved as float32 in `data/private/X_fastcharge_data.npy` and `data/private/y_fastcharge_data.npy`. The encoder reads `n_embd` and `max_len` from its checkpoint.

The encoding entry point processes input rows in batches of 32. Missing temperatures are filled with 25 °C and missing rates with 1C. Targets are passed through numerically: neither the sequence script nor the encoder computes $Q_x/Q_{1\mathrm{C}}$ or converts percentages to fractions. Supply a curated ratio such as 0.721, not 72.1.

`check_npy.py` provides diagnostics. Equal array lengths do not establish record-level pairing. Its standardisation display fits a temporary scaler to the inspected array; that scaler is neither saved nor used for model training.

## 3. Train and evaluate

```bash
python ML_code/train_mlp.py
```

| Setting | Value |
| --- | --- |
| Split | Label-stratified random split; target train/validation/test proportions 70/10/20 |
| Stratification | Five equal-width target bins |
| Seed | 2026 |
| Hidden layers | 512, 512, 29 |
| Hidden-layer block | Linear, BatchNorm, ReLU, Dropout |
| Output | Linear, Sigmoid |
| Dropout | 0.13254228667616397 |
| Optimiser | AdamW |
| Learning rate | 0.0038369729910356348 |
| Weight decay | 1.0931541508233718e-05 |
| Nominal batch size | 64; adjusted by the existing code in some single-sample-remainder cases |
| Loss | MSE |
| Maximum epochs | 600 |
| Early stopping | 100 epochs without validation-RMSE improvement |
| MLP device | CPU |

StandardScaler is fitted on the training split only. The selected model is evaluated on the train, validation and test splits. Outputs include per-split error tables, a scatter plot and the three model artefacts documented in [models/README.md](models/README.md). Default output directory: `outputs/training/FixedBest_FinalRelease_<timestamp>/`.

The split operates on records, without publication/formulation grouping. Interpretation of the reported metrics should follow that split design. The entry point trains one supplied X/y pair; feedback stages use their corresponding input arrays rather than an automatic merging routine in this script.

## 4. Generate candidates

Provide a component workbook with `Short Name,SMILES,MW (g/mol)`:

```bash
export CANDIDATE_MATERIALS_XLSX=/absolute/path/to/candidate_materials.xlsx
python ML_code/generate_candidates.py
```

Default output: `outputs/candidates/place.csv`. The generator samples **mass fractions**, converts them to molar percentages and writes the six-slot `seqA`. The seed is 2026; the target is 2,000,000 accepted rows with an attempt limit of 100,000,000.

Deduplication is limited to the most recent 300,000 sequence hashes; the output is not guaranteed to contain globally unique formulations. Any final global deduplication/filtering is a separate operation.

The ordinary subspaces apply a base-solvent mass-fraction sum of at least 0.65; the `ACADEMIC_HIGH_FUNC` branch does not apply that minimum. Auxiliary-salt mass fraction 0.01 means 1 wt%. The original sampling rules are retained in the script.

## 5. Predict and rank

```bash
export SMITED_DIR=/absolute/path/to/smi_ted_light
export MLP_MODEL_DIR=/absolute/path/to/trained_mlp
export CANDIDATE_CSV=/absolute/path/to/candidates.csv
python ML_code/predict_5c.py
```

`predict_5c.py` always uses 25 °C and 5C. It writes `all_pred_place_5C_<timestamp>.csv`, `top20_pred_place_5C_<timestamp>.csv` and `inference_run_info_place_5C_<timestamp>.json` under `outputs/predictions/5c/`. Use a distinct output directory for separate runs. Its predictions are stored as `pred_retention_5C`.

`predict_default.py` uses row-wise `T_C` and `Rate_C` where available, with defaults of 25 °C and 4C. It writes `pred_retention` to its output and replaces its previous prediction CSVs. It is an alternative entry point, not a prerequisite for the fixed-5C script.

The supplied model, scaler and feature-index files are included in `ML_code/models/final/`. Their dimensions match the prediction code; see [models/README.md](models/README.md) for details and checksums. Dimension checks alone do not establish that files came from the same training run. The matching SMI-TED checkpoint is still required and is not bundled. The vocabulary is included under `external/smi_ted_light/`.

## Paths and overrides

| Environment variable | Default repository-relative path |
| --- | --- |
| `ELECTROLYTE_PROPERTIES_XLSX` | `data/private/electrolyte_properties.xlsx` |
| `FORMULATION_RAW_XLSX` | `data/private/formulations_raw.xlsx` |
| `FORMULATION_MOLES_XLSX` | `outputs/preprocessing/data_mol_ration.xlsx` |
| `FORMULATION_SLOTS_XLSX` | `data/private/formulations_six_slots.xlsx` |
| `FORMULATION_SMILES_XLSX` | `outputs/preprocessing/data_replaced.xlsx` |
| `FORMULATION_INTEGRATED_XLSX` | `outputs/preprocessing/data_smiles_integrate.xlsx` |
| `FORMULATION_SEQUENCE_CSV` | `outputs/preprocessing/data_smited_fastcharge.csv` |
| `FASTCHARGE_X_NPY` | `data/private/X_fastcharge_data.npy` |
| `FASTCHARGE_Y_NPY` | `data/private/y_fastcharge_data.npy` |
| `MLP_OUTPUT_DIR` | `outputs/training/FixedBest_FinalRelease_<timestamp>` |
| `CANDIDATE_MATERIALS_XLSX` | `data/private/candidate_materials.xlsx` |
| `CANDIDATE_CSV` | `outputs/candidates/place.csv` |
| `SMITED_DIR` | `external/smi_ted_light` |
| `MLP_MODEL_DIR` | `ML_code/models/final` |
| `PREDICTION_OUTPUT_DIR` | `outputs/predictions/5c` or `outputs/predictions/default` |

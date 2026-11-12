# Reproducibility and run management

## 1. Preserve the computational context

A result should be traceable to:

- the Git commit or release tag containing the code;
- the input file identity and checksum;
- the command and non-secret configuration used;
- the Python, operating-system and package versions;
- the random seed and device where relevant;
- the encoder checkpoint, vocabulary and MLP artefacts;
- the output files and run-information record.

Do not place API keys, passwords, access tokens or private credentials in run records.

## 2. Separate software environments

The repository intentionally has module-specific dependency files:

| Environment | Dependency file | Notes |
| --- | --- | --- |
| Literature extraction | `LLM_code/requirements.txt` | Recorded OpenAI, pandas, dotenv and httpx versions |
| Optional SOCKS proxy | `LLM_code/requirements-proxy.txt` | Adds optional SOCKS transport |
| ML and SMI-TED | `ML_code/requirements.txt` | Includes scikit-learn 1.7.2; not a complete lockfile |

The original `fast_transformers` distribution/source and the full historical ML package versions have not yet been established. Do not describe the current dependency files as a fully locked reproduction environment. Record the installed versions in the actual environment used for a published result.

## 3. Protect private inputs and credentials

- Put local API credentials in `LLM_code/API_key.env`; the tracked `API_key.env.example` is a template only.
- Keep curated literature data, property workbooks, experimental records and X/y arrays in `data/private/` or another approved private location.
- Keep intermediate files, training runs, candidate tables, prediction results and logs in `outputs/`.
- Both local data and generated outputs are excluded by `.gitignore`.
- Never commit a populated `.env`, `API_key.env`, raw literature collection or private workbook.

Before every push, inspect the staged file list and check for large or sensitive files.

## 4. Track each run

For each important run, preserve the generated run-information JSON where the script provides one. Also record the following in a lab notebook or run log:

| Field | What to record |
| --- | --- |
| Run ID | A unique date/time or experiment identifier |
| Code | Git commit SHA and branch |
| Input | File name, row count, SHA-256 and data version |
| Configuration | Command, environment-variable overrides and non-secret parameters |
| Environment | Python, package versions, OS and hardware/device |
| Randomness | Random seed and any sampling/splitting settings |
| Models | MLP, scaler, feature-index, SMI-TED checkpoint and vocabulary checksums |
| Outputs | Full output paths, row counts, summary files and checksums |
| Review | Metrics, warnings, exclusions and the person/date of review |

The inference scripts record several file checksums and runtime settings. Keep those JSON files with the corresponding full prediction tables.

## 5. Promote a trained MLP model safely

`train_mlp.py` writes a timestamped training directory under `outputs/training/`. Prediction scripts default to `ML_code/models/final/`. Promotion between these locations is a deliberate review step, not an automatic operation.

1. Review the train, validation and test metrics and error tables.
2. Confirm which training run is selected and record its run directory and Git commit.
3. Copy the following three files from the same run as one unit:
   - `best_model_fixed.pth`
   - `best_scaler_fixed.pkl`
   - `best_features.npy`
4. Update `ML_code/models/final/artifacts.json` with the new file sizes, SHA-256 checksums, model/scaler dimensions and training-run provenance.
5. Check that the prediction code expects the same feature order and model architecture.
6. Submit the artefact update through a pull request. Do not overwrite the current selected model before the new set has been reviewed.

The SMI-TED checkpoint is a separate dependency. Use the checksum in [the encoder instructions](../external/README.md) to verify the downloaded file.

## 6. Avoid accidental result overwrites

- `predict_5c.py` includes a timestamp in its output names by default.
- `predict_default.py` uses fixed output names by default and removes the previous full-prediction and Top-20 files before running. Set `PREDICTION_OUTPUT_DIR` to a new directory for each run if previous results must be retained.
- Keep the complete prediction CSV, not only Top 20, when running salt-stratified post-processing.
- Keep the post-processing summary and run-information JSON with the exported per-salt tables.

## 7. Data lineage and scientific interpretation

- The LLM extraction script structures literature text; it does not replace manual verification.
- The curated target ratio is passed into the model dataset. It is not calculated by the preprocessing or embedding scripts.
- The encoding script preserves row order. Maintain the same row order when preparing X and y; matching array lengths alone cannot prove that individual features and labels correspond.
- Candidate-generation mass fractions and sequence molar percentages are different quantities. Preserve the original candidate table and its units.
- Model predictions are prioritisation scores, not experimental measurements.
- Record experimental validation independently and explicitly link each tested formulation to the exact candidate sequence and screening run.

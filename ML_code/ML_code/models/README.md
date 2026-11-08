# Trained MLP artefacts

Prediction requires three files from the same training run:

| File | Contents and use |
| --- | --- |
| `best_model_fixed.pth` | SprintMLP state dictionary, including BatchNorm state |
| `best_scaler_fixed.pkl` | StandardScaler fitted on the training split |
| `best_features.npy` | Feature-index array applied after full-input standardisation |

The training script saves `np.arange(X.shape[1])` as the feature-index array; it represents all input columns in this implementation.

The three supplied files are included in [`final/`](final/). This is the default `MLP_MODEL_DIR`; set that environment variable to use a different directory. The uploaded copies named `best_scaler_fixed(1).pkl` and `best_features(1).npy` have been copied without the `(1)` suffix; their bytes are unchanged.

The bundled state dictionary has linear-layer shapes `770 → 512 → 512 → 29 → 1`, with BatchNorm states matching the supplied SprintMLP. The scaler has 770 input features and records scikit-learn version 1.7.2. The integer feature-index array has 770 entries, exactly `0, …, 769`.

File sizes, SHA-256 checksums and inspected properties are recorded in [`final/artifacts.json`](final/artifacts.json). These checks establish structural compatibility with the scripts; they do not independently establish common training-run provenance or reproduce the reported numerical results. The scaler's `n_samples_seen_` records the number of samples used to fit that scaler, not the complete literature-database size.

Prediction applies SMI-TED encoding, appends temperature and C-rate, standardises the full input, selects columns by index and evaluates the MLP. The encoder checkpoint, vocabulary, input-feature order and scaler must match the training configuration. A 768-dimensional encoder yields 770 full input features before any indexing.

The SMI-TED pretrained checkpoint is separate from the trained MLP artefacts; see [external/README.md](../../external/README.md).

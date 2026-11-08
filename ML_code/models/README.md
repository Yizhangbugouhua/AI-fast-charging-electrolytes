# Trained MLP files

The following files are bundled in [`final/`](final/), which is the default `MLP_MODEL_DIR`:

| File | Contents and use |
| --- | --- |
| `best_model_fixed.pth` | SprintMLP state dictionary, including BatchNorm state |
| `best_scaler_fixed.pkl` | StandardScaler fitted on the training split |
| `best_features.npy` | Input-column indices applied after standardisation |
| `artifacts.json` | File sizes, SHA-256 checksums and inspected model properties |

The linear-layer shapes in the state dictionary are `770 → 512 → 512 → 29 → 1`. The scaler accepts 770 input features and records scikit-learn version 1.7.2. The feature-index array contains all indices from 0 to 769; it is not a reduced feature subset.

Prediction combines the SMI-TED embedding with temperature and C-rate, standardises the full input, selects columns by index and evaluates the MLP. A 768-dimensional embedding gives 770 full input features. These shapes match the supplied scripts, but numerical reproduction and common training-run provenance are not established by dimension checks alone.

Use the model, scaler, indices, feature order and encoder configuration corresponding to the same training procedure. To use another model directory, set `MLP_MODEL_DIR` to that directory. The scaler's `n_samples_seen_` describes its fitted sample count, not the complete literature-database size.

The SMI-TED pretrained checkpoint is separate from these MLP files; see [external/README.md](../../external/README.md).

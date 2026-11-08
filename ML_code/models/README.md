# Trained MLP artefacts

Prediction requires three files from the same training run:

| File | Contents and use |
| --- | --- |
| `best_model_fixed.pth` | SprintMLP state dictionary, including BatchNorm state |
| `best_scaler_fixed.pkl` | StandardScaler fitted on the training split |
| `best_features.npy` | Feature-index array applied after full-input standardisation |

The training script saves `np.arange(X.shape[1])` as the feature-index array; it represents all input columns in this implementation.

Place the three files in `ML_code/models/final/`, or set `MLP_MODEL_DIR` to their directory. These files are **not included in this revision**, and no model-download URL is provided here.

Prediction applies SMI-TED encoding, appends temperature and C-rate, standardises the full input, selects columns by index and evaluates the MLP. The encoder checkpoint, vocabulary, input-feature order and scaler must match the training configuration. A 768-dimensional encoder yields 770 full input features before any indexing.

The SMI-TED pretrained checkpoint is separate from the trained MLP artefacts; see [external/README.md](../../external/README.md).

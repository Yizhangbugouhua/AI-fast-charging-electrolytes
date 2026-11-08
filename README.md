# Artificial intelligence-guided design of multicomponent electrolytes for fast-charging lithium-ion batteries

Research code for LLM-assisted literature data extraction, formulation representation, machine-learning prediction and candidate electrolyte screening.

The workflow connects literature extraction, six-slot formulation sequences, SMI-TED representations, an MLP regressor and experimental feedback. The prediction target is the capacity ratio $R_x=Q_x/Q_{1\mathrm{C}}$, with discharge specific capacities paired within the same rate-test series. It describes rate performance relative to 1C.

## Repository contents

| Directory | Contents | Instructions |
| --- | --- | --- |
| `LLM_code/` | Literature extraction script, complete prompts and API configuration template | [Literature extraction](LLM_code/README.md) |
| `ML_code/` | Formulation preparation, feature encoding, training, evaluation and candidate prediction | [Machine learning](ML_code/README.md) |
| `data/` | Input schemas, units and a format-only prediction example | [Data formats](data/README.md) |
| `external/` | SMI-TED loader, vocabulary and pretrained-checkpoint download instructions | [Encoder setup](external/README.md) |

The trained MLP files are included in [`ML_code/models/final/`](ML_code/models/README.md). The larger SMI-TED checkpoint is provided separately in [release v0.1.0](https://github.com/Yizhangbugouhua/AI-fast-charging-electrolytes/releases/tag/v0.1.0).

## Getting started

### Literature extraction

Follow [LLM_code/README.md](LLM_code/README.md) to install the extraction dependencies, configure an API key and process MinerU JSON files. The complete `SYSTEM_PROMPT` and `PROMPT_TEMPLATE` are included in `LLM_code/extract_literature.py`. The curated literature database is not bundled with the code.

### Formulation prediction

1. Install the ML and encoder dependencies described in [ML_code/README.md](ML_code/README.md) and [external/README.md](external/README.md).
2. Download `smi_ted_light.pt` from [release v0.1.0](https://github.com/Yizhangbugouhua/AI-fast-charging-electrolytes/releases/tag/v0.1.0) and place it in `external/smi_ted_light/`. Verify the SHA-256 given in the encoder instructions.
3. Keep the bundled model, scaler and feature-index files together in `ML_code/models/final/`.
4. Prepare a candidate CSV according to [data/README.md](data/README.md).
5. Run the following commands from the repository root:

```bash
export SMITED_DIR="$PWD/external/smi_ted_light"
export MLP_MODEL_DIR="$PWD/ML_code/models/final"
export CANDIDATE_CSV="$PWD/data/examples/prediction_input.csv"
python ML_code/predict_5c.py
```

This entry point predicts at 25 °C and 5C. It writes complete predictions, a Top20 table and run information to `outputs/predictions/5c/`. The bundled CSV demonstrates the input format only; its composition is not an experimental electrolyte, a recommended formulation or a validation result. Set `CANDIDATE_CSV` to the desired candidate table for screening.

`ML_code/predict_default.py` is an alternative entry point using row-specific temperature and C-rate where supplied, with defaults of 25 °C and 4C. Feature construction, training and candidate generation are documented in the [ML module instructions](ML_code/README.md).

## Software and model availability

| Resource | Availability |
| --- | --- |
| Literature extraction, preprocessing, encoding, training and prediction code | Included |
| Trained MLP weights, StandardScaler and feature indices | Included in `ML_code/models/final/` |
| SMI-TED loader and vocabulary | Included in `external/smi_ted_light/` |
| SMI-TED pretrained checkpoint | Separate [release asset](https://github.com/Yizhangbugouhua/AI-fast-charging-electrolytes/releases/download/v0.1.0/smi_ted_light.pt) |
| Curated literature database, experimental records and component-property workbooks | Not distributed in this repository |

The literature-extraction environment is documented with recorded versions. The ML requirements pin scikit-learn to 1.7.2, as recorded in the supplied scaler. Other original ML versions and the exact `fast_transformers` source/version remain to be established; the dependency list is not a complete reproduction environment.

The supplied MLP tensor shapes, scaler dimensions and feature indices have been inspected against the scripts. End-to-end encoding and prediction have not been executed for this distribution. Reproducing numerical results requires the corresponding data, software and encoder configuration.

## Data availability

Data-access enquiries should be directed to the corresponding author of the associated manuscript. Input schemas and composition units are documented in [data/README.md](data/README.md). Training requires the relevant curated data; the format-only example is not a substitute for that dataset.

## Related software

Molecular representations use SMI-TED; see [IBM/materials](https://github.com/IBM/materials) for the upstream project. The supplied loading and sequence-handling behaviour is described in [external/README.md](external/README.md).

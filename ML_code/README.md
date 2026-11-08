# Artificial intelligence-guided design of multicomponent electrolytes for fast-charging lithium-ion batteries

Research code for literature-based electrolyte data extraction, formulation representation, machine-learning prediction and candidate screening for fast-charging lithium-ion batteries.

The workflow connects **literature extraction → formulation sequences → SMI-TED representations → MLP prediction → candidate screening**. The regression target is the capacity ratio $R_x=Q_x/Q_{1\mathrm{C}}$, where the discharge specific capacities are paired within the same rate-test series. This target describes rate performance relative to 1C.

## Code organisation

| Directory | Description |
| --- | --- |
| [`LLM_code/`](LLM_code/README.md) | Literature extraction script, complete prompt templates and API configuration |
| [`ML_code/`](ML_code/README.md) | Formulation preparation, feature encoding, MLP training/evaluation and candidate prediction |
| [`data/`](data/README.md) | Input schemas, units and an illustrative prediction-input file |
| [`external/`](external/README.md) | SMI-TED loading and encoding implementation, with pretrained-asset requirements |

## Getting started

**Literature extraction:** follow the installation, input and configuration instructions in [LLM_code/README.md](LLM_code/README.md). The complete `SYSTEM_PROMPT` and `PROMPT_TEMPLATE` are included in `extract_literature.py`.

**Machine learning:** follow [ML_code/README.md](ML_code/README.md) for feature construction, training and candidate generation. The formulation representation combines a SMI-TED embedding with temperature and C-rate features.

For prediction at **25 °C and 5C**, first obtain the compatible SMI-TED checkpoint described in [external/README.md](external/README.md). The vocabulary and three MLP artefacts are bundled; see [ML_code/models/README.md](ML_code/models/README.md). Then run from the repository root:

```bash
export SMITED_DIR="$PWD/external/smi_ted_light"
export MLP_MODEL_DIR="$PWD/ML_code/models/final"
export CANDIDATE_CSV="$PWD/data/examples/prediction_input.csv"
python ML_code/predict_5c.py
```

The bundled CSV demonstrates the input format only; it is not an experimental formulation or a validation result. Predictions are written to `outputs/predictions/5c/`, including a full prediction table, a Top20 table and a run-information JSON file. For a real candidate collection, set `CANDIDATE_CSV` to that collection's CSV file.

The alternative `predict_default.py` reads row-specific temperature and C-rate values where available, with defaults of 25 °C and 4C. The two prediction scripts are separate entry points.

## Software and model assets

Dependencies and configuration are documented in the module READMEs. The literature-extraction dependencies are versioned. The ML dependency list pins scikit-learn to 1.7.2, the version recorded in the bundled scaler; the remaining original ML package versions have not been established. This list is not a locked reproduction environment.

This revision contains source code, an illustrative input, the SMI-TED vocabulary and three trained MLP artefacts. It does not include the matching SMI-TED checkpoint or component-property workbooks. The MLP weight shapes, scaler dimensions and feature indices have been checked against the supplied scripts. End-to-end encoding and prediction have not been executed for this package; the matching encoder checkpoint is still required.

## Data availability

The curated literature database and experimental records are not distributed with this repository. Input schemas and composition units are documented in [data/README.md](data/README.md). Data-access enquiries should be directed to the corresponding author of the associated manuscript. Training requires access to the relevant data; prediction on new formulations requires compatible model artefacts.

## Related software

The molecular representations use SMI-TED. See the [IBM materials repository](https://github.com/IBM/materials) for the upstream project and [external/README.md](external/README.md) for this workflow's loading and sequence-handling details.

# Repository structure and file placement

## Directory map

```text
AI-fast-charging-electrolytes/
├── .github/
│   ├── pull_request_template.md
│   └── workflows/
│       └── python-checks.yml
├── docs/
│   ├── RESEARCH_WORKFLOW.md
│   ├── REPOSITORY_STRUCTURE.md
│   └── REPRODUCIBILITY.md
├── LLM_code/
│   ├── extract_literature.py
│   ├── requirements.txt
│   ├── requirements-proxy.txt
│   ├── API_key.env.example
│   └── README.md
├── ML_code/
│   ├── preprocessing/
│   ├── postprocessing/
│   ├── models/
│   │   └── final/
│   ├── encode_fastcharge_embeddings.py
│   ├── check_npy.py
│   ├── train_mlp.py
│   ├── generate_candidates.py
│   ├── predict_5c.py
│   ├── predict_default.py
│   ├── requirements.txt
│   └── README.md
├── data/
│   ├── examples/
│   └── README.md
├── external/
│   ├── smi_ted_light/
│   └── README.md
├── CONTRIBUTING.md
├── README.md
└── .gitignore
```

Local working directories such as `data/private/` and `outputs/` are intentionally absent from the tracked repository. They are created on the research machine and excluded by `.gitignore`.

## Ownership of each directory

| Path | Put here | Do not put here |
| --- | --- | --- |
| `LLM_code/` | Literature extraction source, prompts, module-specific dependencies and instructions | API keys, MinerU source PDFs/JSON collections, private extraction results |
| `ML_code/preprocessing/` | Composition conversion, component mapping and sequence preparation | Intermediate private workbooks or generated CSV outputs |
| `ML_code/postprocessing/` | Reusable prediction-result filtering, ranking and deduplication scripts | Full prediction tables or selected private candidate files |
| `ML_code/models/final/` | The selected, internally consistent MLP weights, scaler, feature indices and metadata | SMI-TED checkpoint or artefacts copied from different training runs |
| `data/examples/` | Small, clearly labelled, non-sensitive examples that demonstrate schemas | Private training records, experimental results or large candidate spaces |
| `external/smi_ted_light/` | The supplied loader and vocabulary; checkpoint installation instructions | Large pretrained checkpoints committed directly to Git |
| `docs/` | Research workflow, repository conventions and reproducibility guidance | Raw data or generated analysis results |
| `outputs/` (local only) | Extraction results, intermediate arrays, training runs, candidate tables, predictions, logs and plots | Source code or files that need to be version-controlled |
| `data/private/` (local only) | Curated literature data, experimental data, property workbooks and training arrays | Anything intended for public distribution without an explicit data-release review |

## File placement rules

1. Keep executable source code in the module responsible for that stage.
2. Keep module-specific setup instructions beside the module; use the root README as the navigation page.
3. Put generated artefacts under `outputs/<stage>/`, not beside source scripts.
4. Keep sensitive or non-distributable research inputs under `data/private/`.
5. Store only small, explicitly shareable examples under `data/examples/`.
6. Keep the selected MLP artefacts together. Do not copy only the weight file without its matching scaler and feature indices.
7. Keep the SMI-TED checkpoint separate from the MLP model artefacts.
8. Do not rename or move established scripts without searching for imports, environment-variable paths, README links and downstream references.

## Naming conventions

- Python modules and scripts: lowercase `snake_case.py`; existing names are retained where changing them would break documented commands.
- Reusable processing stages: descriptive action names, such as `select_topn_by_salt.py`.
- Generated run directories: include stage and timestamp.
- Prediction outputs: preserve the condition tag and timestamp when the script supports them.
- Model artefacts: retain the expected filenames unless the loading code and documentation are updated together.

## Version-control boundary

The Git repository tracks source code, documentation, small examples, selected model artefacts approved for distribution and metadata needed to interpret those artefacts. It does not track local environments, API credentials, raw literature collections, private spreadsheets, large generated candidate tables, prediction outputs or routine logs.

Before adding any data or model file, check its sensitivity, size, provenance, licence and reproducibility value. If a large artefact must be shared, use a versioned release asset or an approved data repository and document its checksum and download location.

# AI-guided design of multicomponent electrolytes for fast-charging lithium-ion batteries

Research code for **Artificial intelligence-guided design of multicomponent electrolytes for fast-charging lithium-ion batteries**.

**Status: repository under preparation.** Literature-extraction code and nine author-supplied preprocessing, feature-encoding, array-diagnostic, training, generation and prediction scripts are included, together with the supplied SMI-TED loader. The code has been reviewed and adapted for portable paths. The training-feature construction script is now included and its input/output contract matches training and prediction. An intermediate table conversion, the matching pretrained checkpoint/vocabulary and trained MLP files remain to be supplied. The supplied loader has been inspected statically; actual neural encoding has not been executed. Full training and inference have not been reproduced.

## Contents

| Location | Contents |
| --- | --- |
| [LLM_code](LLM_code/README.md) | Literature extraction, complete prompts, API configuration template and extraction dependencies |
| [ML_code](ML_code/README.md) | Actual formulation preparation, MLP training/evaluation, candidate sampling and two prediction entry points |
| [ML_code/CODE_REVIEW.md](ML_code/CODE_REVIEW.md) | Verified interfaces, identified gaps, limitations and next files to collect |
| [ML_code/models](ML_code/models/README.md) | Exact three-artifact loading contract; model files are not included yet |
| [data](data/README.md) | Table/array schemas, composition units and data access boundaries |
| [external](external/README.md) | Requirements for the author's SMI-TED implementation and pretrained assets |
| [REPOSITORY_PLAN.md](REPOSITORY_PLAN.md) | Current preparation checklist and reference repository links |

## Literature extraction

Follow [LLM_code/README.md](LLM_code/README.md). Full prompts are inside `extract_literature.py`. SOCKS support is optional. Actual extraction requires an API connection and makes model requests.

## Machine learning

Follow [ML_code/README.md](ML_code/README.md). Data preparation currently has documented missing transitions, so the supplied files must not be presented as a complete automated pipeline. The MLP predicts the author-defined capacity ratio, not ionic conductivity or cycling retention. No reference-paper model parameters, datasets or weights have been substituted for the author's files.

The extraction environment and machine-learning environment are separate. ML dependency names have been identified from imports; historical ML package versions have not yet been established.

## Data and model availability

The literature database and experimental records are not included. Reporting record counts and documenting input schemas do not release the database. From-scratch training requires the corresponding data and split information. Prediction requires the matching three model artifacts plus the supplied SMI-TED implementation and pretrained assets. The implementation is now included; the weights/vocabulary and three MLP artifacts are not yet included.

## Repository preparation

Unzip the package and upload the **contents** of the inner `AI-fast-charging-electrolytes` folder, preserving subfolders. Inspect browser selections explicitly: `.gitignore` does not filter manually selected uploads. Actual credentials and private inputs are excluded. Keep this preparation repository private until its release contents are settled.

No paper DOI, author list, release URL or software licence has been invented. Add the confirmed citation and licensing information before the reader-facing release.

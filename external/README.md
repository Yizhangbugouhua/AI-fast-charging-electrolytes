# SMI-TED representation model

`smi_ted_light/load.py` contains the loading, tokenisation and encoding implementation used by the workflow. The upstream SMI-TED project is [IBM/materials](https://github.com/IBM/materials). The supplied implementation includes the `encode_raw` interface for formulation strings.

## Required assets

The directory selected by `SMITED_DIR` must contain:

| File | Purpose | Included |
| --- | --- | --- |
| `load.py` | Loader and model implementation | Yes |
| `smi_ted_light.pt` | Matching pretrained checkpoint | No |
| `bert_vocab_curated.txt` | Matching vocabulary | No |

The scripts import `load_smi_ted` from `load.py` and pass the checkpoint and vocabulary filenames shown above. An upstream project link alone does not identify a compatible checkpoint/vocabulary pair; use the assets corresponding to this implementation and its original training configuration.

## Dependencies

The implementation imports PyTorch, NumPy, pandas, Transformers, RDKit, regex, tqdm and `fast_transformers`. The ML dependency file lists the named packages except for the original `fast_transformers` installation, whose exact distribution/source and version are not specified in this revision. ML package versions are not pinned here.

The precise upstream revision and attribution for the supplied loader should accompany a versioned software release. The model assets and their source information are currently outside this source-only package.

## Encoding behaviour

`encode_raw` bypasses RDKit SMILES normalisation and passes the formulation string to the tokenizer, Transformer encoder and autoencoder. It returns one vector per input sequence. The embedding dimension `n_embd` and maximum token length `max_len` are read from checkpoint hyperparameters.

The tokenizer uses the SMILES regular expression defined in `load.py`. With this expression, `<sep>` is split into `s`, `p`, `>` rather than emitted as one registered special token; numeric strings are tokenised character by character. Tokenisation uses truncation at `max_len`. The same implementation, vocabulary and checkpoint must be used for training-feature generation and prediction.

`load_smi_ted` sets evaluation mode. `encode_raw` detaches collected embeddings before constructing its returned tensor. The implementation automatically uses CUDA when available, so the MLP's CPU setting does not by itself force all encoder operations to run on CPU.
## Checkpoint download

Download [smi_ted_light.pt](https://github.com/Yizhangbugouhua/AI-fast-charging-electrolytes/releases/download/v0.1.0/smi_ted_light.pt) from release v0.1.0.

Place the downloaded file in `external/smi_ted_light/`, alongside `load.py` and `bert_vocab_curated.txt`.

SHA-256:
`baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375`

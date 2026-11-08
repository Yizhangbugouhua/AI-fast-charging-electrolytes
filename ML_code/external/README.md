# SMI-TED representation model

`smi_ted_light/load.py` contains the loading, tokenisation and encoding implementation used by the workflow. The upstream SMI-TED project is [IBM/materials](https://github.com/IBM/materials). The supplied implementation includes the `encode_raw` interface for formulation strings.

## Required assets

The directory selected by `SMITED_DIR` must contain:

| File | Purpose | Included |
| --- | --- | --- |
| `load.py` | Loader and model implementation | Yes |
| `smi_ted_light.pt` | Matching pretrained checkpoint | No |
| `bert_vocab_curated.txt` | Supplied vocabulary; compatibility with the checkpoint remains to be tested | Yes |

The scripts import `load_smi_ted` from `load.py` and pass the checkpoint and vocabulary filenames shown above. An upstream project link alone does not identify a compatible checkpoint/vocabulary pair; use the assets corresponding to this implementation and its original training configuration. The supplied vocabulary contains 2,393 entries; its SHA-256 is `8576b60e838336837f9e894457ef144f440c4d31bc8ceda2315fb5b28a6dfd95`.

## Dependencies

The implementation imports PyTorch, NumPy, pandas, Transformers, RDKit, regex, tqdm and `fast_transformers`. The ML dependency file lists the named packages except for the original `fast_transformers` installation, whose exact distribution/source and version are not specified in this revision. Only scikit-learn is pinned, using the version recorded in the bundled MLP scaler; this does not establish the encoder's original environment.

The precise upstream revision and attribution for the supplied loader should accompany a versioned software release. The pretrained checkpoint and its source information are not included in this package. The loader expects checkpoint hyperparameters under `hparams`; an MLP checkpoint cannot substitute for this encoder checkpoint.

## Encoding behaviour

`encode_raw` bypasses RDKit SMILES normalisation and passes the formulation string to the tokenizer, Transformer encoder and autoencoder. It returns one vector per input sequence. The embedding dimension `n_embd` and maximum token length `max_len` are read from checkpoint hyperparameters.

The tokenizer uses the SMILES regular expression defined in `load.py`. With this expression, `<sep>` is split into `s`, `p`, `>` rather than emitted as one registered special token; the supplied vocabulary has no `<sep>` entry. Numeric strings are tokenised character by character. Tokenisation uses truncation at `max_len`. The same implementation, vocabulary and checkpoint must be used for training-feature generation and prediction.

`load_smi_ted` sets evaluation mode. `encode_raw` detaches collected embeddings before constructing its returned tensor. The implementation automatically uses CUDA when available, so the MLP's CPU setting does not by itself force all encoder operations to run on CPU.

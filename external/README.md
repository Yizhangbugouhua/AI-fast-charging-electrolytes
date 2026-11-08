# SMI-TED encoder

The workflow uses `smi_ted_light/load.py` for loading, tokenisation and encoding. The upstream SMI-TED project is [IBM/materials](https://github.com/IBM/materials). The supplied implementation provides `encode_raw` for formulation strings.

## Files and placement

The directory selected by `SMITED_DIR` must contain all three files:

| File | Purpose | Availability |
| --- | --- | --- |
| `load.py` | Loader and model implementation | Included in this repository |
| `bert_vocab_curated.txt` | Supplied vocabulary with 2,393 entries | Included in this repository |
| `smi_ted_light.pt` | Pretrained encoder checkpoint | Separate download from [release v0.1.0](https://github.com/Yizhangbugouhua/AI-fast-charging-electrolytes/releases/tag/v0.1.0) |

## Checkpoint download

Download [`smi_ted_light.pt`](https://github.com/Yizhangbugouhua/AI-fast-charging-electrolytes/releases/download/v0.1.0/smi_ted_light.pt) and place it at:

```text
external/smi_ted_light/smi_ted_light.pt
```

Keep it alongside the supplied `load.py` and `bert_vocab_curated.txt`. The checkpoint is a separate release asset and is not embedded in the repository source ZIP. Access to a private repository's release assets requires repository read permission.

Checkpoint SHA-256, as reported for the uploaded release asset:

```text
baf252dbc081a00c68d2fd6ed8b08a0db0fa15244cfea442d49f0619a3a65375
```

Verify from the repository root on Linux:

```bash
sha256sum external/smi_ted_light/smi_ted_light.pt
```

Or on Windows PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 .\external\smi_ted_light\smi_ted_light.pt
```

Compare the complete hash; hexadecimal letter case does not affect the value. A matching checksum establishes file identity, not model compatibility. End-to-end encoding with this checkpoint has not been executed for this distribution.

Vocabulary SHA-256:

```text
8576b60e838336837f9e894457ef144f440c4d31bc8ceda2315fb5b28a6dfd95
```

## Dependencies and checkpoint format

The implementation imports PyTorch, NumPy, pandas, Transformers, RDKit, regex, tqdm and `fast_transformers`. The ML requirements list the named packages except for the original `fast_transformers` distribution/source, which remains to be established. The original encoder package versions are not fully documented.

`load_smi_ted` expects the filenames in the table above. Its checkpoint loader reads hyperparameters from `hparams`, then uses the supported state-dictionary fields. An MLP checkpoint cannot substitute for this pretrained encoder checkpoint. The precise upstream revision and provenance of the supplied loader and checkpoint remain to be documented.

## Encoding behaviour

`encode_raw` bypasses RDKit SMILES normalisation and passes formulation strings to the tokenizer, Transformer encoder and autoencoder. It returns one vector per sequence. The embedding dimension `n_embd` and maximum token length `max_len` are read from checkpoint hyperparameters.

The tokenizer retains the SMILES regular expression in `load.py`. It splits `<sep>` into `s`, `p`, `>`; the supplied vocabulary contains no `<sep>` entry. Numeric strings are tokenised character by character, and sequences are truncated at `max_len`. Training-feature generation and prediction must use the same implementation, vocabulary and checkpoint.

`load_smi_ted` sets evaluation mode, and `encode_raw` detaches the collected embeddings. The loader automatically uses CUDA when available; configuring the MLP for CPU does not by itself force encoder computation onto CPU.

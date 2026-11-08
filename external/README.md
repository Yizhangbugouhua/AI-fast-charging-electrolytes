# Supplied SMI-TED implementation and remaining assets

The author-supplied `smi_ted_light/load.py` is included byte-for-byte. Its `load_smi_ted(folder, ckpt_filename, vocab_filename)` and `encode_raw(seqs, batch_size, return_torch)` interfaces match the workflow scripts. Static checks confirm evaluation mode and detached returned embeddings. The implementation has not been executed with actual pretrained weights in this review.

## Expected files

The directory defaults to `external/smi_ted_light/` and can be changed with `SMITED_DIR`:

| File | Status |
| --- | --- |
| `load.py` | Supplied and included unchanged |
| `smi_ted_light.pt` | Matching pretrained checkpoint not supplied |
| `bert_vocab_curated.txt` | Matching vocabulary not supplied |

The loader reads n_embd and max_len from the checkpoint. Actual embedding width and whether formulation sequences are truncated require those assets. The loader automatically uses CUDA when available, independently of the workflow's MLP device setting.

## Dependencies and provenance

Imports include torch, numpy, pandas, transformers, RDKit, regex, tqdm and fast_transformers. Dependency names have been identified, but historical versions and the exact fast_transformers installation source remain unverified. Do not assume a similarly named package or a different fork reproduces the implementation.

Upstream SMI-TED reference: https://github.com/IBM/materials . The supplied file's exact upstream revision, local changes and applicable source attribution remain to be recorded; the file is not presented as a newly invented encoder. No third-party checkpoint has been substituted.

## Raw sequence handling

encode_raw skips RDKit normalisation and sends the original sequence to the tokenizer. The tokenizer still uses the supplied SMILES regular expression: `<sep>` is split into `s`, `p`, `>` by this regex, rather than being emitted as a single special token. Numeric text is tokenised character by character. The code is preserved unchanged so historical features are not silently altered. Full token-ID handling must be checked with the actual vocabulary.

load_smi_ted calls model.eval(); encode_raw detaches intermediate outputs before returning its Tensor. Complete checkpoint loading, tokenizer compatibility and actual neural encoding remain to be tested.

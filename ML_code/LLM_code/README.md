# LLM-assisted literature data extraction

This module extracts electrolyte compositions, electrode information, test conditions and electrochemical performance from literature text parsed with MinerU. It implements the initial structured-extraction stage described in the manuscript's Supplementary Information.

## Files

| File | Purpose |
| --- | --- |
| `extract_literature.py` | Extraction pipeline, including complete `SYSTEM_PROMPT` and `PROMPT_TEMPLATE` |
| `requirements.txt` | Versioned base dependencies |
| `requirements-proxy.txt` | Optional SOCKS-proxy dependencies |
| `API_key.env.example` | API-key configuration template |

## Installation and configuration

From the repository root:

```bash
cd LLM_code
python -m pip install -r requirements.txt
cp API_key.env.example API_key.env
mkdir -p input_json
```

Edit the local `API_key.env` to set `OPENAI_API_KEY`. Place the MinerU JSON files directly in `input_json/`; the input scan is not recursive. Each filename should remain traceable to its source publication.

```bash
python extract_literature.py
```

The default output directory is `extraction_result/`. Input and output locations can be overridden:

```bash
export LITERATURE_INPUT_DIR=/absolute/path/to/mineru_json
export LITERATURE_OUTPUT_DIR=/absolute/path/to/extraction_results
python extract_literature.py
```

Running the script sends API requests. Access to the configured model is required. The identifier below records the extraction configuration; if a different model is used, record that change with the new results.

## Extraction settings

| Setting | Value |
| --- | --- |
| Model | `gpt-4-1106-preview` |
| Temperature | 0.0 |
| Maximum response length | 2048 tokens |
| Text-window length | 10,000 characters |
| Overlap | 600 characters |
| Additional input truncation | None |
| Saved text preview | Up to 5,000 characters; does not truncate the model input |
| Concurrent window tasks | 8 |
| Maximum request attempts | 3 |
| Retry wait | 8 seconds |
| Response format | Fixed 16-column Markdown table, or `Nothing extracted.` |

Prompt examples specify the output format; they are not experimental records. Only text is sent to the model. Rate-performance figures are read and checked manually during subsequent curation.

## Software

The recorded extraction environment uses Python 3.10.18, MinerU 2.1.10, OpenAI SDK 1.99.9, pandas 2.3.3, python-dotenv 1.1.1 and httpx 0.28.1. MinerU is an upstream PDF-processing tool and is not invoked by this extraction script. These recorded versions are not a complete operating-system or environment lockfile.

SOCKS support is optional and is imported only when a proxy is enabled:

```bash
python -m pip install -r requirements-proxy.txt
export LITERATURE_SOCKS_PROXY=socks5://127.0.0.1:1080
python extract_literature.py
```

Replace the example proxy address with the actual address. Without this setting, the script uses a direct connection. The optional proxy package is not version-pinned.

## Outputs

| Output | Contents |
| --- | --- |
| Grouped `<filename>_parsed.csv` files | Structured records for each publication |
| Grouped `<filename>_skeleton.json` files | Window indices, text previews, responses and processing metadata |
| `skeleton_parsed_all.csv` | Combined extraction records |
| `doc_framework_*.log` | Processing and API-request logs |

Publications are grouped in batches of 50. The output includes `file_name`, `section` and `window_index`; DOI identifiers are not generated automatically. The extracted fields are defined by `EXPECTED_COLS` in the script.

The extraction output requires subsequent screening, duplicate handling and manual verification. The field `Retention efficiency` is an extracted literature field; the script does not calculate the model target $R_x=Q_x/Q_{1\mathrm{C}}$. Capacity pairing and label construction follow the SI methods.

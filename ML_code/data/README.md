# Data formats

This directory documents input/output schemas and contains a format-only prediction example. The curated literature database, experimental records and component-property workbooks are not included. Data-access enquiries should be directed to the corresponding author of the associated manuscript.

## Prediction example

[`examples/prediction_input.csv`](examples/prediction_input.csv) contains one illustrative six-slot input. Its composition was constructed solely to demonstrate the file format. It is not an experimental electrolyte, a recommended formulation or a validation record, and it has no measured or predicted performance label.

The fixed-5C prediction script accepts a `seqA` column and uses 25 °C and 5C for all rows. The example's `formulation_id` is an optional identifier preserved in output tables.

## Six-slot composition table

`replace.py` expects these component columns:

`solvent1, solvent2, solvent3, solvent4, salt1, salt2`

`smiles_integrate.py` additionally requires the corresponding six `*_mol_pct` columns, `Temperature`, `Rate_C` (or `Rate`) and `Retention efficiency`.

| Field | Convention |
| --- | --- |
| Component slots | Four solvent/additive slots followed by two salt slots |
| `*_mol_pct` input values | **Molar fractions, 0–1**, despite the column names |
| Sequence ratios | Molar percentages, obtained by multiplying the input fractions by 100 |
| Empty slot | `O<sep>0`; O is a placeholder here, not added water |
| Temperature | Numerical °C |
| Rate | Numerical C-rate, with the charge/discharge protocol defined by the source record |
| Target | Curated $R_x=Q_x/Q_{1\mathrm{C}}$ as a ratio |

Sequence layout is six `SMILES<sep>molar_percentage` pairs joined with `<sep>`. Component abbreviations are mapped using the property workbook's `Short Name` and `SMILES` columns. Unmatched abbreviations remain unchanged in the mapping script and should be checked before encoding. Records that exceed the available slots require explicit preparation; components should not be silently discarded.

## Training CSV and arrays

The sequence-construction script writes:

| Column | Description |
| --- | --- |
| `seqA` | Six-slot formulation sequence |
| `T_C` | Temperature in °C |
| `Rate_C` | C-rate |
| `y_retention` | Curated capacity ratio, passed from `Retention efficiency` |

The scripts do not calculate the target ratio or convert percentage labels. Use 0.721 for a 72.1% ratio. Missing temperature values are filled with 25; the encoder fills missing rates with 1. These defaults should be distinguished from experimentally reported conditions.

`encode_fastcharge_embeddings.py` preserves CSV row order and writes:

- `X_fastcharge_data.npy`: float32 two-dimensional input, ordered as embedding, temperature, rate.
- `y_fastcharge_data.npy`: float32 one-dimensional target in the corresponding row order.

## Property workbooks and candidate CSV

- The abbreviation/physical-property workbook requires `Short Name`, `SMILES`, a density column containing `Specific Gravity`, and a molecular-weight column containing `MW`. The unit-conversion helper assumes g/mL and g/mol.
- `candidate_materials.xlsx` requires `Short Name,SMILES,MW (g/mol)`.
- `Unify.py` reads original composition fields `Solvent1/ration1` through `Solvent4/ration4`, two `Additive*/Concentration*` pairs and three `Lithium salt */Concentration*` pairs. It exports a `Unified_mole_ratio` string rather than a six-slot table. See [ML_code/README.md](../ML_code/README.md) for conversion boundaries.
- Generated candidates contain component labels, `x1_wt` through `x6_wt`, `mol_sum` and `seqA`. The `x*_wt` values are **mass fractions**, whereas sequence values are **molar percentages**.

Private local inputs default to `data/private/`; generated results default to `outputs/`. Those directories are excluded by `.gitignore`.

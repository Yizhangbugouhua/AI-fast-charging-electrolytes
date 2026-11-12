# Contributing and repository maintenance

This repository supports a research workflow. Changes should preserve scientific traceability as well as code functionality.

## Standard change workflow

1. **Start from an up-to-date `main`.** Check open pull requests before beginning overlapping work.
2. **Create a focused branch.** Use a name such as `feat/<topic>`, `fix/<topic>`, `docs/<topic>` or `data/<topic>`.
3. **Make one coherent change.** Avoid combining unrelated model changes, data updates and documentation edits in one pull request.
4. **Update documentation.** If an input, output, command, dependency, model artefact or default path changes, update the relevant README and workflow documentation in the same change.
5. **Run checks.** Compile Python files and run the available unit tests. For data- or model-dependent work, record which checks could not be run without private inputs or hardware.
6. **Inspect the diff.** Confirm no credentials, private data, accidental generated outputs or unrelated binary files are included.
7. **Open a pull request.** Describe the purpose, affected pipeline stages, validation, input/output changes and reproducibility impact.
8. **Review before merging.** Resolve review comments and confirm automated checks. Merge only after the change is understood and validated.
9. **Tag meaningful releases.** Use a version tag for a stable code/model bundle; document required external checkpoints and checksums.

Do not push research changes directly to `main`.

## Branch and commit names

Suggested branch prefixes:

- `feat/`: new reusable capability
- `fix/`: bug fix
- `docs/`: documentation or repository-structure changes
- `refactor/`: internal code reorganisation without intended scientific changes
- `test/`: tests or validation infrastructure
- `release/`: release preparation

Use short, action-oriented commit messages, for example:

- `Add salt-stratified candidate selection`
- `Document model promotion procedure`
- `Fix candidate output path handling`

## Scientific and data safeguards

- Do not change target definitions, filtering rules, units, sampling constraints, model architecture or split strategy as part of a cosmetic refactor.
- If a scientific method changes, explain the reason and the expected effect in the pull request.
- Keep private literature tables, experimental data, property workbooks, candidate spaces, predictions and local run outputs out of Git unless their release has been explicitly approved.
- Never commit API keys or populated environment files.
- Do not replace a selected model artefact without its matching scaler, feature-index file and updated checksum/provenance metadata.
- Do not claim end-to-end reproducibility when the required private data, checkpoints, package versions or hardware-specific dependencies have not been verified.

## Minimum validation

For a Python-only change:

```bash
python -m compileall -q LLM_code ML_code external
python -m pytest -q
```

For changes to candidate ranking or deduplication, include tests for duplicate handling, salt grouping, ordering and boundary cases.

For changes involving model training or prediction, additionally record the input data version, model/checkpoint checksums, run configuration and output summary. If the full run cannot be executed, state that limitation explicitly.

## Pull request expectations

A pull request should answer:

- What changed and why?
- Which research workflow stage is affected?
- Are input/output schemas or default paths changed?
- Which checks were run, and what were their results?
- Were any private data or large files added?
- Can the result be traced to a code version and run configuration?

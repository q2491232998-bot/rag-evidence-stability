# Reranker batching and evidence stability

Research data and frozen code for **Reranker Batching and Evidence Stability in
Budgeted Retrieval-Augmented Generation**, by Zebin Xu, School of Cyberspace
Security, Beijing University of Posts and Telecommunications.

Version: **1.0.0**, 3 November 2026. Prepared for an Information Processing &
Management submission; this repository does not imply acceptance or publication
of the manuscript.

## Reproduce the reported numerical comparisons

Download the repository at tag `v1.0.0`, extract it, and run:

```sh
python replay.py
```

Python 3.10 or newer is sufficient; the replay uses only the standard library.
It checks the frozen primary results and writes `replay_result.json`. No model,
GPU, study question text or answer text is required for this arithmetic replay.
`manifest.json` supplies SHA-256 digests for the released files.

| Panel | Questions | Native context changes | FP32/MATH context changes |
| --- | ---: | ---: | ---: |
| S22, HotpotQA train-derived | 64 | 17 | 0 |
| T96, MuSiQue train-derived | 96 | 41 | 0 |

These are serial-to-batch-four comparisons within each precision configuration.
T96 changes the question source while keeping the same BEIR HotpotQA retrieval
corpus. It is not an official MuSiQue benchmark result or cross-corpus validation.
Zero observed changes are not a guarantee of batch invariance. The FP32/MATH
control does not preserve native BF16 serial evidence for every question.

## Contents and reconstruction limits

- `S22/` and `T96/`: both repetitions of four arms, reranker scores, full orders,
  selected indices, context/output hashes, measured request components, paired
  F1/EM scores and source identifiers. Repetitions are not independent questions.
- `replay/`: retained mixed-window/order and four-budget packing analyses.
- `historical/`: development summaries, not additional independent replications.
- `protocols/`: source pins, selection rules, frozen configurations and provenance.
- `implementation/`: frozen source/configuration snapshot and original MuSiQue
  metric references. Original paths within this snapshot are retained for audit.

The arithmetic replay reproduces counts, exact primary McNemar probabilities,
score drift, mean quality and latency point estimates from retained observations.
It does not rerun model inference or independently regenerate packed text from
hashes. The archived budget replay results can be inspected but re-executing the
tokenizer/packer requires the original corpus and pinned tokenizer. Re-running
the original inference scripts requires the documented model files, corpus
indices, local asset manifests and original project layout; this release is not
a self-contained inference image. Full original execution traces remain retained
on the research server.

Study questions, passages, generated answer text, model weights, credentials and
private server locations are omitted. Question and corpus identifiers plus hashes
support matching to the original public sources. The MuSiQue source was obtained
from a pinned mirror; byte equivalence with the authors' original download archive
has not been established. See `protocols/musique_source_provenance.json`.

## Sources and licenses

Original code, including `replay.py` and `implementation/src/`, is MIT licensed.
Original derived numerical data, manifests and documentation are CC BY 4.0.
See `LICENSE-CODE.txt`, `LICENSE-DATA.txt` and `THIRD_PARTY.md`. Third-party
source data, models and metric code retain their own licenses; this release does
not grant rights over them. External attribution is preserved.

## Citation

Use the version-specific Zenodo DOI linked from the GitHub release when available.
`CITATION.cff` supplies author, title, date, version and repository metadata.
The archive contains research data and software, not a claim of journal publication.

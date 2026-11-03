# IPM independent question transfer protocol

This protocol is frozen internally before selecting or retrieving 96 MuSiQue questions. It is not a public preregistration. The earlier v2 power calculation proposed this sample size before seeing any new executions.

## Scope and selection

Use the original 256-question MuSiQue-Ans training-derived reserved partition. Verify the original blinded and labeled file SHA-256 values. Audit retained project evidence by question ID and normalized question. Preparation registries are not model evaluations; any matched evidence must be classified before use. The audit cannot establish absence of external or unrecorded use, model pretraining exposure, or shared single-hop components.

Among eligible IDs, sort by SHA256 of `ipm-crossset-v3|20261102|ID`, take the first 96, and retain their original partition order. No answer, hop label, retrieval result or measured outcome enters selection. Select once. Do not replace questions or enlarge the sample after seeing results. Export answer labels only after inference completes.

Keep the BEIR HotpotQA collection of 5,233,329 paragraphs and the existing BGE-small/E5-base indices. This design transfers the question source while holding the retrieval corpus and policy fixed. It is not MuSiQue's official supplied-context or dataset-corpus evaluation. Report that limitation and do not compare answer scores to official MuSiQue leaderboards.

## Execution and primary inference

Use BGE top 100 union E5 top 100, canonical BGE order with corpus-row ties. Reuse the frozen S22 request implementation: Qwen3-Reranker-0.6B, Qwen3-8B, top ten whole-passage packing, 512 reader tokens, greedy generation capped at 32 tokens. Reranker input cap is 1,024, reader input cap 2,048. The four arms are native BF16 serial and batch four, and combined FP32/MATH serial and batch four. FP32 casts the same BF16-rounded parameters and does not change reader execution. Parameter conversion time remains outside request timing.

Two repetitions produce 768 primary requests. Two selected questions provide eight excluded warm-up requests. Question order is hashed independently per repetition; all 24 arm permutations are balanced across 96 questions and reversed on the second repetition. Repetition zero defines the primary per-question outcomes; the second repetition checks repeatability and contributes to per-question median timing. Repetitions are not additional independent questions.

The sole confirmatory test is one-sided exact McNemar at alpha 0.05, comparing native serial-to-batch context-change indicators with FP32/MATH serial-to-batch indicators. The alternative is more native-only than FP32/MATH-only changes. Report all four contingency cells and the exact p-value. A failed structural or repeatability audit prevents a successful confirmation claim. The zero-change Wilson bound remains necessary when applicable.

Selected membership, top-ten order, full ranking, output tokens, alias-aware MuSiQue EM/F1, latency and memory are secondary descriptive results. Use 20,000 question-level bootstrap replicates for paired quality differences and aggregate median-latency saving; report these intervals without a multiplicity-adjusted confirmatory claim. Preserve the original deployment gates without redefining their thresholds. No prompt search, precision search, extra budget sweep or alternative model is authorized by this protocol.

## Checks and stopping

Freeze source dependencies, configuration, selection rule and index manifests before selection. Freeze candidate inputs and execution schedule after an independent CPU retrieval audit and before model execution. Reconstruct all saved input tokenizations, prompt hashes and deterministic packs after completion, and verify parameter and buffer digests.

Run only on the remote RTX 3090 Ti under the existing shared-GPU lock and pause marker. Allow at most four hours of primary model execution and 10 GiB of new result files. Keep every interrupted attempt, fail on external GPU use, and do not retry outcome-bearing requests to obtain significance. Report a null or failed study as such. No submission, publication or external communication is part of execution authorization.

# Tokenizer Rules and Competitor Landscape

Date: 2026-04-25

## Rule Read

Custom tokenizers are allowed, but they carry extra review burden.

The official README says Parameter Golf is tokenizer-agnostic and scored as bits
per byte on FineWeb validation. It also explicitly says that tokenizer or
dataset changes must prove the `val_bpb` calculation with certainty, because
tokenizer bugs can unjustly improve the score.

Relevant local lines:

- `README.md:6`: tokenizer-agnostic, bits-per-byte scoring.
- `README.md:8`: novel tokenizers are in-scope as a creative compression
  scheme.
- `README.md:173`: no external downloads, training dataset access, or network
  calls during evaluation; artifact must be self-contained and reproducible.
- `README.md:204`: tokenizer or dataset changes require proof that `val_bpb` is
  calculated correctly.
- `data/README.md`: custom tokenizer rebuilds should use the same published docs
  cache and preserve reproducible manifests.

The unofficial validity guide gives the operational constraint: BPB must be
computed from actual bytes, not a hardcoded bytes-per-token constant. For
SentencePiece, byte counts must handle the leading-space marker, byte fallback
tokens, and zero-byte control tokens.

## Normalization Risk

The open policy risk is not "can we use a different vocab?" It is "did we make
the task easier with lossy text normalization?"

Issue #1604 says custom tokenizers are explicitly allowed, but asks maintainers
to clarify which normalizations are legal. It calls out NFKC as already lossy,
and flags lowercasing/case folding as an unresolved boundary.

Conservative lane:

- Train a new tokenizer on the same raw docs.
- Keep decoding lossless.
- If we transform text, make it bijective and keep exact original-byte sidecars.

Risky lane:

- Lowercasing, case folding, accent stripping, or any many-to-one transform
  without a reversible side channel.
- Any BPB conversion that uses average bytes/token instead of exact byte counts.

## What Competitors Are Using

Merged leaderboard / older strong records:

- Baseline: SentencePiece BPE 1024.
- Main merged SOTA family before CaseOps: SP4096 and SP8192 SentencePiece BPE.
- The official leaderboard currently shows SP8192 dominating the transformer
  merged records, with SP4096 still important in the April 1 stack.

Current public top transformer/HRC records:

- PR #1729: lossless CaseOps tokenizer/data export, `sp8192_lossless_caps_caseops_v1_reserved`,
  exact validation byte sidecars.
- PR #1736: adopts the same CaseOps tokenizer and byte-sidecar accounting.
- PR #1787, PR #1797, and PR #1801 build on the CaseOps PR #1736/#1779/#1787
  line rather than inventing a new tokenizer.

Other current public branches:

- PR #1791: FLA/GatedDeltaNet branch reports SP8192 tokenizer.
- PR #1795: byte-level PPM mixture branch builds on an SP4096 neural stack and
  adds a byte-level adaptive mixture, with legality/ruling risk.
- PR #1578: custom casefold tokenizer reports a strong number, but it is exactly
  the kind of lowercasing/case-folding normalization that triggered issue #1604.

## Recommendation For Us

The safest high-leverage tokenizer lane is not a naive whole-word tokenizer.
Whole-word vocabularies reduce sequence length, but under sub-4MB they are brutal
because the embedding/output interface gets expensive and rare words fragment
badly unless the vocab is huge.

Better lanes:

1. CaseOps-v2 lossless tokenizer:
   Keep the reversible capitalization side channel, but retrain BPE with a
   sub-4-aware objective: reserve fewer/more controls, tune vocab size
   4096/6144/8192, and measure token fertility plus final artifact bytes.

2. Word-boundary-aware BPE:
   Train SentencePiece BPE/Unigram with stricter whitespace/word-start behavior
   while staying byte-fallback and lossless. This tests the user's "whole word"
   instinct without committing to a brittle word-level vocabulary.

3. Domain-aware reserved symbols:
   Add reversible/lossless operators for common web-text structure only if we
   can prove exact reconstruction and exact byte accounting. Good candidates:
   capitalization, repeated whitespace/newline runs, common URL/email patterns,
   and numeric formatting. Avoid lossy normalization.

4. Sub-4 shortlist:
   Use `VOCAB_SIZE=4096` and `6144` CaseOps/word-boundary variants first,
   because full 8192 embeddings are only affordable because we already have
   factored/tied embeddings. Test final artifact size, BPB, and step speed.

5. Sub-16 shortlist:
   Keep `8192` as the main lane and test only genuinely cleaner token streams,
   because top HRC competitors are already CaseOps-SP8192 and the 16MB lane can
   afford the larger lexical interface.

## Implemented Sweep Scaffold

Added `data/tokenizer_specs_lossless_caseops_sweep.json` as the first safe
tokenizer matrix:

- CaseOps BPE at vocab sizes 4096, 6144, and 8192;
- word-boundary-aware byte-fallback BPE at vocab sizes 4096, 6144, and 8192;
- CaseOps Unigram at vocab sizes 4096, 6144, and 8192.

Use the upstream CaseOps exporter for this sweep because that path already
understands the reversible CaseOps transform and exact validation byte sidecars:

```powershell
.\\.venv-cuda313\\Scripts\\python.exe upstream_records\\records\\track_10min_16mb\\2026-04-18_PR1626_CaseOps_Taper\\download_hf_docs_and_tokenize.py --repo-id willdepueoai/parameter-golf --remote-root datasets --output-root data\\caseops_tokenizer_sweep --tokenizer-config data\\tokenizer_specs_lossless_caseops_sweep.json --max-train-shards 1
.\\.venv-cuda313\\Scripts\\python.exe scripts\\summarize_tokenizer_export.py data\\caseops_tokenizer_sweep\\manifest.json
```

The local summary helper reports token fertility from the actual `.bin` token
counts and exact validation byte sidecars. On the existing CaseOps-SP8192
manifest it reports `47,853,344` validation tokens and `151,080,645` validation
bytes, or `3.157160` bytes/token.

Sources:

- https://github.com/openai/parameter-golf
- https://github.com/openai/parameter-golf/blob/main/data/README.md
- https://github.com/openai/parameter-golf/issues/1017
- https://github.com/openai/parameter-golf/issues/1604
- https://github.com/openai/parameter-golf/pull/1729
- https://github.com/openai/parameter-golf/pull/1736
- https://github.com/openai/parameter-golf/pull/1787
- https://github.com/openai/parameter-golf/pull/1791
- https://github.com/openai/parameter-golf/pull/1795
- https://github.com/openai/parameter-golf/pull/1797
- https://github.com/openai/parameter-golf/pull/1801

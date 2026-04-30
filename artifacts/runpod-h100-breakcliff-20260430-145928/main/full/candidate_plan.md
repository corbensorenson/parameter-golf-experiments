# 16MB Vocab-MoE Matrix

Generated: 2026-04-30 20:07:04
Iterations: 1000000
Validation tokens: 131072

| Candidate | Route | Vocab MoE | Notes |
|---|---|---|---|
| `break_best32k_d704e832_control` | `unique=8 depth=16 start=3 repeats=2` | `k=16 r=2 mode=hybrid layers=input,loop_first temp=1 prior_std=0 site=1/1 train_q=8 spike_k=0 ste=1 norm=1` | Control for the new pod: known best legal 1xH100 family, batch32k, q8 train/export, one core-attention block, LexLoRE, LQER r10/t20. |
| `break_lexlore_exit_rank4_warm_d704e768` | `unique=8 depth=16 start=3 repeats=2` | `k=16 r=4 mode=hybrid layers=input,loop_first,loop_last temp=1 prior_std=0.01 site=1/1 train_q=8 spike_k=0 ste=1 norm=1` | Write-side LexLoRE: lexical low-rank experts advise input, loop entry, and mirrored/exit-side state with a warmer rank-4 adapter. |
| `break_lexlore_spike32_top4_exit_d704e768` | `unique=8 depth=16 start=3 repeats=2` | `k=32 r=2 mode=hybrid layers=input,loop_first,loop_last temp=1 prior_std=0.01 site=1/1 train_q=8 spike_k=4 ste=1 norm=1` | Self-election LexLoRE: token experts make a sparse top-k choice instead of only dense prior mixing, including the write-side site. |
| `break_coreattn2_d704e768` | `unique=8 depth=16 start=3 repeats=2` | `k=16 r=2 mode=hybrid layers=input,loop_first temp=1 prior_std=0 site=1/1 train_q=8 spike_k=0 ste=1 norm=1` | Token-mixing test: keep attention alive in the first two recurrent core blocks instead of a mostly MLP-only middle. |
| `break_coreattn_all_d640e768` | `unique=8 depth=16 start=3 repeats=2` | `k=16 r=2 mode=hybrid layers=input,loop_first temp=1 prior_std=0 site=1/1 train_q=8 spike_k=0 ste=1 norm=1` | Strong token-mixing test: every recurrent core block keeps attention; body width is trimmed to keep the preflight likely cap-safe. |
| `break_i3l7r2_unique_loop_d640e768` | `unique=10 depth=20 start=3 repeats=2` | `k=16 r=2 mode=hybrid layers=input,loop_first temp=1 prior_std=0 site=1/1 train_q=8 spike_k=0 ste=1 norm=1` | More unique loop blocks rather than more repeats: tests the user's hypothesis that physical block diversity is the missing capacity. |
| `break_partial_tail_cycle_lidx_d704e768` | `unique=8 depth=19 start=3 repeats=2` | `k=16 r=2 mode=hybrid layers=input,loop_first temp=1 prior_std=0 site=1/1 train_q=8 spike_k=0 ste=1 norm=1` | HRC route evolution: full core twice, then only the semantic tail before mirror exit, with loop-index conditioning. |
| `break_prime_skip_superloop_d640e768` | `unique=8 depth=16 start=3 repeats=1` | `k=16 r=2 mode=hybrid layers=input,loop_first temp=1 prior_std=0 site=1/1 train_q=8 spike_k=0 ste=1 norm=1` | Prime-route HRC: same 16 virtual steps as i3l5r2, but traverses the five-block core by two coprime skip programs. |
| `break_bigram_sidefeat_d704e704` | `unique=8 depth=16 start=3 repeats=2` | `k=16 r=2 mode=hybrid layers=input,loop_first temp=1 prior_std=0 site=1/1 train_q=8 spike_k=0 ste=1 norm=1` | Cheap token side-channel: BigramHash feature injection on the HRC/LexLoRE spine, borrowed from leaderboard-compatible tricks. |

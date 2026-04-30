# 16MB Vocab-MoE Matrix

Generated: 2026-04-30 21:45:53
Iterations: 1000000
Validation tokens: 131072

| Candidate | Route | Vocab MoE | Notes |
|---|---|---|---|
| `break_precision_width_q16q8q8_d704e704_fixed` | `unique=8 depth=16 start=3 repeats=2` | `k=16 r=2 mode=hybrid layers=input,loop_first temp=1 prior_std=0 site=1/1 train_q=8 spike_k=0 ste=1 norm=1; bits=blocks.0.:16,blocks.1.:8,blocks.2.:8; widths=528,616,704,704,704,704,704,704` | Fixed train-time precision plus width ladder: same q16/q8/q8 IO tail and q8 core, but widths are valid for d704/11 heads. |

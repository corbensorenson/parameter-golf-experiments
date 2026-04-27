# Prior-Work Read: What To Steal Next

Date: 2026-04-27

This note maps the pieces of our HRC/VocabMoE work to established prior work
and records the concrete changes that should improve the next candidates.

## Related Work Map

- Universal Transformers:
  <https://arxiv.org/abs/1807.03819>
  Recurrent transformer blocks are established. The useful lesson is not merely
  "reuse a block"; it is that recurrence usually needs a step/depth signal and
  sometimes adaptive depth.
- ALBERT:
  <https://arxiv.org/abs/1909.11942>
  Factorized embeddings and cross-layer sharing are established. The relevant
  lesson is to separate vocabulary-interface width from hidden-state width.
- Subformer:
  <https://arxiv.org/abs/2101.00234>
  Sandwich-style sharing is established for generative transformers. This is the
  closest prior-art cousin to our mirrored IO-tail idea.
- Lessons on Parameter Sharing:
  <https://arxiv.org/abs/2104.06022>
  Sequence/cycle/cycle-rev sharing schedules are established. Cycle-rev is the
  main thing our current 16MB queue was missing.
- BitNet and BitNet b1.58:
  <https://arxiv.org/abs/2310.11453>
  <https://arxiv.org/abs/2402.17764>
  Training low-bit/ternary weights from scratch is established. Our distinct
  angle is selective low precision in a recurrent hourglass, not ternary itself.
- HAWQ / LSQ / GPTQ:
  <https://arxiv.org/abs/1905.03696>
  <https://arxiv.org/abs/1902.08153>
  <https://arxiv.org/abs/2210.17323>
  Mixed precision and quant-error correction should be sensitivity-aware. Our
  manual q-ladders are probably leaving quality on the table.
- MoE, Switch, BASE, Product-Key Memory:
  <https://arxiv.org/abs/1701.06538>
  <https://arxiv.org/abs/2101.03961>
  <https://arxiv.org/abs/2103.16716>
  <https://arxiv.org/abs/1907.05242>
  Expert routing and large neural memories are established. The lesson for our
  tiny model is to keep routing balanced, batched, and low-rank.
- Transformer FFNs as key-value memories:
  <https://arxiv.org/abs/2012.14913>
  The MLP is a memory surface. Starving the recurrent core of MLP width or
  lexical side information can hurt more than starving attention.

## Things We Were Probably Missing

1. Cycle-rev route ablations.
   Our HRC core has mostly tested forward cycle repeats. Prior work says reverse
   cyclic sharing can use parameters differently, so the 16MB queue now includes
   `transition_recursive_palindrome` rows:
   `leader_i3l3r2rev...` and `leader_i3l5r1rev...`.

2. Recurrence step signals in 16MB, not only sub-4.
   Universal-Transformer style recurrence benefits from timestep/depth
   information. We already have loop-index code, but the 16MB leaderboard-blend
   queue did not test it. It now includes a `loopidx` row.

3. Small per-depth relaxation of shared weights.
   Fully shared depth can underfit because every pass must use exactly the same
   Q/V transforms. We already had `DEPTH_LORA_RANK`; the queue now includes a
   rank-4 per-virtual-depth Q/V LoRA row.

4. Sensitivity-aware bit allocation.
   Our precision ladder is architectural, not measured. HAWQ/GPTQ imply the next
   serious quant step is a calibration probe that ranks block/module sensitivity
   and chooses q8/q6/q4/ternary from evidence instead of intuition.

5. Learned quantizer scales.
   LSQ suggests fixed absmean/absmax scales are probably suboptimal. The clean
   future implementation is learned per-row or per-group step size for q4/q6 and
   ternary train-time forward, then export those scales.

6. Expert balance.
   MoE prior work repeatedly fights expert collapse. Our spike/self-election
   rows needed a token-prior tie-breaker; future rows should log expert load
   entropy and add a tiny load-balancing loss or balanced assignment if hard
   routing wins early but collapses late.

## Promotion Logic

- If cycle-rev beats matching cycle at similar virtual depth, make
  `transition_recursive_palindrome` the default route family for 16MB and retest
  sub-4 i4/i5 shapes.
- If loop-index helps 16MB r3, promote it for all recurrent 16MB rows. If it
  only helps deeper repeats, gate it on effective recurrence depth.
- If depth LoRA wins, spend bytes there before dual-stream. It is a cheaper
  version of "same shared block, different pass identity."
- If all three lose, the high-probability miss is quantization sensitivity, not
  more route creativity.

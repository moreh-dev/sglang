"""gfx950 fused DSA indexer kernels.

Decode: four raw-HIP kernels replacing the 12-launch ROCm chain.  Enabled
wherever supported; the gate is ``utils.gfx950_fused_indexer_runtime_ok``, the
runtime wrapper ``srt.layers.attention.dsa.gfx950_fused_indexer``.

Prefill (k-pool indexer): ``fp8_mqa_topk``, the logits and the pooled
top-k without fp32 logits, behind ``srt.layers.attention.dsa.hip_indexer_topk``."""

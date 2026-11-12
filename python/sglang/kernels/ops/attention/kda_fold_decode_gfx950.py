# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash KDA decode on gfx950 with the f_b / g_b gate projections folded in.

One launch per layer runs, for one token per sequence: both gate mat-vecs (W_fb @ f_a and
W_gb @ g_a, fp32), the q/k/v short convolution and its state update, the bounded safe gate, the
delta-rule recurrence on the paged fp32 state, and the sigmoid-gated RMSNorm. The unfused chain
is fused_fg_b_proj + causal_conv1d_update + the recurrent decode kernel + o_norm.

The kernel is kda_fold_decode_gfx950_kernel.py (moreh-dev/rocm-aiter#17); it launches with the
constexprs and the `fused` config bucket that AITER's fused_recurrent_kda_packed_decode passes for
this call, plus the fold arguments.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from sglang.kernels.arch import current_arch


@lru_cache(maxsize=1)
def available() -> bool:
    """gfx950 only, with AITER's Gluon helpers importable (the kernel module imports them)."""
    if not torch.version.hip or not torch.cuda.is_available():
        return False
    hip = current_arch().hip
    if hip is None or hip.gcn_arch != "gfx950":
        return False
    try:
        from sglang.kernels.ops.attention import (  # noqa: F401
            kda_fold_decode_gfx950_kernel,
        )
    except ImportError:
        return False
    return True


def kda_fold_decode(
    *,
    mixed_qkv: torch.Tensor,
    f_a: torch.Tensor,
    g_a: torch.Tensor,
    w_fb: torch.Tensor,
    w_gb: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    out: torch.Tensor,
) -> torch.Tensor:
    """mixed_qkv [n, 3 * H * K] (pre-conv q|k|v rows), f_a / g_a [n, K], beta [n, H] (raw logits),
    w_fb / w_gb [H * K, K] (out x in), ssm_states [slots, H, V, K] fp32, cache_indices [n] with
    slot 0 the padding slot (its rows are skipped), conv_state [slots, 3 * H * K, >= 3],
    conv_weight [3, 4, H * K], out [n, H, V]. Updates ssm_states and conv_state in place."""
    from sglang.kernels.ops.attention.kda_fold_decode_gfx950_kernel import (
        fused_recurrent_kda_packed_decode_kernel,
        get_kda_config,
    )

    n = mixed_qkv.shape[0]
    HV, V, K = ssm_states.shape[1:]
    D = HV * K
    assert mixed_qkv.shape[1] == 3 * D and mixed_qkv.stride(1) == 1
    assert out.shape == (n, HV, V) and out.stride()[1:] == (V, 1)
    assert beta.shape == (n, HV) and beta.stride(1) == 1
    for a in (f_a, g_a):
        assert a.shape == (n, K) and a.stride(1) == 1 and a.dtype == mixed_qkv.dtype
    for w in (w_fb, w_gb):
        assert w.shape == (D, K) and w.is_contiguous() and w.dtype == mixed_qkv.dtype
    assert ssm_states.dtype == torch.float32
    assert ssm_states.stride()[1:] == (V * K, K, 1) and ssm_states.stride(0) % K == 0
    assert conv_weight.shape == (3, 4, D) and conv_weight.is_contiguous()
    assert conv_state.shape[1] == 3 * D and conv_state.shape[2] >= 3
    assert cache_indices.shape == (n,) and cache_indices.stride(0) == 1
    assert A_log.numel() == HV and dt_bias.numel() == D
    state_indices = cache_indices.unsqueeze(1)
    cfg = get_kda_config(1, n, HV, K, V, fused=True)
    slot_rows = ssm_states.stride(0) // K
    rows = ssm_states.shape[0] * slot_rows
    s = mixed_qkv.stride(0)
    fused_recurrent_kda_packed_decode_kernel[(n * HV,)](
        q_ptr=mixed_qkv,
        k_ptr=mixed_qkv[:, D:],
        v_ptr=mixed_qkv[:, 2 * D :],
        g_ptr=mixed_qkv,  # not read with FOLD_GATES
        beta_ptr=beta,
        A_log_ptr=A_log,
        dt_bias_ptr=dt_bias,
        o_ptr=out,
        state_ptr=ssm_states,
        state_out_ptr=ssm_states,
        cu_seqlens_ptr=None,
        state_indices_ptr=state_indices,
        num_accepted_ptr=None,
        conv_state_ptr=conv_state,
        conv_weight_ptr=conv_weight,
        out_gate_ptr=mixed_qkv,  # not read with FOLD_GATES
        norm_weight_ptr=norm_weight,
        lower_bound=lower_bound,
        norm_eps=norm_eps,
        T=1,
        stride_indices_seq=state_indices.stride(0),
        stride_q_token=s,
        stride_k_token=s,
        stride_v_token=s,
        stride_g_token=s,
        stride_beta_token=beta.stride(0),
        stride_o_token=out.stride(0),
        stride_og_token=s,
        stride_state_slot_rows=slot_rows,
        stride_state_out_slot_rows=slot_rows,
        state_rows=rows,
        state_out_rows=rows,
        stride_cs_slot=conv_state.stride(0),
        stride_cs_dim=conv_state.stride(1),
        stride_cs_pos=conv_state.stride(2),
        scale=K**-0.5,
        H=HV,
        HV=HV,
        K=K,
        V=V,
        BV=cfg["BV"],
        NUM_WARPS=cfg["num_warps"],
        SK=cfg["SK"],
        NUM_BUFFERS=cfg.get("num_buffers", 2),
        IS_VARLEN=False,
        IS_CONTINUOUS_BATCHING=True,
        IS_SPEC_DECODING=False,
        IS_BETA_HEADWISE=False,
        USE_INITIAL_STATE=True,
        STORE_FINAL_STATE=True,
        USE_QK_L2NORM_IN_KERNEL=True,
        USE_GATE_IN_KERNEL=True,
        HAS_DT_BIAS=True,
        USE_LOWER_BOUND=True,
        APPLY_BETA_SIGMOID=True,
        ALLOW_NEG_EIGVAL=False,
        INPLACE_FINAL_STATE=True,
        STATE_V_FIRST=True,
        USE_TDM_STORE=False,
        USE_TDM_LOAD=False,
        CACHE_STATE_UPDATES=False,
        PAD_SLOT_GUARD=True,
        USE_TDM_FUSED_LOAD=False,
        TDM_STORE_BUFS=2,
        W=4,
        USE_CONV=True,
        USE_RMS_GATE=True,
        RESIDENT_STATE=False,
        gate_in_ptr=f_a,
        out_gate_in_ptr=g_a,
        gate_weight_ptr=w_fb,
        out_gate_weight_ptr=w_gb,
        stride_gate_in_token=f_a.stride(0),
        stride_out_gate_in_token=g_a.stride(0),
        FOLD_GATES=True,
        num_warps=cfg["num_warps"],
        enable_fp_fusion=True,
    )
    return out

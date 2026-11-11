"""gfx950 correctness of the folded GLM-5.3-Flash KDA decode kernel.

One decode step of one KDA layer is computed three ways on the layouts SGLang uses
(q|k|v, beta and [f_a | g_a] as views of one fused projection output; conv pool
[slot, 3, dim] bf16; state pool [slot, H, V, K] fp32; slot 0 is the padding slot):

  chain  the unfused ROCm decode: fused_fg_b_proj bmm, causal_conv1d_update,
         fused_sigmoid_gating_delta_rule_update, FusedRMSNormGated;
  fold   kda_fold_decode, one kernel that also computes the f_b / g_b gates;
  ref    an fp32 PyTorch step.

The fold must be close to ref, no worse than chain, leave untouched pool slots
(and slot 0 under padding) bit-identical, replay in a CUDA graph as it runs eagerly,
and keep the recurrent state on ref over consecutive decode steps.
"""

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention import kda_fold_decode_gfx950 as fold_mod
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

if not fold_mod.available():
    pytest.skip("requires one gfx950 GPU with AITER", allow_module_level=True)

K = 128
W = 4
LOWER_BOUND = -5.0
EPS = 1e-5
SLOTS = 256
DEV = torch.device("cuda")

# (decode rows, padded rows): every batch bucket of the kernel, with and without
# padding rows.
SHAPES = [(1, 0), (2, 0), (3, 1), (4, 0), (8, 2), (16, 0), (33, 3), (64, 0), (96, 5)]
# heads per rank: GLM-5.3-Flash at TP4 and TP8
HEADS = [16, 8]


class _Layer:
    """One rank's KDA layer weights (random, model dtypes and layouts)."""

    def __init__(self, heads, seed=0):
        from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated

        g = torch.Generator(device=DEV).manual_seed(seed)
        dim = heads * K
        self.H, self.D = heads, dim
        self.fg_b = (
            torch.randn(2, dim, K, generator=g, device=DEV) * K**-0.5
        ).bfloat16()
        self.conv_w = torch.randn(3 * dim, W, generator=g, device=DEV) * 0.5
        self.A_log = torch.log(
            15 * torch.rand(1, 1, heads, 1, generator=g, device=DEV) + 1
        )
        self.dt_bias = torch.randn(dim, generator=g, device=DEV) * 0.1
        self.norm = FusedRMSNormGated(
            K, eps=EPS, activation="sigmoid", device=DEV, dtype=torch.bfloat16
        )
        with torch.no_grad():
            self.norm.weight.copy_(1 + 0.1 * torch.randn(K, generator=g, device=DEV))
        # the copy glm5_next._prepare_fold_decode hands to the kernel
        self.fold_conv_w = self.conv_w.view(3, dim, W).transpose(1, 2).contiguous()


def _inputs(layer, n, pad_rows, seed, idx_dtype=torch.int32, idx=None):
    g = torch.Generator(device=DEV).manual_seed(seed)
    heads, dim = layer.H, layer.D
    width = 3 * dim + heads + 2 * K  # one fused_qkvbfg_a_proj output row
    proj = torch.randn(n, width, generator=g, device=DEV).bfloat16()
    proj[:, 3 * dim : 3 * dim + heads] *= 2  # beta logits
    mixed_qkv, beta, fg_a = torch.split(proj, [3 * dim, heads, 2 * K], dim=-1)
    if idx is None:
        idx = (torch.randperm(SLOTS - 1, generator=g, device=DEV)[:n] + 1).to(idx_dtype)
        if pad_rows:
            idx[-pad_rows:] = 0  # CUDA-graph padding rows point at the reserved slot 0
    conv_pool = (
        0.5 * torch.randn(SLOTS, W - 1, 3 * dim, generator=g, device=DEV)
    ).bfloat16()
    ssm_pool = 0.1 * torch.randn(SLOTS, heads, K, K, generator=g, device=DEV)
    return mixed_qkv, beta, fg_a, idx, conv_pool, ssm_pool


def _chain(layer, mixed_qkv, beta, fg_a, idx, conv_pool, ssm_pool):
    """The unfused decode: glm5_next.forward + kda_backend.forward_decode on gfx950."""
    from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

    n = mixed_qkv.shape[0]
    forget_gate, g_proj = torch.bmm(
        fg_a.view(-1, 2, K).transpose(0, 1), layer.fg_b.transpose(-1, -2)
    )
    qkv = causal_conv1d_update(
        mixed_qkv,
        conv_pool.transpose(-1, -2),
        layer.conv_w,
        None,
        activation="silu",
        conv_state_indices=idx,
    )
    q, k, v = (
        t.unflatten(-1, (-1, K)).unsqueeze(0) for t in qkv.split([layer.D] * 3, dim=-1)
    )
    core = fused_sigmoid_gating_delta_rule_update(
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        q=q,
        k=k,
        v=v,
        a=forget_gate,
        b=beta.unsqueeze(0),
        initial_state_source=ssm_pool,
        initial_state_indices=idx,
        cu_seqlens=torch.arange(n + 1, device=DEV, dtype=torch.int32),
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=True,
        lower_bound=LOWER_BOUND,
    )
    out = layer.norm(core, g_proj.unflatten(-1, (-1, K)))
    return out.reshape(n, layer.H, K)


def _fold(layer, mixed_qkv, beta, fg_a, idx, conv_pool, ssm_pool, out=None):
    """kda_backend._forward_decode_fold."""
    if out is None:
        out = mixed_qkv.new_empty((mixed_qkv.shape[0], layer.H, K))
    fold_mod.kda_fold_decode(
        mixed_qkv=mixed_qkv,
        f_a=fg_a[:, :K],
        g_a=fg_a[:, K:],
        w_fb=layer.fg_b[0],
        w_gb=layer.fg_b[1],
        beta=beta,
        A_log=layer.A_log.reshape(-1),
        dt_bias=layer.dt_bias,
        lower_bound=LOWER_BOUND,
        ssm_states=ssm_pool,
        cache_indices=idx,
        conv_state=conv_pool.transpose(-1, -2),
        conv_weight=layer.fold_conv_w,
        norm_weight=layer.norm.weight,
        norm_eps=EPS,
        out=out,
    )
    return out


def _ref(layer, mixed_qkv, beta, fg_a, idx, conv_pool, ssm_pool):
    """fp32 PyTorch decode step; padded rows (slot 0) are skipped."""
    n, heads, dim = mixed_qkv.shape[0], layer.H, layer.D
    out = torch.zeros(n, heads, K, device=DEV)
    for i in range(n):
        s = int(idx[i])
        if s <= 0:
            continue
        x = mixed_qkv[i].float()
        win = torch.cat([conv_pool[s].float().t(), x[:, None]], dim=1)  # oldest first
        y = F.silu((win * layer.conv_w).sum(-1))
        conv_pool[s] = win[:, 1:].t().to(conv_pool.dtype)
        q, k, v = (t.view(heads, K) for t in y.split(dim))
        q = F.normalize(q, dim=-1, eps=1e-6) * K**-0.5
        k = F.normalize(k, dim=-1, eps=1e-6)
        gate = (layer.fg_b[0].float() @ fg_a[i, :K].float()).view(heads, K)
        out_gate = (layer.fg_b[1].float() @ fg_a[i, K:].float()).view(heads, K)
        decay = torch.exp(
            LOWER_BOUND
            * torch.sigmoid(
                layer.A_log.view(heads, 1).exp() * (gate + layer.dt_bias.view(heads, K))
            )
        )
        b = torch.sigmoid(beta[i].float())
        h = ssm_pool[s].transpose(-1, -2) * decay[:, :, None]  # [H, K, V]
        v_new = v - torch.einsum("hk,hkv->hv", k, h)
        h = h + b[:, None, None] * k[:, :, None] * v_new[:, None, :]
        ssm_pool[s] = h.transpose(-1, -2)
        o = torch.einsum("hk,hkv->hv", q, h)
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS)
        out[i] = o * layer.norm.weight.float() * torch.sigmoid(out_gate)
    return out


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _run(fn, layer, inputs):
    mixed_qkv, beta, fg_a, idx, conv0, ssm0 = inputs
    conv, ssm = conv0.clone(), ssm0.clone()
    out = fn(layer, mixed_qkv, beta, fg_a, idx, conv, ssm)
    torch.cuda.synchronize()
    return out, conv, ssm


def _check_step(layer, n, pad_rows, idx_dtype=torch.int32, seed=1):
    inputs = _inputs(layer, n, pad_rows, seed, idx_dtype)
    idx, conv0, ssm0 = inputs[3], inputs[4], inputs[5]
    ref_out, ref_conv, ref_ssm = _run(_ref, layer, inputs)
    chain_out, _, chain_ssm = _run(_chain, layer, inputs)
    fold_out, fold_conv, fold_ssm = _run(_fold, layer, inputs)
    real = idx > 0
    rows = idx[real].long()

    fold_o = _rel(fold_out[real], ref_out[real])
    chain_o = _rel(chain_out[real], ref_out[real])
    fold_s = _rel(fold_ssm[rows], ref_ssm[rows])
    chain_s = _rel(chain_ssm[rows], ref_ssm[rows])
    assert torch.isfinite(fold_out[real].float()).all()
    assert fold_o < 1e-2 and fold_s < 1e-3, (fold_o, fold_s)
    # fp32 gates and an fp32 norm input: the fold may only be closer to ref
    assert fold_o <= max(1.5 * chain_o, 2e-3), (fold_o, chain_o)
    assert fold_s <= max(1.5 * chain_s, 1e-4), (fold_s, chain_s)
    # the conv state is a bf16 pass-through of the inputs
    assert torch.equal(fold_conv[rows], ref_conv[rows])
    # untouched slots, and slot 0 under padding, keep their bits
    untouched = torch.ones(SLOTS, dtype=torch.bool, device=DEV)
    untouched[rows] = False
    assert torch.equal(fold_conv[untouched], conv0[untouched])
    assert torch.equal(fold_ssm[untouched], ssm0[untouched])


@pytest.fixture(scope="module", params=HEADS)
def layer(request):
    torch.manual_seed(0)
    return _Layer(request.param)


@pytest.mark.parametrize("n, pad_rows", SHAPES)
def test_fold_matches_reference_and_chain(layer, n, pad_rows):
    _check_step(layer, n, pad_rows)


def test_int64_cache_indices(layer):
    _check_step(layer, 8, 1, idx_dtype=torch.int64)


def test_cuda_graph_replay_matches_eager(layer):
    n = 8
    mq, be, fa, idx, conv, ssm = _inputs(layer, n, 1, seed=3)
    s_mq, s_be, s_fa, s_idx = mq.clone(), be.clone(), fa.clone(), idx.clone()
    conv_g, ssm_g = conv.clone(), ssm.clone()
    out = mq.new_empty((n, layer.H, K))
    _fold(layer, s_mq, s_be, s_fa, s_idx, conv_g, ssm_g, out)  # JIT outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _fold(layer, s_mq, s_be, s_fa, s_idx, conv_g, ssm_g, out)
    # new inputs, indices and pools copied into the captured buffers
    mq2, be2, fa2, idx2, conv2, ssm2 = _inputs(layer, n, 2, seed=4)
    for dst, src in (
        (s_mq, mq2),
        (s_be, be2),
        (s_fa, fa2),
        (s_idx, idx2),
        (conv_g, conv2),
        (ssm_g, ssm2),
    ):
        dst.copy_(src)
    graph.replay()
    conv_e, ssm_e = conv2.clone(), ssm2.clone()
    out_e = _fold(layer, mq2, be2, fa2, idx2, conv_e, ssm_e)
    torch.cuda.synchronize()
    real = idx2 > 0
    assert torch.equal(out[real], out_e[real])
    assert torch.equal(ssm_g, ssm_e)
    assert torch.equal(conv_g, conv_e)


def test_state_follows_reference_over_decode_steps(layer):
    """16 consecutive decode steps of the same sequences: the state must not drift."""
    n, steps = 4, 16
    mq, be, fa, idx, conv0, ssm0 = _inputs(layer, n, 0, seed=7)
    ref_conv, ref_ssm = conv0.clone(), ssm0.clone()
    fold_conv, fold_ssm = conv0.clone(), ssm0.clone()
    rows = idx.long()
    for step in range(steps):
        if step:
            mq, be, fa = _inputs(layer, n, 0, seed=7 + step, idx=idx)[:3]
        ref_out = _ref(layer, mq, be, fa, idx, ref_conv, ref_ssm)
        fold_out = _fold(layer, mq, be, fa, idx, fold_conv, fold_ssm)
        torch.cuda.synchronize()
        assert _rel(fold_out, ref_out) < 1e-2, step
        assert torch.equal(fold_conv[rows], ref_conv[rows]), step
    assert _rel(fold_ssm[rows], ref_ssm[rows]) < 1e-3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

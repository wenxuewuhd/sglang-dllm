"""Pure-torch CPU reference for one GLM-5.3-Flash DSA (NoPE MLA) layer.

Absorbed form, matching what the NPU path computes:
    q_nope_out[t,h] = q[t,h] @ w_kc[h]            (256 -> 512)
    score[t,h,j]    = q_nope_out[t,h] . kv_a[j]   ( == q[t,h] . k_nope[j,h] )
    o512[t,h]       = softmax_j(score) @ kv_a[J]
    out[t,h]        = o512[t,h] @ w_vc[h]         (512 -> 256)
It is algebraically the expanded HF form; stage A cross-checks that against
`Glm5NextTextAttention.forward` itself.

`dtype` rounds every module boundary to that dtype while keeping the arithmetic
in fp32 -- so dtype=float32 gives R32 and dtype=bfloat16 gives R16, the two
references the ACCEPTANCE two-reference method wants.
"""
from __future__ import annotations

import torch

EPS = 1e-5


def _rms(x, w, eps=EPS):
    v = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + eps) * w


class LayerRef:
    def __init__(self, sh, layer, cfg, tp_size=1, tp_rank=0):
        p = f"model.language_model.layers.{layer}.self_attn."
        g = lambda n: sh.get(p + n).float()
        nh = cfg.num_attention_heads
        lh = nh // tp_size
        qk = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        kvo = cfg.qk_nope_head_dim + cfg.v_head_dim
        h0, h1 = tp_rank * lh, (tp_rank + 1) * lh
        self.q_a = g("q_a_proj.weight")
        self.q_a_ln = g("q_a_layernorm.weight")
        self.q_b = g("q_b_proj.weight")[h0 * qk : h1 * qk]
        self.kv_a = g("kv_a_proj_with_mqa.weight")
        self.kv_a_ln = g("kv_a_layernorm.weight")
        self.kv_b = g("kv_b_proj.weight")[h0 * kvo : h1 * kvo]
        # o_proj is row-parallel: rank `tp_rank` owns the input columns for its
        # heads and produces a *partial* sum (reduce_results=False; the model
        # all-reduces after the layer).  The reference must be the same partial.
        self.o = g("o_proj.weight")[:, h0 * cfg.v_head_dim : h1 * cfg.v_head_dim]
        self.nh = lh
        self.nope = cfg.qk_nope_head_dim
        self.vd = cfg.v_head_dim
        self.lora = cfg.kv_lora_rank
        self.scaling = (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) ** -0.5

    def _cast(self, dtype):
        r = lambda t: t.to(dtype).float()
        w_kc, w_vc = (
            self.kv_b.unflatten(0, (-1, self.nope + self.vd))
            .split([self.nope, self.vd], dim=1)
        )
        return dict(
            q_a=r(self.q_a), q_a_ln=r(self.q_a_ln), q_b=r(self.q_b),
            kv_a=r(self.kv_a), kv_a_ln=r(self.kv_a_ln),
            w_kc=r(w_kc), w_vc=r(w_vc.transpose(1, 2)), o=r(self.o),
        )

    def kv_latent(self, x_f32, dtype):
        """kv_a_layernorm(kv_a_proj(x)) -- what lands in the KV cache."""
        W = self._cast(dtype)
        r = lambda t: t.to(dtype).float()
        x = r(x_f32)
        return r(_rms(r(x @ W["kv_a"].T), W["kv_a_ln"]))

    def q_absorbed(self, x_rows_f32, dtype):
        """q_nope_out for a set of rows: [R, nh, kv_lora_rank]."""
        W = self._cast(dtype)
        r = lambda t: t.to(dtype).float()
        x = r(x_rows_f32)
        qa = r(_rms(r(x @ W["q_a"].T), W["q_a_ln"]))
        q = r(qa @ W["q_b"].T).view(-1, self.nh, self.nope)
        return r(torch.bmm(q.transpose(0, 1), W["w_kc"]).transpose(0, 1))

    def attend(self, q_nope_out, kv, idx_rows, dtype):
        """idx_rows: list of LongTensor token ids per row.  Returns [R, hidden]."""
        W = self._cast(dtype)
        r = lambda t: t.to(dtype).float()
        R = q_nope_out.shape[0]
        o512 = torch.empty(R, self.nh, self.lora, dtype=torch.float32)
        for i in range(R):
            ids = idx_rows[i]
            k = kv[ids]                                    # [L, 512]
            s = (q_nope_out[i] @ k.T) * self.scaling       # [nh, L]
            pr = torch.softmax(s, dim=-1)
            o512[i] = pr @ k
        o512 = r(o512)
        ov = r(torch.bmm(o512.transpose(0, 1), W["w_vc"]).transpose(0, 1))
        return r(ov.reshape(R, self.nh * self.vd) @ W["o"].T)


def rel(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    return ((a - b).norm() / b.norm()).item()


def cos(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    return (a @ b / (a.norm() * b.norm())).item()

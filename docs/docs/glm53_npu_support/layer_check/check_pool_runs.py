#!/usr/bin/env python
"""Padded `visible_pool_runs` must segment identically to the nonzero form.

The unpadded form compacts run starts with `nonzero()`, whose output length is the
run count -- data-dependent, so the shape is dynamic and a graph capture rejects it.
That is the last thing keeping prefill out of the graph on this path.

The padded form scatters each boundary into the slot its rank names, so the output
length is a fixed `max_runs`, and the unused slots hold empty runs. On device those
are a no-op: `probe/p6_a1_padded_runs.py` measured bit-identical indexer output when
a 128-run segmentation is padded by 1, 8 and 128 empty runs.

This checks the segmentation itself, on CPU, against the *shipped* source. Two
things have to hold: the leading `max_runs` entries must match the exact form, and
`max_visible_pool_runs` must actually bound the run count -- if it under-counts, runs
are silently dropped and the tail of the batch stops being attended to.

    $ROOT/.venv-ref/bin/python check_pool_runs.py
"""

import os
import ast
import pathlib
import random

import torch

# Root of the workspace holding env/, the goldens and the sibling checkouts.
_GLM53_ROOT = os.environ.get("GLM53_ROOT") or os.environ.get("GLM53_WORKSPACE") or ""

MODULE = pathlib.Path(
    f"{_GLM53_ROOT}/sglang-dllm/python/sglang/srt/"
    "hardware_backend/npu/attention/kpool_indexer_npu.py"
)
KPOOL = 4


def shipped(*names):
    """Pull functions out of the module without importing it (it needs torch_npu)."""
    src = MODULE.read_text()
    ns = {"torch": torch}
    tree = ast.parse(src)
    for want in names:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == want:
                exec(ast.get_source_segment(src, node), ns)  # noqa: S102
                break
        else:
            raise SystemExit(f"{want} not found; was it renamed?")
    return [ns[n] for n in names]


def rows_for(batch, q_lens, prefixes):
    """The per-row key count and owning request an extend forward would produce."""
    rows, req = [], []
    for i in range(batch):
        if q_lens[i] == 0:
            continue
        seq = prefixes[i] + q_lens[i]
        rows.append(torch.arange(seq - q_lens[i] + 1, seq + 1))
        req.append(torch.full((q_lens[i],), i))
    if not rows:
        return torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64)
    return torch.cat(rows) // KPOOL, torch.cat(req)


def main() -> int:
    visible_pool_runs, max_visible_pool_runs = shipped(
        "visible_pool_runs", "max_visible_pool_runs"
    )
    random.seed(0)
    checked = worst_slack = 0
    tightest = 10**9

    configs = [
        (random.randint(1, 16),
         [random.choice([0, 1, random.randint(1, 300)]) for _ in range(16)],
         [random.randint(0, 4000) for _ in range(16)])
        for _ in range(2000)
    ]
    # The shapes the deployment actually produces, including a single 8192-row chunk.
    configs += [(1, [8192], [0]), (1, [8192], [24576]), (16, [512] * 16, [1000] * 16),
                (128, [64] * 128, [500] * 128), (1, [1], [0]), (8, [0] * 8, [0] * 8)]

    for batch, q_lens, prefixes in configs:
        q_lens, prefixes = q_lens[:batch], prefixes[:batch]
        pool_lens, req_index = rows_for(batch, q_lens, prefixes)
        n_rows = pool_lens.shape[0]
        exact = visible_pool_runs(pool_lens, req_index)
        n_runs = exact[0].shape[0]

        bound = max_visible_pool_runs(n_rows, batch, KPOOL)
        assert n_runs <= bound, (
            f"bound {bound} under-counts {n_runs} runs: batch={batch} "
            f"q_lens={q_lens[:8]} prefixes={prefixes[:8]}"
        )
        tightest = min(tightest, bound - n_runs)
        worst_slack = max(worst_slack, bound - n_runs)

        padded = visible_pool_runs(pool_lens, req_index, max_runs=bound)
        for k, (a, b) in enumerate(zip(exact, padded)):
            assert b.shape[0] == bound, f"padded output is {b.shape[0]}, want {bound}"
            assert torch.equal(a, b[:n_runs]), (
                f"output {k} differs in the first {n_runs} runs: batch={batch} "
                f"q_lens={q_lens[:8]}"
            )
        # Every padded slot must be an empty run, or the indexer would attend to it.
        ends = padded[0]
        starts = torch.cat([torch.zeros(1, dtype=ends.dtype), ends[:-1]])
        assert torch.equal(ends[n_runs:], starts[n_runs:]), "padding is not empty"
        checked += 1

    print(f"{checked} configs: padded == exact on the real runs, padding is empty, "
          f"and the bound holds (slack {tightest}..{worst_slack})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

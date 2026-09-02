#!/usr/bin/env python
"""`_extend_rows` on device must agree with the host-side loop it replaced.

The old form built the two row tensors in Python every forward -- a loop over the
batch, an `arange` per request, a `cat`, and a host-to-device copy -- and a graph
capture would have baked one forward's values in permanently. The new form is pure
tensor arithmetic with shapes fixed by the row count.

This runs on CPU against the *shipped* source (extracted from the module rather than
copied here, so the two cannot drift), which is the whole point: the rewrite is index
arithmetic and is exactly reproducible without an NPU. What it does NOT establish is
that the ops are capturable or fast on device -- that needs a machine.

    $ROOT/.venv-ref/bin/python check_extend_rows.py
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


def shipped_extend_rows():
    """Pull `_extend_rows` out of the module without importing it (needs torch_npu)."""
    src = MODULE.read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "_extend_rows":
            body = ast.get_source_segment(src, node)
            code = "\n".join(
                line[4:] if line.startswith("    ") else line
                for line in body.splitlines()
            )
            ns = {"torch": torch}
            exec(code, ns)  # noqa: S102 -- reading our own source, on purpose
            return ns["_extend_rows"]
    raise SystemExit("_extend_rows not found; did the function get renamed?")


def host_side_loop(extend_seq_lens, seq_lens, batch_size):
    """The implementation this replaced, verbatim."""
    rows, req_index = [], []
    for i in range(batch_size):
        q_len = int(extend_seq_lens[i])
        if q_len == 0:
            continue
        seq_len = int(seq_lens[i])
        rows.append(torch.arange(seq_len - q_len + 1, seq_len + 1))
        req_index.append(torch.full((q_len,), i))
    if not rows:
        return torch.zeros(0, dtype=torch.int32), torch.zeros(0, dtype=torch.int32)
    return torch.cat(rows).to(torch.int32), torch.cat(req_index).to(torch.int32)


def main() -> int:
    new = shipped_extend_rows()
    random.seed(0)
    torch.manual_seed(0)
    checked = 0

    for trial in range(3000):
        batch = random.randint(1, 24)
        # Zero-length requests are where the two forms could disagree: the old loop
        # skipped them with `continue`, and a starts-based segmentation would land on
        # the empty request instead of the next one.
        q = [random.choice([0, 0, 1, random.randint(1, 40)]) for _ in range(batch)]
        s = [qi + random.randint(0, 5000) for qi in q]
        q_t = torch.tensor(q, dtype=torch.int64)
        s_t = torch.tensor(s, dtype=torch.int64)
        want = host_side_loop(q_t, s_t, batch)
        got = new(q_t, s_t, int(q_t.sum()))
        assert torch.equal(want[0], got[0]), f"rows differ at trial {trial}: q={q} s={s}"
        assert torch.equal(want[1], got[1]), f"index differs at trial {trial}: q={q}"
        checked += 1

    # The shapes the deployment actually produces.
    for batch, q_len in ((1, 8192), (16, 512), (128, 64), (1, 1), (128, 1)):
        q_t = torch.full((batch,), q_len, dtype=torch.int64)
        s_t = q_t + 20000
        want = host_side_loop(q_t, s_t, batch)
        got = new(q_t, s_t, int(q_t.sum()))
        assert torch.equal(want[0], got[0]) and torch.equal(want[1], got[1]), (
            f"deployment shape batch={batch} q_len={q_len}"
        )
        checked += 1

    q_t = torch.zeros(8, dtype=torch.int64)
    s_t = torch.full((8,), 100, dtype=torch.int64)
    want = host_side_loop(q_t, s_t, 8)
    got = new(q_t, s_t, 0)
    assert torch.equal(want[0], got[0]) and torch.equal(want[1], got[1]), "all-empty"
    checked += 1

    print(f"{checked} cases: device-side form == host-side loop, element for element")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

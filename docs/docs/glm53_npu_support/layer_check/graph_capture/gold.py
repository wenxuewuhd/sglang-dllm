"""Score a *graph replay* output against the layer_check two-reference golden.

Point: 'replay == eager' only says the graph is self-consistent.  This says the
number coming out of the replayed graph is still the right number.
"""
from harness import Case, Result, check, report


def slice_case(case: Case, n: int) -> Case:
    """First n rows of every per-token tensor; scalars/weights untouched."""
    def cut(d):
        out = {}
        for k, v in d.items():
            out[k] = v[:n] if (v.dim() >= 1 and v.shape[0] >= n) else v
        return out
    return Case(case.name, cut(case.inputs), cut(case.ref_fp32), cut(case.ref_bf16),
                case.meta)


def score(title, case: Case, n: int, candidate: dict, extra: str = "") -> int:
    sub = slice_case(case, n)
    keep = {k: v for k, v in candidate.items() if k in sub.ref_fp32}
    sub = Case(sub.name, sub.inputs, {k: sub.ref_fp32[k] for k in keep},
               {k: sub.ref_bf16[k] for k in keep}, sub.meta)
    return report(title, check(sub, keep), extra)

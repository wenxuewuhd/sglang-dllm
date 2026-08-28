# Acceptance criteria

Two different gates, because two different kinds of operator.

---

## A. Operators that return floating-point values — the two-reference method

**Do not use a fixed relative-error threshold.** A measured counter-example from this
project: the KDA layer-0 golden at `seq=64` has an **fp32-vs-bf16 relative error of
1.06e-2**. A 1e-3 gate would reject a bit-perfect bf16 implementation of that layer, and
a 1e-1 gate would accept a badly broken one somewhere else. The right threshold is
per-case and it is measurable.

The method (implemented in [`reference/tolerance.py`](reference/tolerance.py), and
already used elsewhere in this project — see
`docs/docs/glm53_npu_support/tools/golden_kda.py:11`):

1. Build the inputs for the case in **fp32**. Run the pure-torch reference. Call the
   result `R32`.
2. Cast the same inputs to **bf16**. Run the same reference. Call it `R16`.
3. `floor = rel_err(R16, R32)` — this case's **noise floor**. It is the error you get
   from bf16 rounding alone, with the arithmetic held fixed.
4. Run the candidate on bf16 inputs. It is **accepted** when
   `rel_err(candidate, R32) <= floor * SLACK`.

`rel_err(a, b) = ||a - b||₂ / ||b||₂`, computed in fp64 so the metric is not itself the
noise.

### `SLACK` and `ABS_MIN`

* `SLACK` defaults to **2.0** (`GLM53_TOL_SLACK`). It exists because a candidate may
  order its reductions differently from the reference and legitimately land just the
  other side of the floor. Keep it small and explicit; raising it needs a recorded
  reason.
* `ABS_MIN` defaults to **1e-6** (`GLM53_TOL_ABS_MIN`). When `R32` and `R16` happen to
  agree exactly the measured floor is 0, which would demand bit-identity. This absolute
  term keeps such cases sane.

Every failure message prints the measured floor, the slack and the achieved error, so a
borderline result is diagnosable without re-running.

---

## B. Operators that return indices — set equality, never index equality

This applies to **OP-1** (`kpool_topk_transform`).

Index-by-index equality is the wrong gate for three independent reasons:

1. The order of the selected groups is **not part of the contract** — the CUDA kernel
   emits radix-scan completion order above the fast-path threshold and ascending id
   order below it.
2. **Ties at the k-th boundary legitimately differ.** When several groups share the
   k-th largest score, any of them is a correct choice.
3. **There is no CUDA device on this machine**, so nobody can produce reference indices
   from the original kernel even if we wanted to compare them.

The gate is therefore, per row:

| # | Check | Strictness |
|---|---|---|
| 1 | each history pool expands contiguously: `out[rank*P + s]` is the `s`-th token of one group id | exact |
| 2 | the **sorted multiset of selected scores** equals the reference's | **exact**, element by element |
| 3 | no duplicate group ids; every id in `[0, length)` | exact |
| 4 | the tail columns and the `-1` padding columns | exact, index by index |

Check 2 is the interesting one. The multiset of selected *values* is uniquely determined
by `k` and the score vector — ties or no ties — so this is an **exact** test that is
nevertheless blind to which tied index was taken. The weaker "sum of selected scores"
form asked for in the brief is implied by it; we use the multiset because a sum can
coincide by accident.

There is **no floating-point tolerance anywhere in OP-1**: the operator returns integers,
and the scores are only ever compared for equality, never for closeness.

---

## C. Regression gates (exact, not tolerance-based)

For the two EXTEND requests, the strongest and cheapest evidence is that the old path
did not move at all:

* **OP-2**: with `norm_mode = 0` and `norm_bias` absent, `Compressor`'s `cmp_kv` and
  `state_cache` must be **byte-identical** to the currently shipped build on the
  DeepSeek-V4 path.
* **OP-3**: `rope_dim = 64` must behave exactly as it does today.

---

## D. End-to-end gate (ours, not yours)

Once the operators land we run the GLM-5.3-Flash server and compare GPQA-Diamond against
the recorded baseline. For calibration, the numbers already recorded in
`docs/docs/glm53_npu_support/PLAN.md:644-647` for DeepSeek-V4 on this machine are
73.74 % (SD 1.82 pp over 3 rounds) and 73.23 % after a rebase — a ±0.5 pp move is noise
at that sample size. This gate is listed only so you know what "done" looks like
downstream; it is not something you can run.

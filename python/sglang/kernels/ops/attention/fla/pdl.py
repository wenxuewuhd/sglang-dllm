"""PDL intrinsics that survive an AST walk on non-CUDA Triton backends.

Triton resolves attribute chains while walking a kernel's AST, before any
constexpr folding, so wrapping `tl.extra.cuda.gdc_wait()` in `if USE_GDC:` is
not enough: on a backend whose `triton.language.extra` has no `cuda` (Ascend's
triton_ascend, for one) the kernel fails to compile with
`AttributeError: module 'triton.language.extra' has no attribute 'cuda'`
even when USE_GDC is False. Calling through these wrappers keeps the attribute
out of the caller's AST; the no-op definitions are what the walker finds.
"""

import triton
import triton.language as tl

HAS_GDC = hasattr(getattr(tl.extra, "cuda", None), "gdc_wait")

if HAS_GDC:

    @triton.jit
    def gdc_wait():
        tl.extra.cuda.gdc_wait()

    @triton.jit
    def gdc_launch_dependents():
        tl.extra.cuda.gdc_launch_dependents()

else:

    @triton.jit
    def gdc_wait():
        pass

    @triton.jit
    def gdc_launch_dependents():
        pass

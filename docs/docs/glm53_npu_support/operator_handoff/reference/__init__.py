"""Pure-torch reference implementations for the GLM-5.3-Flash Ascend operator handoff.

Every function in this package is CPU-only, depends on nothing but ``torch``, and IS
the definition of correct for the corresponding operator spec under ``../specs/``.

Nothing here is performance code. It is deliberately written the slow, obvious way so
that it can be read as a specification.
"""

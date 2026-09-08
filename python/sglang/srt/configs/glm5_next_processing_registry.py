"""Register GLM-5.3-Flash's processor classes with the transformers Auto* registries.

The checkpoint's `processor_config.json` names `Glm5NextProcessor` and
`Glm5NextImageProcessor`, which ship only on transformers `main` -- no released
version has them (checked through v5.16.0). Without them `AutoProcessor` falls
back to a bare tokenizer *without raising*, so images are accepted and dropped:
the `<|image|>` placeholder never expands and every image yields the same answer.

Substituting the Glm4v/Glm46V family is not equivalent. Their `smart_resize`
rounds rather than ceils, stretches instead of zero-padding, and ignores
`min_image_tokens`/`max_image_tokens`; a 570x380 image gets 280 tokens instead
of 294. The sources below are the upstream implementations, vendored with their
relative imports rewritten, in the same way `deepseek_ocr.py` vendors its own.
Drop this file once a transformers release carries `glm5_next`.
"""

import importlib.util
import logging

from transformers import AutoImageProcessor, AutoProcessor, AutoVideoProcessor

from sglang.srt.configs.glm5_next import Glm5NextConfig
from sglang.srt.configs.image_processing_glm5_next import Glm5NextImageProcessor
from sglang.srt.configs.processing_glm5_next import Glm5NextProcessor
from sglang.srt.configs.video_processing_glm5_next import Glm5NextVideoProcessor

logger = logging.getLogger(__name__)

# The upstream commit these three files were copied from. Keep it in step with the
# per-file provenance headers; it is what a maintainer needs to diff against.
VENDORED_FROM_TRANSFORMERS_COMMIT = "3197876523ebee6adc7877b1853fc4932bef0f9f"


def _is_redundant() -> bool:
    """True once transformers ships ``glm5_next`` itself."""
    return importlib.util.find_spec("transformers.models.glm5_next") is not None


def _warn_if_redundant() -> None:
    """Say so, loudly, on the day this shim stops doing anything.

    What actually happens to each registration on that day is not the same for
    all three, and the difference is why two of them are skipped below. Measured
    against transformers 5.12.1 by registering a class into ``_extra_content``
    under a name transformers also ships natively:

    * ``AutoProcessor`` is keyed on the *config class object*. Once transformers
      ships ``glm5_next``, ``utils/hf_transformers/common.py`` can no longer
      register our config (``exist_ok=False`` raises, and the loop around it
      swallows that), so the config in hand is transformers' ``Glm5NextConfig``
      -- a different object that merely shares our class's name. It is not the
      key we registered under, so ``PROCESSOR_MAPPING`` returns transformers'
      processor and our entry is unreachable. Unreachable, that is, only if
      transformers' config really did win, which this module cannot observe --
      so this registration still runs unconditionally.

    * ``AutoImageProcessor`` and ``AutoVideoProcessor`` are keyed on the config
      too, but that is not how the checkpoint reaches them: its
      processor_config.json names them by *class name*, and the Auto classes
      resolve those names by scanning class names across the mappings. Being
      keyed on our config therefore does not make them unreachable, and for the
      image processor it does the opposite:
      ``image_processing_auto.get_image_processor_class_from_name`` scans
      ``_extra_content`` *before* ``IMAGE_PROCESSOR_MAPPING_NAMES``, so this
      vendored copy would shadow transformers' own ``Glm5NextImageProcessor``
      for every by-name lookup -- a stale copy silently preferred over the
      maintained one, inside transformers' own processor.
      ``video_processing_auto.video_processor_class_from_name`` happens to scan
      the native mapping first, so the video registration goes inert instead;
      that ordering is an implementation detail, not a guarantee.

    Both name-keyed registrations are therefore skipped once ``glm5_next`` is
    importable. That cannot resurrect the silent image-dropping this file exists
    to prevent: with the model shipped, the same by-name lookups resolve to
    transformers' own classes with nothing in ``_extra_content`` at all (also
    measured). The fail-safe argument applies only to ``AutoProcessor``, whose
    outcome depends on which config class won, and that one keeps running.

    This is a warning and not an exception on purpose: being redundant is not a
    failure, it is the good outcome, and turning a transformers upgrade into a
    server that will not start would be a worse bug than the stale code.
    """
    if not _is_redundant():
        return
    logger.warning(
        "transformers now ships glm5_next natively, so "
        "sglang/srt/configs/glm5_next_processing_registry.py and the three vendored "
        "*_glm5_next.py processors beside it are redundant: the image and video "
        "processor registrations are being skipped, because by-name lookups now "
        "resolve to transformers' own classes (and registering ours would shadow "
        "the image one), and the AutoProcessor registration is unreachable once "
        "transformers' Glm5NextConfig wins the AutoConfig mapping. "
        "Delete all four files and the import in sglang/srt/configs/__init__.py. "
        "They were vendored from huggingface/transformers@%s; diff before deleting "
        "if any local fix in them has not gone upstream yet.",
        VENDORED_FROM_TRANSFORMERS_COMMIT,
    )


_warn_if_redundant()

# fast_image_processor_class, not slow: Glm5NextImageProcessor is a TorchvisionBackend.
#
# All three registrations are needed while the shim is live, and none is
# redundant with the others. AutoProcessor.register is what makes
# `AutoProcessor.from_pretrained` return Glm5NextProcessor at all; the other two
# are what let *its* `from_pretrained` resolve the
# "image_processor_type": "Glm5NextImageProcessor" and
# "video_processor_type": "Glm5NextVideoProcessor" strings in the checkpoint's
# processor_config.json, which transformers looks up by class name across the
# Auto mappings' extra content.
#
# The two name-keyed ones stop once transformers ships glm5_next -- see
# _warn_if_redundant above for why that is not the same as them being inert.
if not _is_redundant():
    AutoImageProcessor.register(
        Glm5NextConfig, fast_image_processor_class=Glm5NextImageProcessor, exist_ok=True
    )
    AutoVideoProcessor.register(Glm5NextConfig, Glm5NextVideoProcessor, exist_ok=True)

# Unconditional: whether this one is reachable depends on which Glm5NextConfig
# won the AutoConfig mapping, which cannot be observed from here, and losing it
# when it was still needed means images are accepted and silently dropped.
AutoProcessor.register(Glm5NextConfig, Glm5NextProcessor, exist_ok=True)

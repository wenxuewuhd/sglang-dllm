"""Unit tests for the vendored GLM-5.3-Flash processors and their Auto* registrations.

CPU only, and deliberately checkpoint-free: everything here runs against freshly
constructed processor objects, so it needs no weights, no GPU and no NPU. The
three vendored files under sglang/srt/configs (`image_processing_glm5_next.py`,
`processing_glm5_next.py`, `video_processing_glm5_next.py`) are copies of
transformers' `main`, carrying local fixes that upstream does not have; these
tests pin those fixes, because each one fails in a way nothing else would
notice -- a mutated class default that only misbehaves on a *later* request, a
branch that has never run, and a token count that disagrees with the pixels it
describes.
"""

import importlib.util
import unittest

import torch
from transformers.models.auto.image_processing_auto import (
    IMAGE_PROCESSOR_MAPPING,
    get_image_processor_class_from_name,
)
from transformers.models.auto.processing_auto import PROCESSOR_MAPPING
from transformers.models.auto.video_processing_auto import (
    VIDEO_PROCESSOR_MAPPING,
    video_processor_class_from_name,
)

import sglang.srt.configs  # noqa: F401  -- importing it is what registers the classes
from sglang.srt.configs.glm5_next import Glm5NextConfig
from sglang.srt.configs.image_processing_glm5_next import Glm5NextImageProcessor
from sglang.srt.configs.processing_glm5_next import (
    Glm5NextProcessor,
    Glm5NextProcessorKwargs,
)
from sglang.srt.configs.video_processing_glm5_next import Glm5NextVideoProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _processor_without_checkpoint() -> Glm5NextProcessor:
    """A processor carrying only the two sub-processors.

    `_get_num_multimodal_tokens` reads nothing else, while the real `__init__`
    needs a tokenizer, which needs a checkpoint -- which these tests must not.
    """
    processor = Glm5NextProcessor.__new__(Glm5NextProcessor)
    processor.image_processor = Glm5NextImageProcessor()
    processor.video_processor = Glm5NextVideoProcessor()
    return processor


class TestGlm5NextProcessorTokenCounts(CustomTestCase):
    def test_call_does_not_mutate_class_level_defaults(self):
        """`_defaults` is shared by every processor in the process.

        Upstream calls `_defaults.get("videos_kwargs", {}).update(kwargs)`, and
        because "videos_kwargs" IS a key there, that updates the live dict: one
        caller's kwargs silently become the default for every processor built
        afterwards. The damage shows up in an unrelated later request, so
        nothing about the offending call looks wrong.
        """
        processor = _processor_without_checkpoint()
        before = {k: dict(v) for k, v in Glm5NextProcessorKwargs._defaults.items()}

        processor._get_num_multimodal_tokens(image_sizes=[[448, 448]], merge_size=4)
        processor._get_num_multimodal_tokens(image_sizes=[[448, 448]], merge_size=4)

        self.assertEqual(
            Glm5NextProcessorKwargs._defaults["videos_kwargs"],
            {"return_metadata": True},
        )
        self.assertEqual(Glm5NextProcessorKwargs._defaults, before)

    def test_video_only_call_reports_the_missing_upstream_method(self):
        """A video-only call must name what is missing.

        Upstream binds `merge_size` only inside the `image_sizes` branch and
        then reads it here (UnboundLocalError), and then calls
        `get_number_of_video_patches`, which exists in no version of this
        processor (AttributeError three frames down). Both are replaced by one
        NotImplementedError that says so.
        """
        processor = _processor_without_checkpoint()

        with self.assertRaises(NotImplementedError) as caught:
            processor._get_num_multimodal_tokens(
                video_sizes=[[8, 448, 448]], merge_size=4
            )

        self.assertIn("get_number_of_video_patches", str(caught.exception))
        # The video branch copies `_defaults["videos_kwargs"]` before updating it,
        # and that copy happens before the raise above -- so this is the call that
        # would corrupt the class default if the copy were dropped again.
        self.assertEqual(
            Glm5NextProcessorKwargs._defaults["videos_kwargs"],
            {"return_metadata": True},
        )

    def test_patch_count_agrees_with_the_grid_preprocess_produces(self):
        """`get_number_of_image_patches` must describe the resize that ran.

        sglang counts every image's tokens through this method
        (multimodal/processors/base_processor.resolve_image_token_counts) while
        the pixels come from `_preprocess`, so the two disagreeing is a
        token-count bug. Upstream omits `patch_expand_factor` from the count and
        the two agree only at factor 1 -- which is what this checkpoint ships,
        so the disagreement is latent rather than absent. The invariant pinned
        here is the agreement itself, at every factor, not any particular count.
        """
        image_processor = Glm5NextImageProcessor()

        for patch_expand_factor in (1, 2, 4):
            for height, width in ((448, 448), (570, 380), (450, 450), (113, 97)):
                with self.subTest(factor=patch_expand_factor, size=(height, width)):
                    images_kwargs = {"patch_expand_factor": patch_expand_factor}
                    produced = image_processor.preprocess(
                        images=[torch.zeros(3, height, width, dtype=torch.uint8)],
                        return_tensors="pt",
                        **images_kwargs,
                    )
                    _, grid_h, grid_w = produced["image_grid_thw"][0].tolist()

                    self.assertEqual(
                        image_processor.get_number_of_image_patches(
                            height, width, images_kwargs
                        ),
                        grid_h * grid_w,
                    )

    def test_patch_count_is_invariant_to_temporal_patch_size(self):
        """Why `temporal_patch_size` is *not* a second `patch_expand_factor`.

        `get_number_of_image_patches` reads `temporal_patch_size` off `self`
        while `_preprocess` takes it as a per-call parameter -- the same shape
        as the `patch_expand_factor` bug above, so it invites the same fix. It
        is not the same bug: on the image path `smart_resize` is called with
        `num_frames == temporal_factor`, and the factor then cancels out of
        every budget comparison it appears in, so neither the produced grid nor
        the returned count depends on it. Reading it off `self` therefore cannot
        make the two disagree, and "fixing" it would add a local delta to a
        vendored file for no behavioural difference. This test is what would
        notice if that ever stopped being true.
        """
        image_processor = Glm5NextImageProcessor()

        for temporal_patch_size in (4, 8):
            for height, width in ((448, 448), (570, 380), (113, 97)):
                with self.subTest(
                    temporal_patch_size=temporal_patch_size, size=(height, width)
                ):
                    produced = image_processor.preprocess(
                        images=[torch.zeros(3, height, width, dtype=torch.uint8)],
                        return_tensors="pt",
                        temporal_patch_size=temporal_patch_size,
                    )
                    _, grid_h, grid_w = produced["image_grid_thw"][0].tolist()

                    self.assertEqual(
                        image_processor.get_number_of_image_patches(
                            height, width, {"temporal_patch_size": temporal_patch_size}
                        ),
                        grid_h * grid_w,
                    )


class TestGlm5NextProcessorRegistry(CustomTestCase):
    def test_auto_registries_resolve_the_checkpoints_class_names(self):
        """The registrations importing `sglang.srt.configs` performs must hold.

        Without them `AutoProcessor.from_pretrained` falls back to a bare
        tokenizer *without raising*: images are accepted and dropped. The
        checkpoint's processor_config.json reaches the two sub-processors by
        class name, not by config, so both name lookups are asserted too.
        """
        if importlib.util.find_spec("transformers.models.glm5_next") is not None:
            # The day transformers ships glm5_next, this shim is redundant.
            # glm5_next_processing_registry.py then skips the two name-keyed
            # registrations on purpose -- registering the image processor would
            # shadow transformers' own class of the same name, because
            # get_image_processor_class_from_name scans _extra_content first --
            # so the lookups below must land on transformers' classes instead.
            self.assertNotIn(Glm5NextConfig, IMAGE_PROCESSOR_MAPPING._extra_content)
            self.assertNotIn(Glm5NextConfig, VIDEO_PROCESSOR_MAPPING._extra_content)
            self.assertIsNotNone(
                get_image_processor_class_from_name("Glm5NextImageProcessor")
            )
            self.assertIsNotNone(
                video_processor_class_from_name("Glm5NextVideoProcessor")
            )
            self.fail(
                "transformers now ships glm5_next natively: delete "
                "sglang/srt/configs/glm5_next_processing_registry.py, the three "
                "vendored *_glm5_next.py processors beside it, the import in "
                "sglang/srt/configs/__init__.py, and this test -- after diffing "
                "the vendored copies for local fixes that never went upstream."
            )

        self.assertIs(PROCESSOR_MAPPING[Glm5NextConfig], Glm5NextProcessor)
        self.assertIs(
            IMAGE_PROCESSOR_MAPPING[Glm5NextConfig]["torchvision"],
            Glm5NextImageProcessor,
        )
        self.assertIs(VIDEO_PROCESSOR_MAPPING[Glm5NextConfig], Glm5NextVideoProcessor)

        # What the checkpoint actually names, and how transformers resolves it.
        self.assertIs(
            get_image_processor_class_from_name("Glm5NextImageProcessor"),
            Glm5NextImageProcessor,
        )
        self.assertIs(
            video_processor_class_from_name("Glm5NextVideoProcessor"),
            Glm5NextVideoProcessor,
        )


if __name__ == "__main__":
    unittest.main()

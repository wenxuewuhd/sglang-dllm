# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ---------------------------------------------------------------------------
# Vendored from huggingface/transformers, src/transformers/models/glm5_next/processing_glm5_next.py,
# at commit 3197876523ebee6adc7877b1853fc4932bef0f9f
# ("fix failed test cases for glm5_next" (#48497), 2026-09-04, branch `main`).
#
# Why a copy: no released transformers carries `glm5_next` (checked through
# v5.16.0), yet the GLM-5.3-Flash checkpoint's processor_config.json names these
# classes.  AutoProcessor does not raise when it cannot resolve them -- it falls
# back to a bare tokenizer that accepts `images=` and discards it -- so without
# this copy every image is silently dropped.  glm5_next_processing_registry.py
# registers these classes and warns when the copy becomes redundant.
#
# Upstream generates this file from modular_glm5_next.py, which is NOT vendored
# here, so edits below do not flow back and no CI checks them against a modular
# source.  Anything fixed here should also be sent upstream.
#
# Changes relative to upstream at that commit:
#   * relative imports (`from ...x import y`) rewritten as absolute
#     (`from transformers.x import y`)
#   * reformatted with the formatter this repo pins in .pre-commit-config.yaml,
#     plus isort 7.0.0.  At this base that formatter is ruff-format
#     (astral-sh/ruff-pre-commit v0.15.1) -- not black; branches whose base
#     predates the ruff-format switch pin black 26.1.0 instead.  The text below
#     is byte-identical under both, so re-running either reproduces it exactly --
#     but check the pin before concluding this copy has drifted.
#   * `_get_num_multimodal_tokens`: three local fixes, see the comments there --
#     `merge_size` is bound per branch (a video-only call raised
#     UnboundLocalError), `_defaults["videos_kwargs"]` is copied before being
#     updated (upstream mutated class-level state for the whole process), and the
#     call to the nonexistent `get_number_of_video_patches` now reports itself
#     instead of dying with a bare AttributeError
# ---------------------------------------------------------------------------


import numpy as np
from transformers.processing_utils import (
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
)
from transformers.utils import auto_docstring, logging

logger = logging.get_logger(__name__)


class Glm5NextProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": True,
        },
        "videos_kwargs": {"return_metadata": True},
    }


@auto_docstring
class Glm5NextProcessor(ProcessorMixin):
    valid_processor_kwargs = Glm5NextProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = (
            "<|image|>"
            if not hasattr(tokenizer, "image_token")
            else tokenizer.image_token
        )
        self.video_token = (
            "<|video|>"
            if not hasattr(tokenizer, "video_token")
            else tokenizer.video_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        super().__init__(
            image_processor, tokenizer, video_processor, chat_template=chat_template
        )
        self.video_start_id = tokenizer.convert_tokens_to_ids("<|begin_of_video|>")
        self.video_end_id = tokenizer.convert_tokens_to_ids("<|end_of_video|>")

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        merge_length = self.image_processor.merge_size**2
        num_image_tokens = (
            image_inputs["image_grid_thw"][image_idx].prod() // merge_length
        )
        return self.image_token * num_image_tokens

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        merge_length = self.video_processor.merge_size**2
        num_frames = video_inputs["video_grid_thw"][video_idx][0]
        num_image_tokens = (
            video_inputs["video_grid_thw"][video_idx].prod()
            // merge_length
            // num_frames
        )
        metadata = video_inputs["video_metadata"][video_idx]
        video_structure = ""

        if metadata.fps is None:
            logger.warning_once(
                "GLM5_NEXT requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
            )
        metadata.fps = 24 if metadata.fps is None else metadata.fps
        timestamps = metadata.timestamps[::2]  # mrope

        unique_timestamps = []
        for idx in range(0, len(timestamps)):
            unique_timestamps.append(timestamps[idx])

        selected_timestamps = unique_timestamps[:num_frames]
        while len(selected_timestamps) < num_frames:
            selected_timestamps.append(
                selected_timestamps[-1] if selected_timestamps else 0
            )

        for frame_idx in range(num_frames):
            timestamp_sec = selected_timestamps[frame_idx]
            frame_structure = self.replace_frame_token_id(
                timestamp_sec, num_image_tokens=num_image_tokens
            )
            video_structure += frame_structure

        return video_structure

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.
        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.
            video_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (num_frames, height, width) per each video.
        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """

        vision_data = {}
        if image_sizes is not None:
            # LOCAL FIX (not upstream): copy before update. `_defaults` is class
            # state shared by every instance in the process, so `.get(...).update()`
            # on a key that exists would make one caller's kwargs the default for
            # everybody afterwards. "images_kwargs" happens not to be in `_defaults`
            # today, so this is a no-op guard here -- but it is the same line that
            # is an actual bug in the video branch below, and adding the key
            # upstream would silently make it one here too.
            images_kwargs = dict(
                Glm5NextProcessorKwargs._defaults.get("images_kwargs", {})
            )
            images_kwargs.update(kwargs)
            merge_size = (
                images_kwargs.get("merge_size", None) or self.image_processor.merge_size
            )

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(
                    *image_size, images_kwargs
                )
                for image_size in image_sizes
            ]
            num_image_tokens = [
                (num_patches // merge_size**2) for num_patches in num_image_patches
            ]
            vision_data.update(
                {
                    "num_image_tokens": num_image_tokens,
                    "num_image_patches": num_image_patches,
                }
            )

        if video_sizes is not None:
            # LOCAL FIX (not upstream): "videos_kwargs" IS in `_defaults`, so
            # upstream's `.get(...).update(kwargs)` writes the caller's kwargs into
            # class-level state that every later instance in the process inherits.
            videos_kwargs = dict(
                Glm5NextProcessorKwargs._defaults.get("videos_kwargs", {})
            )
            videos_kwargs.update(kwargs)
            # LOCAL FIX (not upstream): upstream reads `merge_size` here, which is
            # bound only inside the `image_sizes` branch above -- a video-only call
            # raises UnboundLocalError. Bind it from the video processor, which is
            # the one whose patches are being counted.
            merge_size = (
                videos_kwargs.get("merge_size", None) or self.video_processor.merge_size
            )
            # LOCAL FIX (not upstream): no Glm5Next video processor defines
            # `get_number_of_video_patches` -- not here and not in transformers --
            # so this branch has never run anywhere. Supplying the method would be
            # inventing upstream API from this copy, and sglang does not need it
            # (its GLM video path counts tokens in
            # sglang/srt/multimodal/processors/glm4v.py, not through here), so say
            # what is missing instead of failing with a bare AttributeError.
            if not hasattr(self.video_processor, "get_number_of_video_patches"):
                raise NotImplementedError(
                    f"{type(self.video_processor).__name__} does not implement "
                    "get_number_of_video_patches, so video token counts cannot be derived "
                    "from video_sizes alone. This is an upstream gap in transformers' "
                    "glm5_next processor, kept visible here rather than papered over."
                )
            num_video_patches = [
                self.video_processor.get_number_of_video_patches(
                    *video_size, videos_kwargs
                )
                for video_size in video_sizes
            ]
            num_video_tokens = [
                (num_patches // merge_size**2) for num_patches in num_video_patches
            ]
            vision_data["num_video_tokens"] = num_video_tokens

        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(
        self,
        generated_outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        return super().model_input_names + ["mm_token_type_ids"]

    def create_mm_token_type_ids(self, input_ids: list) -> list[list[int]]:
        # We have to iterate for each list separately because inputs
        # might be non-padded lists and we can't cast numpy on that!
        # Then cast numpy as each input for faster indexing
        mm_token_type_ids = []
        for input in input_ids:
            array_ids = np.array(input)
            mm_token_types = np.zeros_like(input)

            # Replace 0 -> 2 only inside video segments because Glm5Next
            # uses the same special token to denote images and video
            # Otherwise replace 0 -> 1 for image modality
            starts = np.cumsum(array_ids == self.video_start_id, axis=0)
            ends = np.cumsum(array_ids == self.video_end_id, axis=0)
            is_video_modality = starts > ends

            mm_token_types[(array_ids == self.image_token_id) & is_video_modality] = 2
            mm_token_types[
                (array_ids == self.image_token_id) & (~is_video_modality)
            ] = 1
            mm_token_type_ids.append(mm_token_types.tolist())
        return mm_token_type_ids

    def replace_frame_token_id(self, timestamp_sec, num_image_tokens: int = 1):
        return f"<|begin_of_image|>{self.image_token * num_image_tokens}<|end_of_image|>{timestamp_sec:.1f} seconds"


__all__ = ["Glm5NextProcessor"]

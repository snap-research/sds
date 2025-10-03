"""
Transform presets for common data processing tasks in the SDS framework.
They are structured as classes because they need to be pickleable for distributed processing
(otherwise, torch.data.utils.DataLoader will fail to pickle them for num_workers > 0).
"""
import json
import random
from typing import Any, Callable, Sequence

import numpy as np
import torch
from beartype import beartype
from PIL import Image

from sds.structs import SampleData, SampleTransform
import sds.transforms.functional as SDF


#----------------------------------------------------------------------------
# Base transforms.
class BaseTransform:
    """Base class for all transforms. Provides a common interface."""
    def __init__(self, input_field: str, output_field: str | None = None, **transform_kwargs):
        self.input_field = input_field
        self.output_field = output_field if output_field is not None else input_field
        self.transform_kwargs = transform_kwargs

    def __call__(self, sample: SampleData) -> SampleData:
        raise NotImplementedError("Subclasses must implement the __call__ method.")

#----------------------------------------------------------------------------
# Image processing transforms.

@beartype
class DecodeImageTransform(BaseTransform):
    """Decodes an image file into a [c, h, w] torch tensor."""
    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=[self.input_field], absent=[])
        sample[self.output_field] = SDF.load_image_from_bytes(sample[self.input_field])
        return sample

@beartype
class ResizeImageTransform(BaseTransform):
    """Resizes images to the specified resolution."""
    def __init__(self, input_field: str, output_field: str | None = None, **resize_kwargs):
        self.input_field = input_field
        self.output_field = output_field if output_field is not None else input_field
        self.resize_kwargs = resize_kwargs

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=[self.input_field], absent=[])
        sample[self.output_field] = SDF.resize_image(sample[self.input_field], **self.resize_kwargs)
        return sample

@beartype
class ReshapeImageAsVideoTransform(BaseTransform):
    """Reshapes an image into a single-frame video."""
    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=[self.input_field], absent=[self.output_field])
        sample[self.output_field] = SDF.reshape_image_as_single_frame_video(sample[self.input_field])
        return sample

@beartype
class ConvertImageToByteTensorTransform(BaseTransform):
    """Converts an image to a torch tensor."""
    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present={self.input_field: Image.Image}, absent=[])
        sample[self.output_field] = SDF.convert_pil_image_to_byte_tensor(sample[self.input_field])
        return sample

#----------------------------------------------------------------------------
# Video processing transforms.

@beartype
class DecodeVideoTransform(BaseTransform):
    """A video transform which decodes a video file into a list of frames."""
    def __init__(self, input_field: str, num_frames: int, output_field: str | None = None, duration_field: str | None = None,
                framerate_field: str | None = None, frame_timestamps_output_field: str | None = None, **decode_kwargs):
        self.input_field = input_field
        self.output_field = output_field if output_field is not None else input_field
        self.num_frames = num_frames
        self.duration_field = duration_field
        self.framerate_field = framerate_field
        self.frame_timestamps_output_field = frame_timestamps_output_field
        self.decode_kwargs = decode_kwargs

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=[self.input_field], absent=[])
        real_duration = float(sample[self.duration_field]) if self.duration_field is not None else None
        real_framerate = float(sample[self.framerate_field]) if self.framerate_field is not None else None
        sample[self.output_field], frame_timestamps, _clip_duration, _waveform, _audio_sr = SDF.decode_video(
            video_file=sample[self.input_field], num_frames_to_extract=self.num_frames,
            real_duration=real_duration, real_framerate=real_framerate, **self.decode_kwargs)
        if self.frame_timestamps_output_field is not None:
            _validate_fields(sample, present=[], absent=[self.frame_timestamps_output_field])
            sample[self.frame_timestamps_output_field] = frame_timestamps
        return sample

@beartype
class ResizeVideoTransform(BaseTransform):
    """Resizes each frame in the video to the specified resolution."""
    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=[self.input_field], absent=[])
        sample[self.output_field] = SDF.lean_resize_frames(frames=sample[self.input_field], **self.transform_kwargs)
        return sample

@beartype
class ConvertVideoToByteTensorTransform(BaseTransform):
    """Converts an image to a torch tensor."""
    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present={self.input_field: list}, absent=[])
        sample[self.output_field] = torch.stack([SDF.convert_pil_image_to_byte_tensor(x) for x in sample[self.input_field]]) # [t, c, h, w]
        return sample

@beartype
class InitVideoDecoderTransform(BaseTransform):
    """Initializes the video decoder with a video file."""
    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=[self.input_field], absent=[])
        sample[self.output_field] = SDF.init_video_decoder(sample[self.input_field])
        return sample

@beartype
class DeleteVideoDecoderTransform(BaseTransform):
    """Deletes the video decoder from the sample. We need a special transform for this to close the decoder manually."""
    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=[self.input_field], absent=[])
        sample[self.input_field].close()
        del sample[self.input_field]
        return sample

@beartype
class NormalizeFramesTransform(BaseTransform):
    """Normalizes video frames to [-1, 1] range."""
    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present={self.input_field: torch.Tensor}, absent=[])
        assert sample[self.input_field].dtype == torch.uint8, \
            f"Expected input field '{self.input_field}' to be of type torch.uint8, but got {sample[self.input_field].dtype}."
        sample[self.output_field] = sample[self.input_field].float() / 127.5 - 1.0  # Normalize to [-1, 1]
        return sample

@beartype
class UndistortFramesTransform(BaseTransform):
    """
    We have some broken distorted videos, so we should undistort them.
    For this, we check for original_height/original_width vs height/width aspect ratio.
    TODO: this transform should not exist... We should better fix the dataset...
    """
    def __init__(self, input_field: str, original_resolution_fields: tuple[str, str], output_field: str | None = None):
        self.input_field = input_field
        self.original_resolution_fields = original_resolution_fields
        self.output_field = output_field if output_field is not None else input_field

    def __call__(self, sample: SampleData) -> SampleData:
        orig_height, orig_width = sample.get(self.original_resolution_fields[0]), sample.get(self.original_resolution_fields[1])
        if orig_height is None or orig_width is None:
            return sample  # If original resolution is not provided, skip undistortion.
        assert isinstance(orig_height, (int, float)) and isinstance(orig_width, (int, float)), \
            f"Original resolution fields must be numeric, got {type(orig_height)} and {type(orig_width)}."
        orig_aspect_ratio = orig_width / orig_height
        cur_width, cur_height = sample[self.input_field][0].size
        cur_aspect_ratio = cur_width / cur_height
        if abs(orig_aspect_ratio - cur_aspect_ratio) > 0.02:  # Allow some tolerance
            new_height = round(cur_width / orig_aspect_ratio)
            sample[self.output_field] = SDF.lean_resize_frames(frames=sample[self.input_field], resolution=(new_height, cur_width))
        return sample

#----------------------------------------------------------------------------
# Audio processing transforms.

@beartype
class ConvertAudioToFloatTensorTransform(BaseTransform):
    """Converts an audio file to a torch tensor."""
    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present={self.input_field: np.ndarray}, absent=[])
        sample[self.output_field] = torch.from_numpy(sample[self.input_field]).float() # [c, t]
        return sample

@beartype
class AverageAudioTransform(BaseTransform):
    """Averages audio data across channels."""
    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present={self.input_field: torch.Tensor}, absent=[])
        if sample[self.input_field].ndim == 1:
            # Mono audio, no need to average
            sample[self.output_field] = sample[self.input_field]
        else:
            # Multi-channel audio, average across channels
            sample[self.output_field] = sample[self.input_field].mean(dim=0, keepdim=True)  # [1, t] for mono audio
        return sample

@beartype
class ResizeAudioTransform(BaseTransform):
    """Resizes the audio by resampling it and then trimming or padding it to a specified duration."""
    def __init__(self, audio_input_field: str, original_sr_input_field: str, clip_duration_input_field: str, target_audio_sr: int, output_field: str | None = None, **resampling_kwargs):
        self.audio_input_field = audio_input_field
        self.original_sr_input_field = original_sr_input_field
        self.clip_duration_input_field = clip_duration_input_field
        self.output_field = output_field if output_field is not None else audio_input_field
        self.target_audio_sr = target_audio_sr
        self.resampling_kwargs = resampling_kwargs

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present={self.audio_input_field: torch.Tensor, self.clip_duration_input_field: float, self.original_sr_input_field: int}, absent=[])
        waveform = SDF.resample_waveform(waveform=sample[self.audio_input_field], orig_freq=sample[self.original_sr_input_field], new_freq=self.target_audio_sr, **self.resampling_kwargs)
        waveform = SDF.resize_waveform(waveform, target_length=int(self.target_audio_sr * sample[self.clip_duration_input_field]))
        sample[self.output_field] = waveform
        return sample

class NormalizeAudioTransform(BaseTransform):
    """Normalizes the audio waveform to have maximum absolute value of 0.95."""
    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present={self.input_field: torch.Tensor}, absent=[])
        waveform = sample[self.input_field]
        max_val = waveform.abs().max()
        sample[self.output_field] = (waveform / max_val * 0.95) if max_val > 0 else waveform
        return sample

#----------------------------------------------------------------------------
# VAE latents processing transforms.

class LoadLatentFromDiskTransform(BaseTransform):
    """Loads VAE latents from a file."""
    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=[self.input_field], absent=[])
        sample[self.output_field] = SDF.load_torch_state_from_pickle(sample[self.input_field], non_torch_fields=['input_shape', 'input_framerate'])
        return sample

class SampleImageVAELatentTransform(BaseTransform):
    """Samples image VAE latents from a sample."""
    def __init__(self, input_field: str, output_field: str | None = None, mean_field: str='mean', logvar_field: str='logvar'):
        self.input_field = input_field
        self.output_field = output_field if output_field is not None else input_field
        self.mean_field = mean_field
        self.logvar_field = logvar_field

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present={self.input_field: dict}, absent=[])
        latents_dict = dict(mean=sample[self.input_field][self.mean_field], logvar=sample[self.input_field][self.logvar_field])
        sample[self.output_field] = SDF.sample_image_vae_latents(latents_dict)
        return sample

class SampleVideoVAELatentTransform:
    def __init__(self, input_field: str, output_field: str | None = None, mean_field: str='mean', logvar_field: str='logvar', framerate: float | None=None, random_offset: bool=False):
        self.input_field = input_field
        self.output_field = output_field if output_field is not None else input_field
        self.mean_field = mean_field
        self.logvar_field = logvar_field
        self.framerate = framerate
        self.random_offset = random_offset

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present={self.input_field: dict}, absent=[])
        raw_latents_dict = sample[self.input_field]
        latents_dict = dict(mean=raw_latents_dict[self.mean_field], logvar=raw_latents_dict[self.logvar_field])
        sample[self.output_field] = SDF.sample_video_vae_latents(
            latents_dict=latents_dict,
            orig_shape=tuple(raw_latents_dict['input_shape']),
            fps_orig=raw_latents_dict.get('input_framerate'),
            fps_trg=self.framerate,
            random_offset=self.random_offset
        )
        return sample

#----------------------------------------------------------------------------
# Multi-modal transforms.

@beartype
class DecodeVideoAndAudioTransform(BaseTransform):
    """Decodes a video file and extracts both video and audio, returning both as tensors."""
    def __init__(
            self, input_field: str, video_output_field: str, original_clip_duration_output_field: str,
            audio_output_field: str, original_sr_output_field: str, num_frames: int, duration_field: str | None = None,
            framerate_field: str | None = None, frame_timestamps_output_field: str | None = None, **decode_kwargs
        ):
        self.input_field = input_field
        self.video_output_field = video_output_field
        self.original_clip_duration_output_field = original_clip_duration_output_field
        self.audio_output_field = audio_output_field
        self.original_sr_output_field = original_sr_output_field
        self.duration_field = duration_field
        self.framerate_field = framerate_field
        self.num_frames = num_frames
        self.decode_kwargs = decode_kwargs
        self.frame_timestamps_output_field = frame_timestamps_output_field

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=[self.input_field], absent=[self.audio_output_field, self.original_sr_output_field])
        real_duration = float(sample[self.duration_field]) if self.duration_field is not None else None
        real_framerate = float(sample[self.framerate_field]) if self.framerate_field is not None else None
        video_data, frame_timestamps, clip_duration, waveform_data, waveform_sampling_rate = SDF.decode_video(
            sample[self.input_field], num_frames_to_extract=self.num_frames, real_duration=real_duration,
            real_framerate=real_framerate, **self.decode_kwargs, return_audio=True)
        sample[self.video_output_field] = video_data
        sample[self.original_clip_duration_output_field] = clip_duration
        sample[self.audio_output_field] = waveform_data
        sample[self.original_sr_output_field] = waveform_sampling_rate
        if self.frame_timestamps_output_field is not None:
            _validate_fields(sample, present=[], absent=[self.frame_timestamps_output_field])
            sample[self.frame_timestamps_output_field] = frame_timestamps
        return sample

#----------------------------------------------------------------------------
# Label/text/metadata processing transforms.

def is_dummy_field(d: dict, field: str, return_reason: bool=False) -> bool | tuple[bool, str]:
    """Checks if a field is dummy: absent, None, or an empty string."""
    is_dummy = False
    reason = ""
    if field not in d:
        is_dummy = True
        reason = f"Field '{field}' is absent."
    elif d[field] is None:
        is_dummy = True
        reason = f"Field '{field}' is None."
    elif isinstance(d[field], str) and d[field] == '':
        is_dummy = True
        reason = f"Field '{field}' is an empty string."
    elif isinstance(d[field], (list, dict)) and len(d[field]) == 0:
        is_dummy = True
        reason = f"Field '{field}' is an empty {type(d[field]).__name__}."
    elif isinstance(d[field], float) and np.isnan(d[field]):
        is_dummy = True
        reason = f"Field '{field}' is float and NaN."
    elif isinstance(d[field], torch.Tensor) and torch.isnan(d[field]).any():
        is_dummy = True
        reason = f"Field '{field}' is a torch.Tensor and contains NaN values."
    return (is_dummy, reason) if return_reason else is_dummy

@beartype
class TextEmbLoaderTransform:
    """Loads text embeddings from a pickle file."""
    def __init__(self, input_field: str, num_tokens: int, output_field: str | None = None, allow_missing: bool = False):
        self.input_field = input_field
        self.output_field = output_field if output_field is not None else input_field
        self.num_tokens = num_tokens
        self.allow_missing = allow_missing

    def __call__(self, sample: SampleData) -> SampleData:
        if self.allow_missing and is_dummy_field(sample, self.input_field):
            sample[self.output_field] = None
        else:
            _validate_fields(sample, present=[self.input_field], absent=[])
            sample[self.output_field] = SDF.load_pickle_embeddings(sample[self.input_field], self.num_tokens)
        return sample

@beartype
class TextEmbSamplingTransform:
    """Subsamples text embeddings from a sample."""
    def __init__(self, input_text_fields: Sequence[str], input_text_emb_fields: Sequence[str], probabilities: Sequence[float], output_text_field: str, output_text_emb_field: str, cleanup: bool=True, allow_missing: bool=False):
        assert len(input_text_fields) == len(input_text_emb_fields) == len(probabilities), \
            f"Input fields, text embedding fields, and probabilities must have the same length: {len(input_text_fields)}, {len(input_text_emb_fields)}, {len(probabilities)}."
        self.input_text_fields = input_text_fields
        self.input_text_emb_fields = input_text_emb_fields
        self.output_text_field = output_text_field
        self.output_text_emb_field = output_text_emb_field
        self.probabilities = probabilities
        self.allow_missing = allow_missing
        self.cleanup = cleanup

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=self.input_text_emb_fields, absent=[])
        assert self.allow_missing or all(not is_dummy_field(sample, f) for f in self.input_text_emb_fields), \
            f"Some input fields are missing: {self.input_text_emb_fields} not found in sample with keys {list(sample.keys())}."
        input_text_emb_fields = [f for f in self.input_text_emb_fields if not is_dummy_field(sample, f)]
        weights = [p for p, f in zip(self.probabilities, self.input_text_emb_fields) if f in input_text_emb_fields]
        input_text_fields = [tf for tf, tef in zip(self.input_text_fields, self.input_text_emb_fields) if tef in input_text_emb_fields]
        assert len(input_text_emb_fields) > 0, f"No valid input fields found in sample with keys {list(sample.keys())}."
        assert len(input_text_emb_fields) == len(input_text_fields) == len(weights), \
            f"Mismatch in lengths: input_text_emb_fields={len(input_text_emb_fields)}, input_text_fields={len(input_text_fields)}, weights={len(weights)}."
        selected_index = random.choices(range(len(input_text_emb_fields)), k=1, weights=weights)[0]
        selected_text_emb = sample[input_text_emb_fields[selected_index]]
        selected_text = sample[input_text_fields[selected_index]]
        if self.cleanup:
            for field in self.input_text_emb_fields + self.input_text_fields:
                if field in sample:
                    del sample[field]
        sample[self.output_text_field] = selected_text
        sample[self.output_text_emb_field] = selected_text_emb

        return sample

@beartype
class LoadJsonMetadataTransform(BaseTransform):
    """Loads metadata from a JSON file."""
    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present={self.input_field: bytes}, absent=[])
        sample[self.output_field] = json.loads(sample[self.input_field])
        return sample

@beartype
class MergeDictFieldTransform:
    """Takes a dictionary field and merges its contents right into the sample root."""
    def __init__(self, input_field: str, skip_if_exists: bool=False, overwrite: bool=False, cleanup: bool=False):
        self.input_field = input_field
        self.skip_if_exists = skip_if_exists
        self.overwrite = overwrite
        self.cleanup = cleanup

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present={self.input_field: dict}, absent=[])

        for k, v in sample[self.input_field].items():
            if k in sample:
                if self.skip_if_exists:
                    continue
                else:
                    assert self.overwrite, f"Field '{k}' already exists in sample with keys {list(sample.keys())}, while skip_if_exists is False and overwrite is False."
            sample[k] = v
        if self.cleanup:
            del sample[self.input_field]
        return sample

@beartype
class ExtractMetadataSubfieldTransform:
    """Extracts a specific field from the metadata."""
    def __init__(self, metadata_subfield: str, output_field: str, metadata_field: str = 'meta'):
        self.metadata_subfield = metadata_subfield
        self.output_field = output_field
        self.metadata_field = metadata_field

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present={self.metadata_field: dict}, absent=[self.output_field])
        _validate_fields(sample[self.metadata_field], present=[self.metadata_subfield], absent=[])
        sample[self.output_field] = sample[self.metadata_field][self.metadata_subfield]
        return sample

@beartype
class OneHotEncodeTransform:
    """One-hot encodes a categorical field."""
    def __init__(self, input_field: str, num_classes: int, output_field: str | None = None):
        self.input_field = input_field
        self.output_field = output_field if output_field is not None else input_field
        self.num_classes = num_classes

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present={self.input_field: int}, absent=[])
        sample[self.output_field] = SDF.one_hot_encode(sample[self.input_field], num_classes=self.num_classes)
        return sample

@beartype
class AddConstantFieldTransform:
    """Adds a constant field to the sample."""
    def __init__(self, output_field: str, value: Any):
        self.output_field = output_field
        self.value = value

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=[], absent=[self.output_field])
        sample[self.output_field] = self.value
        return sample

@beartype
class ClassNameToIDTransform:
    """Converts a string class name to an integer ID."""
    def __init__(self, input_field: str, class_name_to_id_mapping: dict[str, int] | Callable, output_field: str | None = None):
        self.input_field = input_field
        self.output_field = output_field if output_field is not None else input_field
        self.class_name_to_id_mapping = class_name_to_id_mapping if class_name_to_id_mapping is not None else {}

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=[self.input_field], absent=[])
        if callable(self.class_name_to_id_mapping):
            class_id = self.class_name_to_id_mapping(sample[self.input_field])
        else:
            class_id = self.class_name_to_id_mapping.get(sample[self.input_field], -1)
        assert class_id != -1, f"Class name '{sample[self.input_field]}' not found in mapping."
        sample[self.output_field] = class_id
        return sample

@beartype
class GatherFieldsTransform:
    """Gathers specified fields from the sample into a new dictionary."""
    _supported_output_types = (dict, list, torch.Tensor)
    def __init__(self, input_fields: Sequence[str], output_field: str, output_type: type = dict):
        assert output_type in self._supported_output_types, f"output_type must be in {self._supported_output_types}, got {output_type}."
        assert len(input_fields) > 0, "At least one field must be specified to gather."
        self.input_fields = input_fields
        self.output_field = output_field
        self.output_type = output_type

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=self.input_fields, absent=[])
        if self.output_type is dict:
            gathered_data = {field: sample[field] for field in self.input_fields}
        elif self.output_type is list:
            gathered_data = [sample[field] for field in self.input_fields]
        elif self.output_type is torch.Tensor:
            gathered_data = torch.as_tensor([sample[field] for field in self.input_fields])
        else:
            raise ValueError(f"Unsupported output type: {self.output_type}. Supported types are dict and list.")
        sample[self.output_field] = gathered_data
        return sample

#----------------------------------------------------------------------------
# Misc data-processing transforms.

@beartype
class NameToIndexTransform:
    """Is used to augment with dataset IDs by mapping a string field to an index."""
    def __init__(self, field: str, name_to_index_mapping: dict[str, int]):
        self.field = field
        self.name_to_index_mapping = name_to_index_mapping

    def __call__(self, sample: SampleData) -> SampleData:
        assert self.field in sample, f"Field '{self.field}' not found in sample with keys {list(sample.keys())}."
        assert sample[self.field] in self.name_to_index_mapping, f"Value '{sample[self.field]}' not found in mapping for field '{self.field}'."
        sample[self.field] = self.name_to_index_mapping[sample[self.field]]
        return sample

@beartype
class RenameFieldsTransform:
    """Renames one or more fields in the sample."""
    def __init__(self, old_to_new_mapping: dict[str, str], strict: bool = True):
        self.old_to_new_mapping = old_to_new_mapping
        self.strict = strict

    def __call__(self, sample: SampleData) -> SampleData:
        for old_field, new_field in self.old_to_new_mapping.items():
            if old_field in sample:
                sample[new_field] = sample.pop(old_field)
            else:
                assert not self.strict, f"Field '{old_field}' not found in sample with keys {list(sample.keys())} while it's required."
        return sample

@beartype
class LoadFromDiskTransform:
    """Loads specified binary files from disk into memory."""
    def __init__(self, fields_to_load: Sequence[str]):
        assert len(fields_to_load) > 0, "At least one field must be specified to load from disk."
        self.fields_to_load = fields_to_load

    def __call__(self, sample: SampleData) -> SampleData:
        for field in self.fields_to_load:
            assert field in sample, f"Column {field} not found in sample with keys {list(sample.keys())}."
            with open(sample[field], 'rb') as f:
                sample[field] = f.read()
        return sample

@beartype
class FieldsFilteringTransform:
    """Filters fields in the sample data, keeping or removing specified fields."""
    def __init__(self, fields_to_keep: Sequence[str] | None = None, fields_to_remove: Sequence[str] | None = None):
        assert fields_to_keep is not None or fields_to_remove is not None, "At least one of fields_to_keep or fields_to_remove must be provided."
        self.fields_to_keep = fields_to_keep
        self.fields_to_remove = fields_to_remove

    def __call__(self, sample: SampleData) -> SampleData:
        if self.fields_to_remove is not None:
            for field in self.fields_to_remove:
                sample.pop(field, None)
        if self.fields_to_keep is not None:
            for field in list(sample.keys()):
                if field not in self.fields_to_keep:
                    sample.pop(field, None)
        return sample

@beartype
class AugmentNewFieldsTransform:
    """Augments the sample with new fields."""
    def __init__(self, new_fields: dict[str, Any]):
        self.new_fields = new_fields

    def __call__(self, sample: SampleData) -> SampleData:
        for field, value in self.new_fields.items():
            assert field not in sample, f"Field '{field}' already exists in sample with keys {list(sample.keys())}."
            sample[field] = value
        return sample

@beartype
class ConvertDTypeTransform:
    def __init__(self, input_field: str, output_field: str | None = None, output_dtype: torch.dtype = torch.float32):
        self.input_field = input_field
        self.output_field = output_field if output_field is not None else input_field
        self.output_dtype = output_dtype

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present={self.input_field: torch.Tensor}, absent=[])
        sample[self.output_field] = sample[self.input_field].to(self.output_dtype)
        return sample

@beartype
class EnsureFieldsTransform:
    """
    A transform which checks that the values for given fields are not None or empty.
    """
    def __init__(self, fields_whitelist: Sequence[str] | dict[str, type], check_dummy_values: bool = False, drop_others: bool=False):
        self.fields_whitelist = fields_whitelist
        self.check_dummy_values = check_dummy_values
        self.drop_others = drop_others

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=self.fields_whitelist, absent=[], check_dummy_values=self.check_dummy_values)
        if self.drop_others:
            for field in list(sample.keys()):
                if field not in self.fields_whitelist:
                    del sample[field]
        return sample

@beartype
class FindNonDummyValueTransform:
    """Searches over a given list of fields to find the very first non-dummy value."""
    def __init__(self, input_fields: Sequence[str], output_field: str):
        assert len(input_fields) > 0, "At least one input field must be specified."
        self.input_fields = input_fields
        self.output_field = output_field

    def __call__(self, sample: SampleData) -> SampleData:
        reasons = {}
        for field in self.input_fields:
            is_dummy, reason = is_dummy_field(sample, field, return_reason=True)
            if not is_dummy:
                sample[self.output_field] = sample[field]
                return sample
            reasons[field] = reason
        raise ValueError(f"All input fields are dummy. Reasons: {reasons}. Sample keys: {list(sample.keys())}.")

@beartype
class IdentityTransform:
    """A no-op transform that simply returns the sample as is."""
    def __call__(self, sample: SampleData) -> SampleData:
        return sample

#----------------------------------------------------------------------------
# Some composite pipelines for standard use cases. Should cover 80% of the cases.

@beartype
def create_standard_image_pipeline(
    image_field: str,
    resolution: tuple[int, int],
    return_image_as_single_frame_video: bool = False,
    normalize: bool=False,
    resize_kwargs: dict={},
    output_field: str='image', # In which field should we store the result.
    video_output_field: str='video', # If `return_image_as_single_frame_video` is enabled, the result will be stored in this field.
) -> Sequence[SampleTransform]:
    """Creates a standard image dataloading transform by composing transform classes."""
    transforms: Sequence[SampleTransform] = [
        LoadFromDiskTransform([image_field]),
        DecodeImageTransform(input_field=image_field, output_field=output_field),
        ResizeImageTransform(input_field=output_field, resolution=resolution, **resize_kwargs),
        ConvertImageToByteTensorTransform(input_field=output_field, output_field=output_field),
    ]

    if normalize:
        transforms.append(NormalizeFramesTransform(input_field=output_field))

    if return_image_as_single_frame_video:
        transforms.extend([
            ReshapeImageAsVideoTransform(input_field=output_field, output_field=video_output_field),
            FieldsFilteringTransform(fields_to_remove=[output_field]),
            AugmentNewFieldsTransform(new_fields=dict(framerate=960.0)),
        ])

    return transforms

@beartype
def create_standard_image_latent_pipeline(image_latent_field: str, return_image_as_single_frame_video: bool = False) -> Sequence[SampleTransform]:
    """Creates a standard image dataloading transform by composing transform classes."""
    transforms: Sequence[SampleTransform] = [
        LoadLatentFromDiskTransform(input_field=image_latent_field, output_field='image'),
        SampleImageVAELatentTransform(input_field='image'),
    ]

    if return_image_as_single_frame_video:
        transforms.extend([
            ReshapeImageAsVideoTransform(input_field='image', output_field='video'),
            FieldsFilteringTransform(fields_to_remove=['image']),
            AugmentNewFieldsTransform(new_fields=dict(framerate=960.0)),
        ])

    return transforms

@beartype
def create_standard_video_latent_pipeline(video_latent_field: str, framerate: float | None=None, random_offset: bool=True) -> Sequence[SampleTransform]:
    """Creates a standard video dataloading transform by composing transform classes."""
    transforms: Sequence[SampleTransform] = [
        LoadLatentFromDiskTransform(input_field=video_latent_field, output_field='video'),
        SampleVideoVAELatentTransform(input_field='video', framerate=framerate, random_offset=random_offset),
    ]

    return transforms

@beartype
def create_standard_metadata_pipeline(
    metadata_field: str,
    class_label_metadata_field: str | None = None,
    one_hot_encode_to_size: int | None = None,
    return_raw_metadata: bool = True,
    class_label_target_field: str = 'class_label'
) -> Sequence[SampleTransform]:
    """Creates a standard metadata processing pipeline by composing transform classes."""
    transforms: Sequence[SampleTransform] = [
        LoadFromDiskTransform([metadata_field]),
        LoadJsonMetadataTransform(input_field=metadata_field),
        RenameFieldsTransform(old_to_new_mapping={metadata_field: 'meta'}),
    ]
    if class_label_metadata_field is not None:
        transforms.append(ExtractMetadataSubfieldTransform(class_label_metadata_field, class_label_target_field))
        if one_hot_encode_to_size is not None:
            transforms.append(OneHotEncodeTransform(input_field=class_label_target_field, num_classes=one_hot_encode_to_size))
    else:
        assert one_hot_encode_to_size is None, f"one_hot_encode_to_size={one_hot_encode_to_size} is only applicable when class_label_metadata_field is provided."

    if not return_raw_metadata:
        transforms.append(FieldsFilteringTransform(fields_to_remove=['meta']))

    return transforms

@beartype
def create_standard_video_pipeline(
    video_field: str,
    num_frames: int,
    resolution: tuple[int, int], # Target spatial video resolution.
    decode_kwargs={}, # Extra decoding parameters for DecodeVideoTransform
    normalize: bool = False, # Whether to normalize the video frames to [-1, 1] range.
    original_resolution_fields: tuple[str, str] | None = None, # If provided, will undistort the video frames based on original height/width.
    resize_kwargs: dict = {}, # Extra resizing parameters for ResizeVideoTransform
    output_field: str = 'video', # In which field should we store the result.
):
    """
    Creates a standard text/video dataloading transform, which loads and decodes a video.
    """
    transforms: Sequence[SampleTransform] = [
        DecodeVideoTransform(input_field=video_field, output_field=output_field, num_frames=num_frames, **decode_kwargs),
        UndistortFramesTransform(input_field=output_field, original_resolution_fields=original_resolution_fields) if original_resolution_fields else IdentityTransform(),
        ResizeVideoTransform(input_field=output_field, resolution=resolution, **resize_kwargs),
        ConvertVideoToByteTensorTransform(input_field=output_field),
    ]

    if normalize:
        transforms.append(NormalizeFramesTransform(input_field=output_field))

    return transforms

@beartype
def create_standard_joint_video_audio_pipeline(
    video_field: str,
    num_frames: int,
    resolution: tuple[int, int], # Target spatial video resolution.
    decode_kwargs={}, # Extra decoding parameters for DecodeAudioFromVideoTransform
    mono_audio: bool = True,
    normalize_audio: bool = True, # Whether to normalize the audio by waveform / waveform.abs().max() * 0.95
    target_audio_sr: int = 16000, # Audio sampling rate
    resize_kwargs: dict = {}, # Extra resizing parameters for ResizeVideoTransform
    video_output_field: str = 'video', # In which field should we store the video result.
):
    transforms: Sequence[SampleTransform] = [
        DecodeVideoAndAudioTransform(
            input_field=video_field,
            video_output_field=video_output_field,
            original_clip_duration_output_field='original_clip_duration',
            audio_output_field='audio',
            original_sr_output_field='original_audio_sampling_rate',
            num_frames=num_frames,
            **decode_kwargs
        ),
        ResizeVideoTransform(input_field=video_output_field, resolution=resolution, **resize_kwargs),
        ConvertVideoToByteTensorTransform(input_field=video_output_field),
        ConvertAudioToFloatTensorTransform(input_field='audio'),
        ResizeAudioTransform(
            audio_input_field='audio',
            clip_duration_input_field='original_clip_duration',
            original_sr_input_field='original_audio_sampling_rate',
            target_audio_sr=target_audio_sr,
        ),
    ]
    if mono_audio:
        transforms.append(AverageAudioTransform(input_field='audio'))
    if normalize_audio:
        transforms.append(NormalizeAudioTransform(input_field='audio'))

    return transforms

#----------------------------------------------------------------------------
# Misc utils.

def _validate_fields(sample: SampleData, present: Sequence[str] | dict[str, type], absent: Sequence[str], check_dummy_values: bool=False) -> None:
    """Validates that all present are present in the sample."""
    for field in present:
        assert field in sample, f"Field '{field}' not found in sample with keys {list(sample.keys())}."
        if check_dummy_values:
            is_dummy, reason = is_dummy_field(sample, field, return_reason=True)
            assert not is_dummy, f"Field '{field}' is dummy: {reason} Sample keys: {list(sample.keys())}."
        if isinstance(present, dict) and present[field] is not None:
            expected_type = present[field]
            assert isinstance(sample[field], expected_type), f"Field '{field}' should be of type {expected_type}, but got {type(sample[field])}."
    for field in absent:
        assert field not in sample, f"Field '{field}' should not be present in sample with keys {list(sample.keys())}."

#----------------------------------------------------------------------------

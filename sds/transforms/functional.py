"""
A set of presets for image/video/audio decoding and processing.
"""
import io
import pickle
from typing import Union, Any
from collections.abc import Callable

from beartype import beartype
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch
import torchvision.transforms.functional as TVF

try:
    import torchaudio # Only import if available since it is an optional dependency.
except ImportError:
    torchaudio = None

from sds.structs import VIDEO_EXT
from sds.transforms.video_decoder import VideoDecoder
from sds.utils import os_utils

#----------------------------------------------------------------------------
# Image/video processing functions.

@beartype
def resize_image(image: Image.Image | torch.Tensor, **resize_kwargs) -> Image.Image | torch.Tensor:
    return lean_resize_frames([image], **resize_kwargs)[0]

@beartype
def lean_resize_frames(
        frames: list[Image.Image] | list[torch.Tensor] | torch.Tensor,
        resolution: tuple[int, int],
        crop_before_resize: bool=True,
        allow_vertical: bool=False,
        random_resize: dict[str, float] | None=None,
        interpolation_mode=TVF.InterpolationMode.LANCZOS,
    ) -> list[Image.Image] | list[torch.Tensor]:
    """
    Resizes each frame in the batch to the specified resolution.
    Possibly inverts it if it's vertical and allowed to do so.
    Also, can randomly downsample the frames given the `random_resize` dict of the form {(h,w): probability}.
    Args:
        - frames: List of frames to resize, either as PIL Images or torch Tensors.
        - resolution: Target resolution as a tuple (width, height).
        - crop_before_resize: If True, crops the frames to the target aspect ratio before resizing.
        - allow_vertical: If True, allows the frames to be resized to a vertical resolution via flipping input `resolution` as (width, height).
        - random_resize: A dictionary mapping resolutions to their probabilities for random downsampling.
        - interpolation_mode: Interpolation mode to use for resizing.
    Returns:
        - List of resized frames.
    """
    assert len(resolution) == 2, f"Wrong resolution: {resolution}"
    w, h = frames[0].size if isinstance(frames[0], Image.Image) else (frames[0].shape[2], frames[0].shape[1]) # [h, w]
    is_originally_vertical = h > w

    if random_resize is not None:
        assert sum(random_resize.values()) == 1.0, f"Probabilities should sum to 1.0: {random_resize}"
        random_resize = {k: v for k, v in random_resize.items() if k[0] <= w and k[1] <= h} # Only keep resolutions that are smaller than the original one.
        if len(random_resize) > 0:
            resolutions, probs = zip(*random_resize.items()) # [num_resolutions], [num_resolutions]
            resolution = resolutions[np.random.choice(len(resolutions), p=np.array(probs) / sum(probs))] # [2]

    h_trg, w_trg = (max(resolution), min(resolution)) if is_originally_vertical and allow_vertical else resolution

    if w == w_trg and h == h_trg:
        # TVF.resize has a similar shortcut, but here we won't even iterate.
        return frames

    if crop_before_resize:
        frames = [crop_to_aspect_ratio(x, target_aspect_ratio=w_trg / h_trg) for x in frames]
    frames = [TVF.resize(x, size=(h_trg, w_trg), interpolation=interpolation_mode) for x in frames]

    return frames

@beartype
def reshape_image_as_single_frame_video(image: torch.Tensor) -> torch.Tensor:
    # Returns a [c, h, w] image as [1, c, h, w] video.
    assert len(image.shape) == 3, f"Wrong shape: {image.shape}."
    assert image.shape[0] in (1, 3, 4), f"Wrong shape: {image.shape}."
    return image.unsqueeze(0) # [1, c, h, w]

@beartype
def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Loads an image from bytes and returns it as a PIL Image.
    TODO: we lose RGBA channels here, so maybe we should handle grayscale/RGB vs RGBA cases separately.
    """
    return Image.open(io.BytesIO(image_bytes)).convert('RGB') # Ensure the image is in RGB format

@beartype
def convert_pil_image_to_byte_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image)).permute(2, 0, 1)

@beartype
def load_image_tensor_from_bytes(image_bytes: bytes) -> torch.Tensor:
    image = load_image_from_bytes(image_bytes)
    image_pt = convert_pil_image_to_byte_tensor(image)
    return image_pt

@beartype
def crop_to_aspect_ratio(image: Image.Image | torch.Tensor, target_aspect_ratio: float) -> Image.Image | torch.Tensor:
    """Crops the image to the specified aspect ratio."""
    if isinstance(image, Image.Image):
        cur_w, cur_h = image.size
    elif isinstance(image, torch.Tensor):
        assert image.shape[0] in (1, 3, 4), f"Unsupported number of channels in shape {image.shape}. Must be 1, 3, or 4."
        _c, cur_h, cur_w = image.shape
    else:
        raise TypeError(f"Unsupported type: {type(image)}. Must be PIL.Image or torch.Tensor.")

    cur_aspect_ratio = cur_w / cur_h

    if cur_aspect_ratio > target_aspect_ratio:
        # Too wide: crop width
        new_width = int(cur_h * target_aspect_ratio)
        offset_left = (cur_w - new_width) // 2
        return _apply_crop(image, (offset_left, 0, offset_left + new_width, cur_h))
    else:
        # Too tall: crop height
        new_height = int(cur_w / target_aspect_ratio)
        offset_top = (cur_h - new_height) // 2
        return _apply_crop(image, (0, offset_top, cur_w, offset_top + new_height))

@beartype
def _apply_crop(x: Image.Image | torch.Tensor, crop: tuple[int, int, int, int]) -> Image.Image | torch.Tensor:
    if isinstance(x, Image.Image):
        return x.crop(crop)
    elif isinstance(x, torch.Tensor):
        return x[:, crop[1]:crop[3], crop[0]:crop[2]]

@beartype
def decode_frames_from_video(
        video_file: bytes | str | None=None,
        num_frames_to_extract: int=1,
        num_frames_total: int | None=None,
        video_decoder: VideoDecoder | None=None,
        random_offset: bool=True,
        frame_seek_timeout_sec: float=5.0,
        allow_shorter_videos: bool=False,
        framerate: float | None = None,
        thread_type: str | None = None,
    ) -> list[Image.Image]:
    """
    Decodes frames from a video file or bytes. Either video_file or video_decoder must be provided.
    """
    should_close_decoder = video_decoder is None
    if video_decoder is None:
        assert video_file is not None, "Video bytes must be provided if no video decoder is specified."
        assert isinstance(video_file, bytes) or os_utils.file_ext(video_file) in VIDEO_EXT, f"Unsupported video file type: {video_file}. Supported types: {VIDEO_EXT}."
        video_decoder = VideoDecoder(file=video_file, default_thread_type=thread_type)
        should_close_decoder = True # We should close it since it's us who opened it.
    if num_frames_total is None:
        num_frames_total = video_decoder.video_stream.frames # Relying on a guessed amount of frames in the video stream.

    base_framerate = video_decoder.framerate
    target_framerate = base_framerate if framerate is None else framerate

    num_frames_to_extract = min(num_frames_total, num_frames_to_extract) if allow_shorter_videos else num_frames_to_extract
    clip_duration_to_extract = num_frames_to_extract / target_framerate
    video_duration = num_frames_total / base_framerate
    assert video_duration >= clip_duration_to_extract or allow_shorter_videos, \
        f"Video duration {video_duration} is shorter than the requested clip duration {clip_duration_to_extract} while allow_shorter_videos={allow_shorter_videos}."
    start_frame_timestamp = np.random.rand() * max(video_duration - clip_duration_to_extract, 0.0) if random_offset else 0.0
    frame_timestamps = np.linspace(start_frame_timestamp, start_frame_timestamp + clip_duration_to_extract, num_frames_to_extract,)
    frame_timestamps = [t for t in frame_timestamps if t <= video_duration] # Filter out timestamps that are beyond the video duration.
    frames = video_decoder.decode_frames_at_times(frame_timestamps, frame_seek_timeout_sec=frame_seek_timeout_sec) # (num_frames, Image)

    if should_close_decoder:
        video_decoder.close()

    return frames

#----------------------------------------------------------------------------
# Audio processing functions.

# TODO.

#----------------------------------------------------------------------------
# Miscellaneous transforms.

@beartype
def load_pickle_embeddings(embeddings_path: str, num_tokens: int) -> torch.Tensor:
    """
    Loads raw pickle embeddings and converts them into a torch 2D tensor (with paddings).
    TODO: make it support a bytes input as well.
    """
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    embeddings = embeddings_data['embeddings'] # [num_tokens, d]
    embeddings[embeddings_data['eot_location'] + 1:] = 0.0 # Making sure that we have no junk after the EOT token.
    embeddings = embeddings[:num_tokens] # [num_tokens, d]
    paddings = torch.zeros((num_tokens - embeddings.shape[0], embeddings.shape[1]), dtype=embeddings.dtype, device=embeddings.device) # [num_paddings, d]
    embeddings = torch.cat([embeddings, paddings], axis=0) # [num_tokens, d]

    return embeddings

@beartype
def none_to_empty_str(data: dict[str, Union[str, None]]) -> dict[str, Any]:
    return {k: (v if v is not None else '') for k, v in data.items()}

def rename_fields(sample_data: dict[str, str], old_to_new_mapping: dict[str, str]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    pass

def one_hot_encode(label: int, num_classes: int) -> NDArray:
    """Converts a label to a one-hot encoded vector."""
    assert 0 <= label < num_classes, f"Label {label} is out of bounds for {num_classes} classes."
    one_hot = np.zeros(num_classes, dtype=np.float32)
    one_hot[label] = 1.0
    return one_hot

#----------------------------------------------------------------------------

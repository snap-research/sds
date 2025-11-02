"""
A set of presets for image/video/audio decoding and processing.
"""
import io
import math
import pickle
from typing import Union, Any, Sequence, BinaryIO

from beartype import beartype
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch
import torchvision.transforms.functional as TVF
from loguru import logger

try:
    import torchaudio # Only import if available since it is an optional dependency.
except ImportError:
    torchaudio = None
except Exception as e:
    logger.error(f"torchaudio installation is broken: {e}")
    torchaudio = None

try:
    import audiotools
except ImportError:
    audiotools = None

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
    frames: Sequence[Image.Image] | Sequence[torch.Tensor] | torch.Tensor,
    resolution: tuple[int, int],
    crop_before_resize: bool=True,
    allow_vertical: bool=False,
    random_resize: dict[str, float] | None=None,
    interpolation_mode='bilinear',
) -> Sequence[Image.Image] | Sequence[torch.Tensor]:
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
    frames = [TVF.resize(x, size=(h_trg, w_trg), interpolation=TVF.InterpolationMode(interpolation_mode)) for x in frames]

    return frames

@beartype
def reshape_image_as_single_frame_video(image: torch.Tensor) -> torch.Tensor:
    # Returns a [c, h, w] image as [1, c, h, w] video.
    assert len(image.shape) == 3, f"Wrong shape: {image.shape}."
    return image.unsqueeze(0) # [1, c, h, w]

@beartype
def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Loads an image from bytes and returns it as a PIL Image.
    TODO: we lose RGBA channels here, so maybe we should handle grayscale/RGB vs RGBA cases separately.
    """
    return Image.open(io.BytesIO(image_bytes)).convert('RGB') # Ensure the image is in RGB format

@beartype
def convert_pil_image_to_byte_tensor(image: Image.Image, convert_to_rgb: bool=True, cut_alpha: bool=False) -> torch.Tensor:
    img_pt = torch.from_numpy(np.array(image)) # [h, w, c | null]
    img_pt = img_pt.unsqueeze(-1) if img_pt.ndim == 2 else img_pt # [h, w, 1]
    img_pt = img_pt.repeat(1, 1, 3) if img_pt.shape[2] == 1 and convert_to_rgb else img_pt # [h, w, 3]
    img_pt = img_pt[:, :, :3] if img_pt.shape[2] > 3 and cut_alpha else img_pt # [h, w, 3]
    img_pt = img_pt.permute(2, 0, 1) # [3, h, w]

    return img_pt # [3, h, w]

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
def _maybe_expand_crop_interval(crop_interval: tuple[float, float] | None, crop_interval_type: str, clip_duration: float, full_video_duration: float) -> tuple[float, float]:
    """
    Expands the crop interval to be at least clip_duration long, while not exceeding full_video_duration.
    """

    assert clip_duration <= full_video_duration, f"Clip duration {clip_duration} is longer than the full video duration {full_video_duration}."
    if crop_interval_type == "hard" and crop_interval is not None and (crop_interval[1] - crop_interval[0]) < clip_duration:
        raise ValueError(f"Provided crop_interval {crop_interval} is shorter than the requested clip duration {clip_duration} while crop_interval_type is 'hard'.")
    if crop_interval_type == "hard" and crop_interval is not None and (crop_interval[0] < 0.0 or crop_interval[1] > full_video_duration):
        raise ValueError(f"Provided crop_interval {crop_interval} is out of bounds [0.0, {full_video_duration}] while crop_interval_type is 'hard'.")

    if crop_interval is None:
        return (0.0, full_video_duration)

    crop_interval = (max(0.0, crop_interval[0]), min(full_video_duration, crop_interval[1]))
    sequence_crop_duration = crop_interval[1] - crop_interval[0]
    # If needed, expand the sequence crop duration equally on each side to fit the clip duration
    if sequence_crop_duration < clip_duration:
        assert(crop_interval_type == "soft"), f"Crop interval type is 'hard' but the provided crop interval {crop_interval} is shorter than the requested clip duration {clip_duration}."

        padding = (clip_duration - sequence_crop_duration) / 2
        crop_interval = (crop_interval[0] - padding, crop_interval[1] + padding)
        # Shifts to the left if we went beyond the video limits
        if crop_interval[1] > full_video_duration:
            shift = crop_interval[1] - full_video_duration
            crop_interval = (crop_interval[0] - shift, full_video_duration)
        # Shifts to the right if we went beyond the video limits
        if crop_interval[0] < 0.0:
            shift = -crop_interval[0]
            crop_interval = (0.0, crop_interval[1] + shift)

        # Ensures we are within bounds
        crop_interval = (max(0.0, crop_interval[0]), min(full_video_duration, crop_interval[1]))

    return crop_interval

@beartype
def decode_video(
    video_file: BinaryIO | str | None=None,
    num_frames_to_extract: int=1,
    video_decoder: VideoDecoder | None=None,
    random_offset: bool=True,
    frame_seek_timeout_sec: float=5.0,
    allow_shorter_videos: bool=False,
    framerate: float | None = None,
    thread_type: str | None = None,
    return_audio: bool = False,
    approx_frame_seek: bool = False,
    frame_timestamps: Sequence[float] | None = None, # If provided, we ignore the internally computed frame_timestamps and use the provided ones instead.
    real_duration: float | None = None, # If provided, we ignore the video metadata.
    real_framerate: float | None = None, # If provided, we ignore the video metadata.
    speedup_factor: Union[int, float, None] = None, # If provided, we simulate video speed by this factor (i.e., we divide the framerate by this factor).
    crop_interval: tuple[float, float] | None = None, # If provided, use this as a interval to crop the video from which to extract frames.
    crop_interval_type: str | None = None # One of "hard" or "soft". If "hard", we strictly crop within the provided crop_interval. If "soft", we expand it if needed to fit the requested clip duration.
) -> tuple[Sequence[Image.Image], list[float], float, NDArray | None, int | None]:
    """
    Decodes frames from a video file or bytes. Either video_file or video_decoder must be provided.
    """
    if crop_interval_type is None:
        crop_interval_type = "hard"

    should_close_decoder = video_decoder is None
    if video_decoder is None:
        assert video_file is not None, "Video bytes must be provided if no video decoder is specified."
        assert not isinstance(video_file, str) or os_utils.file_ext(video_file) in VIDEO_EXT, f"Unsupported video file type: {video_file}. Supported types: {VIDEO_EXT}."
        video_decoder = VideoDecoder(file=video_file, default_thread_type=thread_type)
        should_close_decoder = True # We should close it since it's us who opened it.

    # Computing the full and target video durations and framerates.
    # AV video metadata is not always accurate, so we can opt for using our pre-computed one.
    base_framerate = video_decoder.framerate if real_framerate is None else real_framerate
    target_framerate = base_framerate if framerate is None else framerate
    target_framerate = target_framerate / speedup_factor if speedup_factor is not None else target_framerate  # Simulating speedup by reducing the framerate.
    full_video_duration = (video_decoder.video_stream.frames / base_framerate) if real_duration is None else real_duration
    clip_duration = num_frames_to_extract / target_framerate

    if clip_duration > full_video_duration:
        assert allow_shorter_videos, f"Video duration {full_video_duration} is shorter than the requested clip duration {clip_duration} while allow_shorter_videos={allow_shorter_videos}."
        num_frames_to_extract = max(1, math.floor(full_video_duration * target_framerate))
        clip_duration = num_frames_to_extract / target_framerate  # We changed the number of frames so we must update the clip duration

    # Adjusting the sequence crop time to conform to the requested clip duration
    crop_interval = _maybe_expand_crop_interval(crop_interval, crop_interval_type, clip_duration, full_video_duration)

    if frame_timestamps is None:
        # Computes the base spacing for frames extraction according to the target framerate
        frame_timestamps = np.linspace(0, max(clip_duration - 1 / target_framerate, 0), num_frames_to_extract)
        sequence_crop_duration = crop_interval[1] - crop_interval[0]
        start_ts = np.random.rand() * max(sequence_crop_duration - clip_duration, 0.0) + crop_interval[0] if random_offset else crop_interval[0]
        frame_timestamps = start_ts + frame_timestamps
        frame_timestamps = [min(t.item(), full_video_duration) for t in frame_timestamps]
    else:
        start_ts: float = frame_timestamps[0]
    decoding_fn = video_decoder.decode_frames_at_times_approx if approx_frame_seek else video_decoder.decode_frames_at_times
    frames = decoding_fn(frame_timestamps, frame_seek_timeout_sec=frame_seek_timeout_sec) # (num_frames, Image)

    if return_audio:
        waveform, sampling_rate = decode_audio_from_video_decoder(video_decoder, start_ts=start_ts, end_ts=start_ts + clip_duration)
    else:
        waveform = sampling_rate = None

    if should_close_decoder:
        video_decoder.close()

    return frames, frame_timestamps, clip_duration, waveform, sampling_rate

#----------------------------------------------------------------------------
# Audio processing functions.

@beartype
def decode_audio(
    audio_file_path: str,
    duration: float | None = None,
    clip_offset_strategy: str = 'random',
    allow_shorter_audio: bool = False,
    loudness_cutoff: float = -40.0, # in dBFS, only used for 'random_loud' strategy.
) -> tuple[torch.Tensor, int]:
    """Decodes the audio waveform from wav/mp3/flac file or bytes."""
    if clip_offset_strategy == 'random_loud':
        assert audiotools is not None, "audiotools is not available, please run: `pip install git+https://github.com/descriptinc/audiotools`"
        audio_signal = audiotools.AudioSignal.salient_excerpt(audio_file_path, duration=duration, loudness_cutoff=loudness_cutoff)
        assert audio_signal.audio_data.shape[0] == 1, f'Wrong batch size from audiotools: {audio_signal.audio_data.shape}.'
        return audio_signal.audio_data.squeeze(0), audio_signal.sample_rate

    waveform, sr = torchaudio.load(audio_file_path)
    full_duration = waveform.shape[1] / sr
    assert duration is None or full_duration >= duration or allow_shorter_audio, f"Audio duration {full_duration} is shorter than the requested duration {duration} while allow_shorter_audio={allow_shorter_audio}."
    duration = full_duration if duration is None else min(duration, full_duration)
    num_trg_samples = int(sr * duration)

    if duration == full_duration:
        start_id = 0
    elif clip_offset_strategy == 'random':
        start_id = np.random.randint(0, waveform.shape[1] - num_trg_samples + 1)
    elif clip_offset_strategy == 'max_mean':
        start_id = max_mean_window_idx(waveform, window=num_trg_samples)
    elif clip_offset_strategy == 'max_center':
        max_val_id = waveform.abs().mean(dim=0).argmax().item()
        start_id = max_val_id - num_trg_samples // 2
        start_id = max(0, min(start_id, waveform.shape[1] - num_trg_samples))
    elif clip_offset_strategy == 'start':
        start_id = 0
    else:
        raise ValueError(f"Unsupported clip_offset_strategy: {clip_offset_strategy}.")

    waveform = waveform[:, start_id:start_id + num_trg_samples] # [c, num_trg_samples]

    return waveform, sr


def decode_audio_from_video_decoder(video_decoder: VideoDecoder, start_ts: float, end_ts: float | None = None) -> tuple[NDArray, int]:
    audio_stream = next(s for s in video_decoder.container.streams if s.type == 'audio')
    time_base = audio_stream.time_base

    # compute presentation timestamp
    start_pts = int(start_ts / time_base)
    end_pts = int(end_ts / time_base) if end_ts is not None else None

    # Any frame disable seeking key frames which can be slower
    video_decoder.container.seek(start_pts, stream=audio_stream, any_frame=False)

    # Decode audio frames and convert to numpy
    audio_frames = []
    for frame in video_decoder.container.decode(audio_stream):
        if frame.pts < start_pts:
            continue
        if end_pts is not None and frame.pts > end_pts:
            break

        frame_np = frame.to_ndarray() # [n_channels, T_i]
        frame_np = frame_np[None, :] if len(frame_np.shape) == 1 else frame_np  # Ensure it's 2D [n_channels, T_i]
        audio_frames.append(frame_np)

    if audio_frames:
        waveform = np.concatenate(audio_frames, axis=1) # [n_channels, T_1 + T_2 + ... + T_n]
    else:
        waveform = np.zeros((1, 0), dtype=np.float32)  # No audio frames decoded, return empty waveform

    return waveform, audio_stream.rate


@beartype
def resample_waveform(waveform: torch.Tensor, **resampling_kwargs) -> torch.Tensor:
    assert torchaudio is not None, "torchaudio is not available. Please install it to use audio resampling."
    return torchaudio.functional.resample(waveform, **resampling_kwargs)


@beartype
def resize_waveform(waveform: torch.Tensor, target_length: int, mode: str='pad_or_trim') -> torch.Tensor:
    """
    Pads or trims the waveform to the target length.
    Assumes the shape is [..., num_channels, time]
    TODO: should we do interpolation?
    """
    assert mode in ['pad_or_trim'], f"Unsupported mode: {mode}. Supported modes: ['pad_or_trim']."

    if waveform.shape[-1] == target_length:
        return waveform
    elif waveform.shape[-1] < target_length:
        # Pad the waveform to the target length
        padding_size = target_length - waveform.shape[-1]
        padding = torch.zeros((*waveform.shape[:-1], padding_size), dtype=waveform.dtype, device=waveform.device)
        return torch.cat([waveform, padding], dim=-1) # [..., num_channels, target_length]
    else:
        return waveform[..., :target_length] # [..., num_channels, target_length]


def max_mean_window_idx(x: torch.Tensor, window: int):
    """Returns the starting index of the window with the maximum mean absolute value."""
    assert x.ndim == 2, f"Expected a 2D tensor, got {x.shape}."
    assert window > 0 and window <= x.shape[1], f"Window size {window} is out of bounds for tensor with shape {x.shape}."
    x = x.abs().mean(0) # [t]
    cumsum = x.cumsum(dim=0) # [t]
    cumsum_window = cumsum[window:] - cumsum[:-window] # [t - window]
    return torch.argmax(cumsum_window).item()

#----------------------------------------------------------------------------
# VAE latents processing functions.

@beartype
def load_torch_state_from_pickle(latents_path: str, non_torch_fields: Sequence[str] | None=None) -> dict[str, torch.Tensor]:
    with open(latents_path, 'rb') as f:
        state = pickle.load(f)
    non_torch_fields = non_torch_fields if non_torch_fields is not None else []
    state = {k: v if k in non_torch_fields else torch.tensor(v, dtype=torch.float32) for k, v in state.items()}

    return state

@beartype
def sample_image_vae_latents(latents_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    mean, logvar = latents_dict['mean'], latents_dict['logvar'] # [lc, lh, lw], [lc, lh, lw]

    assert mean.ndim == 3, f"Unsupported latent shape: {mean.shape}. Expected 3D tensor."
    assert logvar.shape == mean.shape, f"Mean and logvar shapes do not match: {mean.shape} vs {logvar.shape}."

    return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) # [lc, lh, lw]

@beartype
def sample_video_vae_latents(
        latents_dict: dict[str, torch.Tensor],
        orig_shape: tuple[int, int, int, int],
        num_rgb_frames_to_extract: int | None=None,
        fps_orig: float | None=None,
        fps_trg: float | None=None,
        random_offset: bool=True,
    ) -> torch.Tensor:

    mean, logvar = latents_dict['mean'], latents_dict['logvar'] # [lt | null, lc, lh, lw], [lt | null, lc, lh, lw]

    assert mean.ndim == 4, f"Unsupported latent shape: {mean.shape}. Expected 4D tensor."
    assert logvar.shape == mean.shape, f"Mean and logvar shapes do not match: {mean.shape} vs {logvar.shape}."

    temporal_compression_rate = math.ceil(orig_shape[0] / mean.shape[0])
    num_rgb_frames_to_extract = orig_shape[0] if num_rgb_frames_to_extract is None else num_rgb_frames_to_extract
    num_latent_frames_to_extract = math.ceil(num_rgb_frames_to_extract / temporal_compression_rate)
    if fps_trg is not None:
        assert fps_orig is not None, "Original FPS must be provided if target FPS is specified."
        assert (fps_orig / fps_trg).is_integer() and fps_orig >= fps_trg, f"FPS ratio {fps_orig} / {fps_trg} is not a positive integer. For latents, we cant decode at arbitrary framerates."
        frames_skip_factor = int(fps_orig / fps_trg)
    else:
        frames_skip_factor = 1

    assert (num_latent_frames_to_extract * frames_skip_factor) <= mean.shape[0], f"Requested {num_latent_frames_to_extract} latent frames with skip factor {frames_skip_factor}, but only {mean.shape[0]} are available in the latents."

    start_frame_idx = np.random.randint(low=0, high=mean.shape[0] - num_latent_frames_to_extract + 1) if random_offset else 0
    latent_frames_idx = np.arange(start_frame_idx, start_frame_idx + num_latent_frames_to_extract, frames_skip_factor)
    latent_frames_mean = mean[latent_frames_idx]  # [clip_length, lc, lh, lw]
    latent_frames_logvar = logvar[latent_frames_idx]  # [clip_length, lc, lh, lw]

    frames = latent_frames_mean + torch.randn_like(latent_frames_logvar) * np.exp(0.5 * latent_frames_logvar) # [clip_length | null, lc, lh, lw]

    return frames

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
    embeddings = torch.from_numpy(embeddings) if isinstance(embeddings, np.ndarray) else embeddings # [num_tokens, d]
    paddings = torch.zeros((num_tokens - embeddings.shape[0], embeddings.shape[1]), dtype=embeddings.dtype, device=embeddings.device) # [num_paddings, d]
    embeddings = torch.cat([embeddings, paddings], axis=0) # [num_tokens, d]

    return embeddings

@beartype
def none_to_empty_str(data: dict[str, Union[str, None]]) -> dict[str, Any]:
    return {k: (v if v is not None else '') for k, v in data.items()}

def one_hot_encode(label: int, num_classes: int) -> NDArray:
    """Converts a label to a one-hot encoded vector."""
    assert 0 <= label < num_classes, f"Label {label} is out of bounds for {num_classes} classes."
    one_hot = np.zeros(num_classes, dtype=np.float32)
    one_hot[label] = 1.0
    return one_hot

#----------------------------------------------------------------------------

from enum import Enum
from typing import Any, Callable, TypeAlias, Generator
from dataclasses import dataclass
from PIL import Image

#---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 60.0 # In seconds.

#---------------------------------------------------------------------------
# Data-related constants and types.

class DataSampleType(Enum):
    IMAGE = 'IMAGE'
    VIDEO = 'VIDEO'
    AUDIO = 'AUDIO'
    TEXT = 'TEXT'
    IMAGE_LATENT = 'IMAGE_LATENT'
    VIDEO_LATENT = 'VIDEO_LATENT'
    AUDIO_LATENT = 'AUDIO_LATENT'
    TEXT_LATENT = 'TEXT_LATENT'

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)

    @staticmethod
    def from_str(type_str: str) -> "DataSampleType":
        return DataSampleType[type_str.upper()]

    @staticmethod
    def from_ext(file_ext: str) -> "DataSampleType":
        ext = file_ext.lower()
        file_ext = f'.{ext}' if not ext.startswith('.') else ext
        for data_type, exts in DATA_TYPE_TO_EXT.items():
            if ext in exts:
                return data_type
        raise ValueError(f"Unknown file extension: {file_ext}")


IMAGE_EXT = {ext.lower() for ext in Image.registered_extensions()}
VIDEO_EXT = {'.avi', '.mp4', '.webm', '.mov', '.mkv', '.m4v', '.mpg', '.mpeg', '.wmv', '.flv', '.3gp', '.3g2'}
AUDIO_EXT = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
TEXT_EXT  = {'.txt', '.json', '.csv', '.xml', '.md', '.html', '.pdf', '.docx', '.rtf'}
LATENT_EXT = {'.pkl', '.pt', '.ckpt', '.npy', '.npz', '.h5', '.hdf5'}

DATA_TYPE_TO_EXT = {
    DataSampleType.IMAGE: IMAGE_EXT,
    DataSampleType.VIDEO: VIDEO_EXT,
    DataSampleType.AUDIO: AUDIO_EXT,
    DataSampleType.TEXT: TEXT_EXT,
    DataSampleType.IMAGE_LATENT: LATENT_EXT,
    DataSampleType.VIDEO_LATENT: LATENT_EXT,
    DataSampleType.AUDIO_LATENT: LATENT_EXT,
    DataSampleType.TEXT_LATENT: LATENT_EXT,
}

EXT_TO_DATA_TYPE = {
    **dict.fromkeys(IMAGE_EXT, 'image'),
    **dict.fromkeys(VIDEO_EXT, 'video'),
    **dict.fromkeys(AUDIO_EXT, 'audio'),
    **dict.fromkeys(TEXT_EXT, 'text'),
    **dict.fromkeys(LATENT_EXT, 'latent'),
}

# Sample dict and sample transform type aliases.
SampleData = dict[str, Any]
SampleTransform: TypeAlias = Callable[[SampleData], SampleData | Generator[SampleData, None, None]]

#---------------------------------------------------------------------------

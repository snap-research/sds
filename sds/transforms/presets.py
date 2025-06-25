"""
Transform presets for common data processing tasks in the SDS framework.
They are structured as classes because they need to be pickleable for distributed processing
(otherwise, torch.data.utils.DataLoader will fail to pickle them for num_workers > 0).
"""
import io
import json
from typing import Any

from beartype import beartype
from PIL import Image

from sds.structs import SampleData, SampleTransform
import sds.transforms.functional as SDF

#----------------------------------------------------------------------------
# Misc utils.

def _validate_fields(sample: SampleData, present: list[str] | dict[str, type], absent: list[str]) -> None:
    """Validates that all present are present in the sample."""
    for field in present:
        assert field in sample, f"Field '{field}' not found in sample with keys {list(sample.keys())}."
        if isinstance(present, dict) and present[field] is not None:
            expected_type = present[field]
            assert isinstance(sample[field], expected_type), f"Field '{field}' should be of type {expected_type}, but got {type(sample[field])}."
    for field in absent:
        assert field not in sample, f"Field '{field}' should not be present in sample with keys {list(sample.keys())}."

#----------------------------------------------------------------------------
# Image processing transforms.

@beartype
class DecodeImageTransform:
    """Decodes an image file into a [c, h, w] torch tensor."""
    def __init__(self, input_field: str, output_field: str | None = None):
        self.input_field = input_field
        self.output_field = output_field if output_field is not None else input_field

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=[self.input_field], absent=[])
        sample[self.output_field] = SDF.load_image_from_bytes(sample[self.input_field])
        return sample

@beartype
class ResizeImageTransform:
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
class ReshapeImageAsVideoTransform:
    """Reshapes a single image tensor into a single-frame video tensor."""
    def __init__(self, input_field: str, output_field: str):
        self.input_field = input_field
        self.output_field = output_field

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=[self.input_field], absent=[self.output_field])
        sample[self.output_field] = SDF.reshape_image_as_single_frame_video(sample[self.input_field])
        return sample

@beartype
class ConvertImageToByteTensorTransform:
    """Converts an image to a torch tensor."""
    def __init__(self, input_field: str, output_field: str | None = None):
        self.input_field = input_field
        self.output_field = output_field if output_field is not None else input_field

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present={self.input_field: Image.Image}, absent=[])
        sample[self.output_field] = SDF.convert_pil_image_to_byte_tensor(sample[self.input_field])
        return sample

#----------------------------------------------------------------------------
# Video processing transforms.

@beartype
class DecodeVideoTransform:
    """A video transform which decodes a video file into a list of frames."""
    def __init__(self, input_field: str, output_field: str, **decoding_kwargs):
        self.input_field = input_field
        self.output_field = output_field
        self.decoding_kwargs = decoding_kwargs

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=[self.input_field], absent=[])
        sample[self.output_field] = SDF.decode_frames_from_video(sample[self.input_field], **self.decoding_kwargs)
        return sample

@beartype
class ResizeVideoTransform:
    """Resizes each frame in the video to the specified resolution."""
    def __init__(self, input_field: str, output_field: str | None = None, **resize_kwargs):
        self.input_field = input_field
        self.output_field = output_field if output_field is not None else input_field
        self.resize_kwargs = resize_kwargs

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=[self.input_field], absent=[])
        sample[self.output_field] = SDF.lean_resize_frames(frames=sample[self.input_field], **self.resize_kwargs)
        return sample

#----------------------------------------------------------------------------
# Label/text processing transforms.

@beartype
class PickleLoaderTransform:
    """Loads embeddings from a pickle file."""
    def __init__(self, input_field: str, label_shape: tuple[int], output_field: str | None = None):
        self.input_field = input_field
        self.output_field = output_field if output_field is not None else input_field
        self.label_shape = label_shape

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present=[self.input_field], absent=[])
        sample[self.output_field] = SDF.convert_pickle_embeddings_to_numpy(sample[self.input_field], self.label_shape)
        return sample

@beartype
class LoadJsonMetadataTransform:
    """Loads metadata from a JSON file."""
    def __init__(self, input_field: str, output_field: str | None = None):
        self.input_field = input_field
        self.output_field = output_field if output_field is not None else input_field

    def __call__(self, sample: SampleData) -> SampleData:
        _validate_fields(sample, present={self.input_field: bytes}, absent=[])
        sample[self.output_field] = json.loads(sample[self.input_field])
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

#----------------------------------------------------------------------------
# Misc data-processing transforms.

@beartype
class NameToIndexMappingTransform:
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
    def __init__(self, old_to_new_mapping: dict[str, str]):
        self.old_to_new_mapping = old_to_new_mapping

    def __call__(self, sample: SampleData) -> SampleData:
        for old_field, new_field in self.old_to_new_mapping.items():
            assert old_field in sample, f"Field '{old_field}' not found in sample with keys {list(sample.keys())}."
            sample[new_field] = sample.pop(old_field)
        return sample

@beartype
class LoadFromDiskTransform:
    """Loads specified binary files from disk into memory."""
    def __init__(self, fields_to_load: list[str]):
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
    def __init__(self, fields_to_keep: list[str] | None = None, fields_to_remove: list[str] | None = None):
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

#----------------------------------------------------------------------------
# Some composite pipelines for standard use cases. Should cover 80% of the cases.

@beartype
def create_standard_image_pipeline(image_field: str, return_image_as_single_frame_video: bool = False, **resize_kwargs) -> list[SampleTransform]:
    """Creates a standard image dataloading transform by composing transform classes."""
    transforms: list[SampleTransform] = [
        LoadFromDiskTransform([image_field]),
        RenameFieldsTransform(old_to_new_mapping={image_field: 'image'}),
        DecodeImageTransform(input_field='image', output_field='image'),
        ResizeImageTransform(input_field='image', **resize_kwargs),
        ConvertImageToByteTensorTransform(input_field='image', output_field='image'),
    ]

    if return_image_as_single_frame_video:
        transforms.extend([
            ReshapeImageAsVideoTransform(input_field='image', output_field='video'),
            FieldsFilteringTransform(fields_to_remove=['image']),
            AugmentNewFieldsTransform(new_fields=dict(framerate=960.0)),
        ])

    return transforms

@beartype
def create_standard_metadata_pipeline(
    metadata_field: str,
    class_label_metadata_field: str | None = None,
    one_hot_encode_to_size: int | None = None,
    return_raw_metadata: bool = True,
) -> list[SampleTransform]:
    """Creates a standard metadata processing pipeline by composing transform classes."""
    transforms: list[SampleTransform] = [
        LoadFromDiskTransform([metadata_field]),
        LoadJsonMetadataTransform(input_field=metadata_field),
        RenameFieldsTransform(old_to_new_mapping={metadata_field: 'meta'}),
    ]
    if class_label_metadata_field is not None:
        transforms.append(ExtractMetadataSubfieldTransform(class_label_metadata_field, 'class_label'))
        if one_hot_encode_to_size is not None:
            transforms.append(OneHotEncodeTransform(input_field='class_label', num_classes=one_hot_encode_to_size))
    else:
        assert one_hot_encode_to_size is None, f"one_hot_encode_to_size={one_hot_encode_to_size} is only applicable when class_label_metadata_field is provided."

    if not return_raw_metadata:
        transforms.append(FieldsFilteringTransform(fields_to_remove=['meta']))

    return transforms

# Note: These pipelines can now be built using the class-based transforms above.
@beartype
def create_standard_video_pipeline():
    """
    Creates a standard text/video dataloading transform, which loads and decodes a video.
    """
    raise NotImplementedError

@beartype
def create_standard_text_image_pipeline():
    raise NotImplementedError

@beartype
def create_standard_text_video_pipeline():
    raise NotImplementedError

#----------------------------------------------------------------------------

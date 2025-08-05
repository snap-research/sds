from enum import Enum
import math
from typing import Any, Iterator
from dataclasses import dataclass, field

import numpy as np
import torch

from sds.structs import DataSampleType
from sds.utils import misc
from sds.utils import os_utils
import sds.utils.distributed as dist_utils

#----------------------------------------------------------------------------
# Helper enums and structs.

class ScheduleType(Enum):
    RANDOM = 'random'
    ROUND_ROBIN = 'round_robin'
    SHUFFLED_ROUND_ROBIN = 'shuffled_round_robin'

    @classmethod
    def from_str(cls, schedule: str) -> 'ScheduleType':
        if schedule == 'random':
            return cls.RANDOM
        elif schedule == 'round_robin':
            return cls.ROUND_ROBIN
        elif schedule == 'shuffled_round_robin':
            return cls.SHUFFLED_ROUND_ROBIN
        else:
            raise ValueError(f"Unsupported schedule: {schedule}. Supported schedules are 'random', 'round_robin', and 'shuffled_round_robin'.")


class Batch(dict):
    """A clumsy dict wrapper to augment batches with additional metadata."""
    def __init__(self, raw_batch: dict, stream_name: str | None = None, num_accum_rounds_left: int = 0, data_type: DataSampleType | None=None):
        assert isinstance(raw_batch, dict), f"Expected batch to be a dict, but got {type(raw_batch)}. Make sure the dataset collates batches into a dict."
        super().__init__(**raw_batch) # Initialize the dict with the raw batch.
        self.stream_name = stream_name
        self.num_accum_rounds_left = num_accum_rounds_left
        self.data_type = data_type


@dataclass(frozen=True)
class StreamOptions:
    """
    Contains dataset configuration details with some extra dataloading options/information.
    TODO: it would be great if we could init the dataset from it...
    """
    name: str
    ratio: float
    dataloader_kwargs: dict[str, Any] = field(default_factory=dict)
    is_main: bool = False # We want to know what's the "main" stream to compute visualizations/statistics.
    batch_size: int | None = None # Total batch size across all the ranks.
    batch_gpu: int | None = None # Maximum batch size per GPU.
    num_accum_rounds: int = 1 # Number of gradient accumulation rounds.

    @staticmethod
    def infer_batch_info(batch_size: int | None = None, batch_gpu: int | None = None, num_accum_rounds: int | None = None) -> tuple[int, int, int]:
        """
        Fixes batch config options (with some values missing) into the fully-filled one.
        We provide the option to specify the batch size either through batch size or batch_gpu + num_accum_rounds.
        """
        world_size = dist_utils.get_world_size()
        if batch_size is None:
            assert not batch_gpu is None, f"If batch_size={batch_size} is None, batch_gpu={batch_gpu} must be specified."
            num_accum_rounds = 1 if num_accum_rounds is None else int(num_accum_rounds)
            batch_size = int(batch_gpu * world_size * num_accum_rounds)
        else:
            # Note: batch_size/batch_gpu/num_accum_rounds can be equal to 0 --- that means that we don't train on a given dataset.
            assert batch_size % world_size == 0, f"batch_size={batch_size} must be divisible by world_size={world_size}"
            assert batch_gpu is None or (batch_size // world_size) % batch_gpu == 0, f"If batch_size is specified, batch_gpu must be divisible by (batch_size={batch_size} // world_size={world_size})"
            batch_gpu = (batch_size // world_size) if batch_gpu is None else int(batch_gpu)
            num_accum_rounds = 0 if batch_gpu == 0 else (batch_size // (batch_gpu * world_size))

        return batch_size, batch_gpu, num_accum_rounds

    @staticmethod
    def init_group(raw_stream_configs: list[dict[str, dict | str]], drop_unused: bool=False) -> list["StreamOptions"]:
        """
        Initializes a group of streams using their configs in a way that the ratios are correctly normalized.
        TODO: it contains some fields which are not present in the StreamOptions, but are used in the dataset...
        """
        if any(s.get('ratio') is not None for s in raw_stream_configs):
            assert all(s.get('ratio') is not None for s in raw_stream_configs), f"If one stream has a ratio, all streams must have ratios: {raw_stream_configs}"
            ratios = misc.normalize_ratios([s['ratio'] for s in raw_stream_configs])
        else:
            ratios = np.array([1.0 / len(raw_stream_configs)] * len(raw_stream_configs))

        raw_stream_configs = [raw_stream_configs[i] for i in np.where(ratios > 0)[0]] if drop_unused else raw_stream_configs
        stream_configs = []
        unique_names = set()
        for i, raw_config in enumerate(raw_stream_configs):
            # Selecting the name.
            name = raw_config.get('name', f'stream_{i:03d}')
            assert not name in unique_names, f"Stream names must be unique. Found duplicate name: {name}."
            unique_names.add(name)

            # Validating and deciding the batching information.
            batch_size, batch_gpu, num_accum_rounds = StreamOptions.infer_batch_info(
                batch_size=raw_config.get('batch_size'),
                batch_gpu=raw_config.get('batch_gpu'),
                num_accum_rounds=raw_config.get('num_accum_rounds'),
            )

            stream_configs.append(StreamOptions(
                ratio=ratios[i],
                is_main=raw_config.get('is_main', False),
                dataloader_kwargs=raw_config.get('dataloader_kwargs', {}),
                name=name,
                batch_size=batch_size,
                batch_gpu=batch_gpu,
                num_accum_rounds=num_accum_rounds,
            ))

        return stream_configs


@dataclass
class Stream:
    dataset: torch.utils.data.Dataset
    opts: StreamOptions
    iterator: Iterator
    dataloader: torch.utils.data.DataLoader

#----------------------------------------------------------------------------

class MultiStreamDataLoader:
    def __init__(
            self,
            datasets: list[torch.utils.data.Dataset],
            stream_opts: list[StreamOptions],
            num_workers: int = 0,
            shuffle_seed: int | None = 42,
            schedule: str = 'shuffled_round_robin',
            **common_dataloader_kwargs,
        ):
        assert schedule in ['random', 'round_robin', 'shuffled_round_robin'], f"Unsupported schedule: {schedule}. Supported schedules are 'random', 'round_robin', and 'shuffled_round_robin'."
        assert num_workers == 0 or num_workers >= len(stream_opts), f"num_workers ({num_workers}) must be 0 or at least the number of stream_opts ({len(stream_opts)})."
        assert len(datasets) == len(stream_opts), f"Number of datasets ({len(datasets)}) must match the number of stream configs ({len(stream_opts)})."

        self.ratios = np.array([s.ratio for s in stream_opts])
        unused_stream_idx = np.where(self.ratios <= 0)[0]
        datasets = [d for i, d in enumerate(datasets) if i not in unused_stream_idx]
        stream_opts = [s for i, s in enumerate(stream_opts) if i not in unused_stream_idx]
        self.counts = misc.probabilities_to_counts(self.ratios)

        worker_counts = MultiStreamDataLoader.split_across_consumers(self.ratios, num_workers) if num_workers > 0 else [0] * len(stream_opts)
        # We now have the notion of a "mini-epoch", for which we iterate over all the streams.
        self.epoch_size = sum(self.counts)
        self.shuffle_seed = shuffle_seed
        self.schedule: ScheduleType = ScheduleType.from_str(schedule)
        assert self.schedule == 'random' or self.epoch_size < 5_000_000, f"TODO: we have a poor implementation of shuffled_round_robin which materializes the indices."

        # Initializing the streams.
        self.streams = []
        for stream_idx, opts in enumerate(stream_opts):
            assert isinstance(opts, StreamOptions), f"Expected stream_config to be of type StreamOptions, but got {type(opts)}."
            assert 'batch_size' not in opts.dataloader_kwargs or opts.dataloader_kwargs['batch_size'] == opts.batch_gpu, \
                f"Batch size in dataloader_kwargs ({opts.dataloader_kwargs.get('batch_size')}) must match the stream's per-gpu batch size ({opts.batch_gpu})."
            kwargs = {**common_dataloader_kwargs, **opts.dataloader_kwargs, **{'batch_size': opts.batch_gpu, 'num_workers': worker_counts[stream_idx]}}
            dataloader = torch.utils.data.DataLoader(dataset=datasets[stream_idx], **kwargs)
            iterator = inf_loop_dataloader(dataloader)
            self.streams.append(Stream(dataset=datasets[stream_idx], iterator=iterator, dataloader=dataloader, opts=opts))

        self._main_stream_idx = next(i for i, s in enumerate(self.streams) if s.opts.is_main) if any(s.opts.is_main for s in self.streams) else np.argmax(self.ratios)

    @staticmethod
    def split_across_consumers(ratios: list[float], total: int) -> list[int]:
        # Computing worker counts for each stream in a way that they sum to `total`.
        portions: list[int] = [math.ceil(total * p) for p in ratios]
        assert sum(portions) >= total, f"Worker counts {portions} exceed the total number of workers {total}."
        while sum(portions) > total:
            # Taking the stream with the maximum ratio and reducing it.
            largest_stream_idx = np.argmax(portions)
            assert portions[largest_stream_idx] > 0, f"Impossible worker counts: {portions}"
            portions[largest_stream_idx] -= 1
        return portions

    @property
    def main_stream(self) -> Iterator:
        """Gets the stream with the highest weight."""
        return self.streams[self._main_stream_idx]

    def state_dict(self) -> dict[str, Any]:
        return {'stream_states': [s.dataset.state_dict() for s in self.streams]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        assert 'stream_states' in state_dict, "State dict must contain 'stream_states'."
        assert len(state_dict['stream_states']) == len(self.streams), f"Expected {len(self.streams)} stream states, but got {len(state_dict['stream_states'])}."
        for s, state in zip(self.streams, state_dict['stream_states']):
            s.dataset.load_state_dict(state)

    def _yield_batches_from_stream(self, stream_idx: int) -> Iterator[Batch]:
        # TODO: we assume that the batch is a dict and has already been collated.
        stream = self.streams[stream_idx]
        num_accum_rounds_left = stream.opts.num_accum_rounds

        while num_accum_rounds_left > 0:
            num_accum_rounds_left -= 1

            yield Batch(
                raw_batch=next(stream.iterator),
                stream_name=stream.opts.name or f'stream_{stream_idx:03d}',
                num_accum_rounds_left=num_accum_rounds_left,
                data_type=stream.dataset.data_type,
            )

    def __iter__(self) -> Any:
        num_batches_yielded = 0

        while True:
            if self.schedule in [ScheduleType.ROUND_ROBIN, ScheduleType.SHUFFLED_ROUND_ROBIN]:
                cur_plan: list[int] = sum([[i] * c for i, c in enumerate(self.counts)], start=[]) # Counts to counted idx: [1,2,3] => [0,1,1,2,2,2]
                if self.schedule == ScheduleType.SHUFFLED_ROUND_ROBIN:
                    np.random.RandomState(num_batches_yielded + self.shuffle_seed).shuffle(cur_plan)
                for stream_idx in cur_plan:
                    yield from self._yield_batches_from_stream(stream_idx)
                    num_batches_yielded += 1
            elif self.schedule == ScheduleType.RANDOM:
                stream_idx = np.random.RandomState(num_batches_yielded + self.shuffle_seed).choice(len(self.streams), p=self.ratios)
                yield self._yield_batches_from_stream(stream_idx)
                num_batches_yielded += 1
            else:
                raise ValueError(f"Unsupported schedule: {self.schedule}. Supported schedules are 'random', 'round_robin', and 'shuffled_round_robin'.")

#----------------------------------------------------------------------------
# Helper dataloading functions.

def inf_loop_dataloader(dataloader: torch.utils.data.DataLoader) -> Iterator[dict[str, Any]]:
    while True: yield from dataloader

#----------------------------------------------------------------------------
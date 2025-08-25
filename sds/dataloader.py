from enum import Enum
import math
from typing import Any, Iterator
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch

from sds.structs import DataSampleType
from sds.utils import misc
import sds.utils.distributed as dist_utils

#----------------------------------------------------------------------------
# Helper enums and structs.

class ScheduleType(Enum):
    RANDOM = 'random'
    CONSECUTIVE = 'consecutive'
    RANDOM_ORDER = 'random_order'
    FIXED_RANDOM_ORDER = 'fixed_random_order'

    @classmethod
    def from_str(cls, schedule: str) -> 'ScheduleType':
        if schedule == 'random':
            return cls.RANDOM
        elif schedule == 'random_order':
            return cls.RANDOM_ORDER
        elif schedule == 'fixed_random_order':
            return cls.FIXED_RANDOM_ORDER
        elif schedule == 'consecutive':
            return cls.CONSECUTIVE
        else:
            raise ValueError(f"Unsupported schedule: {schedule}. Supported schedules are {list(cls)}.")

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
    mixing_group_id: int
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
    def init_group(raw_stream_configs: list[dict[str, dict | str]], drop_unused: bool=False, mixing_strategy: str='no_mixing') -> list["StreamOptions"]:
        """
        Initializes a group of streams using their configs in a way that the ratios are correctly normalized.
        TODO: it contains some fields which are not present in the StreamOptions, but are used in the dataset...
        """
        assert mixing_strategy in ['no_mixing', 'mix_all', 'custom'], f"Unsupported mixing strategy: {mixing_strategy}. Supported strategies are 'no_mixing', 'mix_all', and 'custom'."
        assert mixing_strategy != 'custom' or all(s.get('mixing_group_id') is not None for s in raw_stream_configs), \
            f"If mixing_strategy is 'custom', all streams must have a 'mixing_group_id' specified. Found: {raw_stream_configs}"

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

            mixing_group_id = raw_config['mixing_group_id'] if mixing_strategy == 'custom' else (i if mixing_strategy == 'no_mixing' else 0)

            stream_configs.append(StreamOptions(
                ratio=ratios[i],
                is_main=raw_config.get('is_main', False),
                dataloader_kwargs=raw_config.get('dataloader_kwargs', {}),
                name=name,
                batch_size=batch_size,
                batch_gpu=batch_gpu,
                num_accum_rounds=num_accum_rounds,
                mixing_group_id=mixing_group_id,
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
            schedule: str = 'fixed_random_order',
            counts_precision: int = 6,
            **common_dataloader_kwargs,
        ):
        assert num_workers == 0 or num_workers >= len(stream_opts), f"num_workers ({num_workers}) must be 0 or at least the number of stream_opts ({len(stream_opts)})."
        assert len(datasets) == len(stream_opts), f"Number of datasets ({len(datasets)}) must match the number of stream configs ({len(stream_opts)})."

        stream_ratios = np.array([s.ratio for s in stream_opts])
        unused_stream_id = np.where(stream_ratios <= 0)[0]
        datasets = [d for i, d in enumerate(datasets) if i not in unused_stream_id]
        stream_opts = [s for i, s in enumerate(stream_opts) if i not in unused_stream_id]
        worker_counts = MultiStreamDataLoader.split_across_consumers(stream_ratios, num_workers) if num_workers > 0 else [0] * len(stream_opts)

        # Unfortunately, we have a nasty "business logic" for how we mix the streams across GPUs. This makes us recompute ratios/counts for each mixing group.
        # We now have the notion of a "meta-iteration", for which we iterate over all the streams.
        self._mixing_group_counts, self._mixing_group_ratios, self._mixing_group_id_to_stream_indices = get_mixing_groups(stream_opts, counts_precision)
        self.meta_iteration_size = sum(self._mixing_group_counts)
        self.shuffle_seed = shuffle_seed
        self.schedule: ScheduleType = ScheduleType.from_str(schedule)
        assert self.schedule == ScheduleType.RANDOM or self.meta_iteration_size < 100_000, f"TODO: we have a poor implementation of random_order which materializes the indices."

        # Initializing the streams.
        self.streams = []
        for stream_id, opts in enumerate(stream_opts):
            assert isinstance(opts, StreamOptions), f"Expected stream_config to be of type StreamOptions, but got {type(opts)}."
            assert 'batch_size' not in opts.dataloader_kwargs or opts.dataloader_kwargs['batch_size'] == opts.batch_gpu, \
                f"Batch size in dataloader_kwargs ({opts.dataloader_kwargs.get('batch_size')}) must match the stream's per-gpu batch size ({opts.batch_gpu})."
            kwargs = {**common_dataloader_kwargs, **opts.dataloader_kwargs, **{'batch_size': opts.batch_gpu, 'num_workers': worker_counts[stream_id]}}
            dataloader = torch.utils.data.DataLoader(dataset=datasets[stream_id], **kwargs)
            iterator = inf_loop_dataloader(dataloader)
            self.streams.append(Stream(dataset=datasets[stream_id], iterator=iterator, dataloader=dataloader, opts=opts))

        self._main_stream_id = next(i for i, s in enumerate(self.streams) if s.opts.is_main) if any(s.opts.is_main for s in self.streams) else np.argmax(stream_ratios)
        self._num_batches_yielded = 0

    @staticmethod
    def split_across_consumers(ratios: list[float], total: int) -> list[int]:
        # Computing worker counts for each stream in a way that they sum to `total`.
        portions: list[int] = [math.ceil(total * p) for p in ratios]
        assert sum(portions) >= total, f"Worker counts {portions} exceed the total number of workers {total}."
        while sum(portions) > total:
            # Taking the stream with the maximum ratio and reducing it.
            largest_stream_id = np.argmax(portions)
            assert portions[largest_stream_id] > 0, f"Impossible worker counts: {portions}"
            portions[largest_stream_id] -= 1
        return portions

    @property
    def main_stream(self) -> Iterator:
        """Gets the stream with the highest weight."""
        return self.streams[self._main_stream_id]

    def state_dict(self) -> dict[str, Any]:
        return {'stream_states': [s.dataset.state_dict() for s in self.streams]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        assert 'stream_states' in state_dict, "State dict must contain 'stream_states'."
        assert len(state_dict['stream_states']) == len(self.streams), f"Expected {len(self.streams)} stream states, but got {len(state_dict['stream_states'])}."
        for s, state in zip(self.streams, state_dict['stream_states']):
            s.dataset.load_state_dict(state)

    def _yield_batch_from_stream(self, stream_id: int) -> Iterator[Batch]:
        # TODO: we assume that the batch is a dict and has already been collated.
        stream = self.streams[stream_id]
        num_accum_rounds_left = stream.opts.num_accum_rounds

        while num_accum_rounds_left > 0:
            num_accum_rounds_left -= 1

            yield Batch(
                raw_batch=next(stream.iterator),
                stream_name=stream.opts.name or f'stream_{stream_id:03d}',
                num_accum_rounds_left=num_accum_rounds_left,
                data_type=stream.dataset.data_type,
            )

    def __iter__(self) -> Any:
        while True:
            if self.schedule in (ScheduleType.CONSECUTIVE, ScheduleType.RANDOM_ORDER, ScheduleType.FIXED_RANDOM_ORDER):
                cur_meta_iteration = self._num_batches_yielded // self.meta_iteration_size
                cur_sub_iteration = self._num_batches_yielded % self.meta_iteration_size
                # TODO: we shouldn't recreate the plan on each iteration, we can do this on each meta-iteration, but that makes that code a bit uglier...
                cur_meta_iteration_order: list[int] = sum([[i] * c for i, c in enumerate(self._mixing_group_counts)], start=[]) # Counts to counted idx: [1,2,3] => [0,1,1,2,2,2]
                if self.schedule in (ScheduleType.RANDOM_ORDER, ScheduleType.FIXED_RANDOM_ORDER):
                    shuffle_seed = (cur_meta_iteration + 1007 * self.shuffle_seed) if self.schedule == ScheduleType.RANDOM_ORDER else self.shuffle_seed
                    np.random.RandomState(shuffle_seed).shuffle(cur_meta_iteration_order)
                mixing_group_idx = cur_meta_iteration_order[cur_sub_iteration]
            elif self.schedule == ScheduleType.RANDOM:
                mixing_group_idx = np.random.RandomState(self._num_batches_yielded + 1007 * self.shuffle_seed).choice(len(self._mixing_group_counts), p=self._mixing_group_ratios)
            else:
                raise ValueError(f"Unsupported schedule: {self.schedule}. Supported schedules are 'random', 'consecutive', and 'random_order'.")

            # Now, we can choose a random stream from a given mixing group.
            stream_indices = self._mixing_group_id_to_stream_indices[mixing_group_idx]
            stream_ratios = np.array([self.streams[i].opts.ratio for i in stream_indices])
            stream_ratios = stream_ratios / stream_ratios.sum()
            stream_id = np.random.RandomState(self._num_batches_yielded + 1007 * self.shuffle_seed + 1_000_003 * dist_utils.get_rank()).choice(stream_indices, p=stream_ratios)

            yield from self._yield_batch_from_stream(stream_id)
            self._num_batches_yielded += 1

#----------------------------------------------------------------------------
# Helper dataloading functions.

def inf_loop_dataloader(dataloader: torch.utils.data.DataLoader) -> Iterator[dict[str, Any]]:
    while True: yield from dataloader

#----------------------------------------------------------------------------
# Helper misc functions.

def get_mixing_groups(stream_opts: list[StreamOptions], counts_precision: float) -> list[int]:
    stream_id_to_group_id: dict[int, int] = {i: s.mixing_group_id for i, s in enumerate(stream_opts)}
    group_id_to_stream_indicies: dict[int, list[int]] = defaultdict(list)
    for stream_id, group_id in stream_id_to_group_id.items():
        group_id_to_stream_indicies[group_id].append(stream_id)
    group_ratios = [sum(stream_opts[i].ratio for i in group) for group in group_id_to_stream_indicies.values()]
    assert all(r > 0 for r in group_ratios), f"All groups must have a positive ratio. Found: {group_ratios} for stream_opts: {stream_opts}"
    group_counts = misc.probabilities_to_counts(group_ratios, precision=counts_precision)
    return group_counts, group_ratios, group_id_to_stream_indicies

#----------------------------------------------------------------------------

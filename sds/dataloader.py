from enum import Enum
import math
from typing import Any, Iterator
from dataclasses import dataclass, field

import numpy as np
import torch

from sds.structs import DataSampleType
from sds.utils import misc
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

@dataclass
class StreamConfig:
    dataset: torch.utils.data.Dataset
    ratio: float | None = None
    is_main: bool = False # We want to know what's the "main" stream to compute visualizations/statistics.
    dataloader_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str | None = None
    batch_size: int | None = None # Total batch size across all the ranks.
    batch_gpu: int | None = None # Maximum batch size per GPU.
    num_accum_rounds: int = 1 # Number of gradient accumulation rounds.

    def __post_init__(self):
        """
        Fixes batch config options (with some values missing) into the fully-filled one.
        We provide the option to specify the batch size either through batch size or batch_gpu + num_accum_rounds.
        """
        world_size = dist_utils.get_world_size()
        if self.batch_size is None:
            assert not self.batch_gpu is None, f"If batch_size={self.batch_size} is None, batch_gpu={self.batch_gpu} must be specified."
            self.num_accum_rounds = 1 if self.num_accum_rounds is None else int(self.num_accum_rounds)
            self.batch_size = int(self.batch_gpu * world_size * self.num_accum_rounds)
        else:
            # Note: batch_size/batch_gpu/num_accum_rounds can be equal to 0 --- that means that we don't train on a given dataset.
            assert self.batch_size % world_size == 0, f"batch_size={self.batch_size} must be divisible by world_size={world_size}"
            assert self.batch_gpu is None or (self.batch_size // world_size) % self.batch_gpu == 0, f"If batch_size is specified, batch_gpu must be divisible by (batch_size={self.batch_size} // world_size={world_size})"
            self.batch_gpu = (self.batch_size // world_size) if self.batch_gpu is None else int(self.batch_gpu)
            self.num_accum_rounds = 0 if self.batch_gpu == 0 else (self.batch_size // (self.batch_gpu * world_size))

@dataclass
class Stream:
    dataset: torch.utils.data.Dataset
    iterator: Iterator
    dataloader: torch.utils.data.DataLoader
    config: StreamConfig

#----------------------------------------------------------------------------

class MultiStreamDataLoader:
    def __init__(
            self,
            stream_configs: list[dict[str, Any] | StreamConfig],
            num_workers: int = 0,
            shuffle_seed: int | None = 42,
            schedule: str = 'shuffled_round_robin',
            **common_dataloader_kwargs,
        ):
        stream_configs = [StreamConfig(**s) if isinstance(s, dict) else s for s in stream_configs]
        assert schedule in ['random', 'round_robin', 'shuffled_round_robin'], f"Unsupported schedule: {schedule}. Supported schedules are 'random', 'round_robin', and 'shuffled_round_robin'."
        assert num_workers == 0 or num_workers >= len(stream_configs), f"num_workers ({num_workers}) must be 0 or at least the number of streams ({len(stream_configs)})."

        if any(s.ratio is not None for s in stream_configs):
            # Computing stream probabilities/counts and filtering out streams with zero weight.
            self.ratios = misc.normalize_ratios([s.ratio for s in stream_configs])
            non_zero_stream_idx = np.where(self.ratios > 0)[0]
            stream_configs = [stream_configs[i] for i in non_zero_stream_idx]
            self.counts = misc.probabilities_to_counts(self.ratios)
        else:
            self.counts = [1] * len(stream_configs)  # Each stream gets one worker.

        self.stream_configs = stream_configs
        # We know have the notion of a "mini-epoch", for which we iterate over all the streams.
        self.epoch_size = sum(self.counts)
        self.shuffle_seed = shuffle_seed
        self.schedule: ScheduleType = ScheduleType.from_str(schedule)
        assert self.schedule == 'random' or self.epoch_size < 5_000_000, f"TODO: we have a poor implementation of shuffled_round_robin which materializes the indices."

        # Computing worker counts for each stream in a way that they sum to `num_workers`.
        worker_counts: list[int] = [math.ceil(num_workers * p) for p in self.ratios]
        assert sum(worker_counts) >= num_workers, f"Worker counts {worker_counts} exceed the total number of workers {num_workers}."
        while sum(worker_counts) > num_workers:
            # Taking the stream with the maximum proportion and
            largest_stream_idx = np.argmax(worker_counts)
            assert worker_counts[largest_stream_idx] > 0, f"Impossible worker counts: {worker_counts}"
            worker_counts[largest_stream_idx] -= 1

        self.streams = []
        for i, cfg in enumerate(stream_configs):
            kwargs = {**common_dataloader_kwargs, **cfg.dataloader_kwargs}
            dataloader = torch.utils.data.DataLoader(dataset=cfg.dataset, num_workers=worker_counts[i], **kwargs)
            self.streams.append(Stream(dataset=cfg.dataset, iterator=inf_loop_dataloader(dataloader), dataloader=dataloader, config=cfg))
        self._main_stream_idx = next(i for i, cfg in enumerate(stream_configs) if cfg.is_main) if any(cfg.is_main for cfg in stream_configs) else np.argmax(self.ratios)

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
        num_accum_rounds_left = stream.config.num_accum_rounds

        while num_accum_rounds_left > 0:
            num_accum_rounds_left -= 1

            yield Batch(
                raw_batch=next(stream.iterator),
                stream_name=stream.config.name or f'stream_{stream_idx:03d}',
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
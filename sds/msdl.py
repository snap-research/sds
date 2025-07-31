from enum import Enum
import math
from typing import Any, Iterator
from dataclasses import dataclass, field

import numpy as np
import torch

from sds.utils import misc

#----------------------------------------------------------------------------

STREAM_NAME_BATCH_KEY = '__stream_name__'

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

@dataclass(frozen=True)
class StreamConfig:
    dataset: torch.utils.data.Dataset
    weight: float | None = None
    is_main: bool = False
    kwargs: dict[str, Any] = field(default_factory=dict)
    name: str | None = None

#----------------------------------------------------------------------------

class MultiStreamDataLoader:
    def __init__(
            self,
            stream_configs: list[StreamConfig | dict[str, Any]],
            num_workers: int = 0,
            shuffle_seed: int | None = 42,
            schedule: str = 'shuffled_round_robin',
            **common_dataloader_kwargs,
        ):
        stream_configs = [StreamConfig(**s) if isinstance(s, dict) else s for s in stream_configs]
        assert schedule in ['random', 'round_robin', 'shuffled_round_robin'], f"Unsupported schedule: {schedule}. Supported schedules are 'random', 'round_robin', and 'shuffled_round_robin'."
        assert num_workers == 0 or num_workers >= len(stream_configs), f"num_workers ({num_workers}) must be 0 or at least the number of streams ({len(stream_configs)})."
        assert sum(s.weight is None for s in stream_configs) in [0, len(stream_configs)], "All streams must either have a weight or none of them should have a weight."

        # Computing the stream probabilities and filtering out streams with zero weight.
        weights = np.array([s.weight if s.weight is not None else 1 for s in stream_configs])
        assert weights.sum() > 0, "Weights must sum to a positive value."
        non_zero_stream_idx = np.where(weights > 0)[0]
        self.stream_configs = [stream_configs[i] for i in non_zero_stream_idx]
        weights = weights[non_zero_stream_idx]
        self.probabilities = weights / weights.sum()
        self.counts = misc.probabilities_to_counts(self.probabilities)

        # We know have the notion of a "mini-epoch", for which we iterate over all the streams.
        self.epoch_size = sum(self.counts)
        self.shuffle_seed = shuffle_seed
        self.schedule: ScheduleType = ScheduleType.from_str(schedule)
        assert self.schedule == 'random' or self.epoch_size < 1_000_000, f"TODO: we have a poor implementation of shuffled_round_robin."

        # Computing worker counts for each stream in a way that they sum to `num_workers`.
        worker_counts: list[int] = [math.ceil(num_workers * p) for p in self.probabilities]
        assert sum(worker_counts) >= num_workers, f"Worker counts {worker_counts} exceed the total number of workers {num_workers}."
        while sum(worker_counts) > num_workers:
            # Taking the stream with the maximum proportion and
            largest_stream_idx = np.argmax(worker_counts)
            assert worker_counts[largest_stream_idx] > 0, f"Impossible worker counts: {worker_counts}"
            worker_counts[largest_stream_idx] -= 1

        self.dataloaders = [torch.utils.data.DataLoader(dataset=cfg.dataset, **cfg.kwargs, **common_dataloader_kwargs, num_workers=nw) for cfg, nw in zip(stream_configs, worker_counts)]
        self.streams = [inf_loop_dataloader(dl) for dl in self.dataloaders]
        self._main_stream_idx = next(i for i, cfg in enumerate(stream_configs) if cfg.is_main) if any(cfg.is_main for cfg in stream_configs) else np.argmax(self.probabilities)

    @property
    def main_stream(self) -> Iterator:
        """Gets the stream with the highest weight."""
        return self.streams[self._main_stream_idx]

    def state_dict(self) -> dict[str, Any]:
        stream_states = [dl.dataset.state_dict() if hasattr(dl, 'dataset') and hasattr(dl.dataset, 'state_dict') else None for dl in self.dataloaders]
        return {'stream_states': stream_states}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        assert 'stream_states' in state_dict, "State dict must contain 'stream_states'."
        for dl, state in zip(self.dataloaders, state_dict['stream_states']):
            if hasattr(dl, 'dataset') and hasattr(dl.dataset, 'load_state_dict') and state is not None:
                dl.dataset.load_state_dict(state)
            elif state is not None:
                raise ValueError("Stream dataset does not support loading state dict.")

    def _post_process_batch(self, batch: dict[str, Any], stream_idx: int) -> dict[str, Any]:
        f"""Adds a special {STREAM_NAME_BATCH_KEY} key to each element in the batch."""
        # TODO: we assume that the batch is a dict and has already been collated.
        batch_size = len(next(iter(batch.values())))
        stream_name = self.stream_configs[stream_idx].name or f'stream_{stream_idx:03d}'
        batch[STREAM_NAME_BATCH_KEY] = [stream_name] * batch_size
        return batch

    def __iter__(self) -> Any:
        num_batches_yielded = 0

        while True:
            if self.schedule in [ScheduleType.ROUND_ROBIN, ScheduleType.SHUFFLED_ROUND_ROBIN]:
                cur_plan: list[int] = sum([[i] * c for i, c in enumerate(self.counts)], start=[]) # Counts to counted idx: [1,2,3] => [0,1,1,2,2,2]
                if self.schedule == ScheduleType.SHUFFLED_ROUND_ROBIN:
                    np.random.RandomState(num_batches_yielded + self.shuffle_seed).shuffle(cur_plan)
                for stream_idx in cur_plan:
                    yield self._post_process_batch(next(self.streams[stream_idx]), stream_idx)
                    num_batches_yielded += 1
            elif self.schedule == ScheduleType.RANDOM:
                stream_idx = np.random.RandomState(num_batches_yielded + self.shuffle_seed).choice(len(self.streams), p=self.probabilities)
                yield self._post_process_batch(next(self.streams[stream_idx]), stream_idx)
                num_batches_yielded += 1
            else:
                raise ValueError(f"Unsupported schedule: {self.schedule}. Supported schedules are 'random', 'round_robin', and 'shuffled_round_robin'.")

#----------------------------------------------------------------------------
# Helper dataloading functions.

def inf_loop_dataloader(dataloader: torch.utils.data.DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        for batch in dataloader:
            yield batch

#----------------------------------------------------------------------------
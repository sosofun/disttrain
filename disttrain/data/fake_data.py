from __future__ import annotations

from collections import deque
import time
from typing import Deque, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from disttrain.config import TrainingConfig
from disttrain.models.base import TensorDict


class FakeMultiModalDataset(Dataset):
    """
    Deterministic fake dataset for smoke/e2e.
    """

    def __init__(
        self,
        training: TrainingConfig,
        input_modalities: list[str],
        length: int = 100_000,
    ):
        self.training = training
        self.input_modalities = input_modalities
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> TensorDict:
        g = torch.Generator()
        g.manual_seed(self.training.data_seed + idx)
        out: TensorDict = {
            "text_tokens": torch.randint(
                low=0,
                high=self.training.vocab_size,
                size=(self.training.seq_len,),
                generator=g,
                dtype=torch.long,
            )
        }
        if "image" in self.input_modalities:
            out["image"] = torch.randn(
                3,
                self.training.image_size,
                self.training.image_size,
                generator=g,
            )
        if "video" in self.input_modalities:
            out["video"] = torch.randn(
                self.training.video_frames,
                3,
                self.training.image_size,
                self.training.image_size,
                generator=g,
            )
        if "audio" in self.input_modalities:
            out["audio"] = torch.randn(1, self.training.audio_length, generator=g)
        return out


class FakeBatchProvider:
    """
    Placeholder I/O pipeline with DataLoader + optional prefetch queue.
    """

    def __init__(
        self,
        training: TrainingConfig,
        input_modalities: list[str],
        device: torch.device,
        dp_size: int,
        dp_rank: int,
    ):
        self.training = training
        self.input_modalities = input_modalities
        self.device = device
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.epoch = 0

        self.dataset = FakeMultiModalDataset(training=training, input_modalities=input_modalities)

        sampler: Optional[DistributedSampler] = None
        if dp_size > 1:
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=dp_size,
                rank=dp_rank,
                shuffle=True,
                seed=training.data_seed,
                drop_last=True,
            )
        self.sampler = sampler
        self.loader = DataLoader(
            self.dataset,
            batch_size=training.micro_batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=True,
            pin_memory=training.io.pin_memory,
            num_workers=training.io.num_workers,
        )
        self.iterator = iter(self.loader)
        self.prefetch_enabled = training.io.enable_prefetch and training.io.prefetch_size > 0
        self.prefetch_size = max(training.io.prefetch_size, 0)
        self.queue: Deque[TensorDict] = deque()
        if self.prefetch_enabled:
            self._prime_prefetch()

    def _reset_epoch(self) -> None:
        self.epoch += 1
        if self.sampler is not None:
            self.sampler.set_epoch(self.epoch)
        self.iterator = iter(self.loader)

    def _next_loader_batch(self) -> TensorDict:
        try:
            return next(self.iterator)
        except StopIteration:
            self._reset_epoch()
            return next(self.iterator)

    def _to_device(self, batch: TensorDict) -> Tuple[TensorDict, float]:
        t0 = time.perf_counter()
        if self.device.type == "cpu":
            return batch, 0.0
        moved: TensorDict = {}
        for k, v in batch.items():
            moved[k] = v.to(self.device, non_blocking=self.training.io.pin_memory)
        return moved, time.perf_counter() - t0

    def _prime_prefetch(self) -> None:
        while len(self.queue) < self.prefetch_size:
            self.queue.append(self._next_loader_batch())

    def next_batch(self) -> Tuple[TensorDict, Dict[str, float]]:
        wait_t0 = time.perf_counter()
        if self.prefetch_enabled:
            if not self.queue:
                self._prime_prefetch()
            batch = self.queue.popleft()
            self.queue.append(self._next_loader_batch())
        else:
            batch = self._next_loader_batch()
        wait_sec = time.perf_counter() - wait_t0
        moved, h2d_sec = self._to_device(batch)
        return moved, {"dataloader_wait_sec": wait_sec, "host_to_device_sec": h2d_sec}

"""Contiguous batches of model inputs, grouped by dtype for device transfers."""

from collections import defaultdict
from math import prod

import numpy as np
import torch


BatchSchema = tuple[tuple[str, tuple[int, ...], str], ...]


class PackedInputs:
    def __init__(
        self,
        schema: BatchSchema,
        capacity: int,
        device: torch.device,
        *,
        pin_memory: bool = False,
        force_bool: bool = False,
    ):
        self.schema = schema
        self.capacity = capacity
        self.force_bool = force_bool
        fields_by_dtype = defaultdict(list)
        for name, shape, dtype in schema:
            target_dtype = (
                torch.bool
                if force_bool
                else torch.from_numpy(np.empty((), dtype=dtype)).dtype
            )
            fields_by_dtype[target_dtype].append((name, (capacity, *shape)))
        self.group_tensors = {}
        self.tensors = {}
        for dtype, fields in fields_by_dtype.items():
            sizes = [prod(shape) for _, shape in fields]
            group = torch.empty(
                sum(sizes), dtype=dtype, device=device, pin_memory=pin_memory
            )
            self.group_tensors[dtype] = group
            for (name, shape), view in zip(fields, group.split(sizes), strict=True):
                self.tensors[name] = view.view(shape)
        self.arrays = (
            {k: t.numpy() for k, t in self.tensors.items()}
            if device.type == "cpu"
            else {}
        )


class InputTransfer:
    """Reusable host and device views for one rollout batch."""

    def __init__(self, arrays, device, *, force_bool=False):
        schema = tuple((k, a.shape[1:], a.dtype.str) for k, a in arrays.items())
        count = next(iter(arrays.values())).shape[0]
        self.host = PackedInputs(
            schema,
            count,
            torch.device("cpu"),
            pin_memory=device.type == "cuda",
            force_bool=force_bool,
        )
        self.target = (
            self.host
            if device.type == "cpu"
            else PackedInputs(schema, count, device, force_bool=force_bool)
        )

    def stage(self, arrays):
        for key, destination in self.host.arrays.items():
            destination[...] = arrays[key]
        return self.upload()

    def upload(self):
        """Transfer the populated host views together by dtype."""
        if self.target is not self.host:
            for dtype, source in self.host.group_tensors.items():
                self.target.group_tensors[dtype].copy_(source, non_blocking=True)
        return self.target.tensors

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


TensorDict = Dict[str, torch.Tensor]


class StageModel(nn.Module):
    """
    Common stage interface:
      forward(inputs, meta) -> outputs
    """

    stage_name: str = "unknown"

    def forward(
        self, inputs: TensorDict, meta: Optional[Dict[str, object]] = None
    ) -> TensorDict:
        raise NotImplementedError

    def set_tp_group(self, tp_group: Optional[object]) -> None:
        """
        Optional hook for TP-capable layers.
        """
        del tp_group

# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from torchao.quantization.pt2e import MappingType, PerBlock
from torchao.quantization.pt2e._affine_quantization import (
    _get_reduction_params,
    AffineQuantizedMinMaxObserver,
    choose_qparams_affine_with_min_max,
)


class PerBlockParamObserver(AffineQuantizedMinMaxObserver):
    def __init__(
        self,
        dtype: torch.dtype,
        block_size: torch.Size,
        quant_min=None,
        quant_max=None,
        eps=torch.finfo(torch.float32).eps,  # noqa: B008
        **kwargs,
    ):
        super().__init__(
            mapping_type=MappingType.SYMMETRIC,
            target_dtype=dtype,
            granularity=PerBlock,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            **kwargs,
        )
        self.block_size = block_size
        self.calibrated = False

    def forward(self, input: torch.Tensor):
        if input.numel() == 0 or self.calibrated:
            return input

        input_detached = input.detach()
        self.original_dtype = input_detached.dtype
        shape_for_reduction, reduction_dims = _get_reduction_params(
            self.block_size, input_detached.size()
        )
        input_detached = input_detached.view(shape_for_reduction)
        min_val = torch.amin(input_detached, dim=reduction_dims)
        max_val = torch.amax(input_detached, dim=reduction_dims)
        if not hasattr(self, "min_val") or not hasattr(self, "max_val"):
            self.min_val = min_val
            self.max_val = max_val
        else:
            assert (
                self.min_val.shape == min_val.shape
            ), f"Can't update existing min_val - shape mismatch, self.min_val:{self.min_val.shape} != min_val:{min_val.shape}"
            assert (
                self.max_val.shape == max_val.shape
            ), f"Can't update existing max_val - shape mismatch, self.max_val {self.max_val.shape} != max_val:{max_val.shape}"
            min_val = torch.min(self.min_val, min_val)
            max_val = torch.max(self.max_val, max_val)
            self.min_val.copy_(min_val)
            self.max_val.copy_(max_val)

        self.calibrated = True
        return input

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert hasattr(self, "min_val") and hasattr(
            self, "max_val"
        ), "Expecting the observer has min_val and max_val, please run the observer before calling calculate_qparams"
        return choose_qparams_affine_with_min_max(
            self.min_val,
            self.max_val,
            self.mapping_type,
            [],
            self.target_dtype,
            self.quant_min,
            self.quant_max,
            self.eps,
            self.scale_dtype,
            self.zero_point_dtype,
            self.preserve_zero,
            self.zero_point_domain,
        )

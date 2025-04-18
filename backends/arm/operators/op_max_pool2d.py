# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import torch

import tosa_tools.v0_80.serializer.tosa_serializer as ts  # type: ignore

from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg


@register_node_visitor
class MaxPool2dVisitor(NodeVisitor):
    target = "aten.max_pool2d.default"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        input_tensor = inputs[0]
        kernel_size = inputs[1].special
        stride = inputs[2].special

        try:
            pad_size_list = inputs[3].special
            pad_size_list = [
                pad_size_list[0],
                pad_size_list[0],
                pad_size_list[1],
                pad_size_list[1],
            ]
        except IndexError:
            pad_size_list = [0, 0, 0, 0]

        accumulator_type = output.dtype

        # Initilize zero point to zero.
        input_zp = 0
        if inputs[0].dtype == ts.DType.INT8:
            input_qparams = get_input_qparams(node)
            input_zp = input_qparams[0].zp

        output_zp = 0
        if output.dtype == ts.DType.INT8:
            output_qparams = get_output_qparams(node)
            output_zp = output_qparams[0].zp

        attr = ts.TosaSerializerAttribute()
        attr.PoolAttribute(
            kernel=kernel_size,
            stride=stride,
            pad=pad_size_list,
            input_zp=input_zp,
            output_zp=output_zp,
            accum_dtype=accumulator_type,
        )

        tosa_graph.addOperator(
            ts.TosaOp.Op().MAX_POOL2D,
            [input_tensor.name],
            [output.name],
            attr,
        )

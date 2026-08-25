# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure FCB multi-SoC reference-weight sharing during QNN AOT."""

import argparse
import json
from pathlib import Path

import torch

from executorch.backends.qualcomm.qnn_preprocess import QnnBackend
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)


class TwoConvs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(1, 3, 10)
        self.second = torch.nn.Conv2d(3, 2, 10)

    def forward(self, x):
        return self.second(self.first(x))


def export_fcb(soc_models, reference_weight_sharing_enabled):
    model = TwoConvs().eval()
    modules = {"two_convs": model, "second": model.second}
    inputs = {
        "two_convs": (torch.randn(1, 1, 80, 80),),
        "second": (torch.randn(1, 3, 60, 60),),
    }
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=soc_models[0],
        soc_models=soc_models,
        backend_options=generate_htp_compiler_spec(use_fp16=False, use_weight_sharing=True),
        reference_weight_sharing_enabled=reference_weight_sharing_enabled,
    )
    program = to_edge_transform_and_lower_to_qnn(
        module=modules,
        inputs=inputs,
        compiler_specs={name: compiler_specs for name in modules},
    ).to_executorch()
    return program.buffer, QnnBackend.last_fcb_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soc_models", nargs="+", required=True, choices=QcomChipset.__members__)
    parser.add_argument("--output_dir", type=Path, default=Path("/tmp/qnn_fcb_weight_sharing"))
    args = parser.parse_args()
    if len(args.soc_models) < 2:
        parser.error("FCB requires at least two SoCs")

    soc_models = [QcomChipset[model] for model in args.soc_models]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    unshared, unshared_stats = export_fcb(soc_models, False)
    torch.manual_seed(0)
    shared, shared_stats = export_fcb(soc_models, True)
    unshared_path = args.output_dir / "unshared_fcb.pte"
    shared_path = args.output_dir / "shared_fcb.pte"
    unshared_path.write_bytes(unshared)
    shared_path.write_bytes(shared)

    summary = {
        "requested_socs": args.soc_models,
        "unshared_pte_bytes": len(unshared),
        "shared_pte_bytes": len(shared),
        "cache_record_count": shared_stats["cache_record_count"],
        "intermediate_context_binary_bytes_in_python": shared_stats[
            "intermediate_context_binary_bytes_in_python"
        ],
        "max_live_contexts": shared_stats["max_live_contexts"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


    if summary["shared_pte_bytes"] >= summary["unshared_pte_bytes"]:
        raise RuntimeError("FCB reference weight sharing did not reduce PTE size")
    if summary["intermediate_context_binary_bytes_in_python"] != 0:
        raise RuntimeError("FCB materialized an intermediate context binary in Python")
    if summary["max_live_contexts"] != 1:
        raise RuntimeError("FCB retained more than one live QNN context")

if __name__ == "__main__":
    main()

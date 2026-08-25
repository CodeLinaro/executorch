# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a two-method QNN context with shared weights and optionally run both methods."""

import argparse
import shutil
from pathlib import Path

import torch

from executorch.backends.qualcomm.export_utils import QnnConfig, SimpleADB
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import (
    dump_context_from_pte,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)


class TwoConvs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(1, 3, 3)
        self.second = torch.nn.Conv2d(3, 2, 3)

    def forward(self, x):
        return self.second(self.first(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soc_model", default="SM8850", choices=QcomChipset.__members__)
    parser.add_argument("--output_dir", type=Path, default=Path("/tmp/qnn_weight_sharing"))
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--host")
    parser.add_argument("--build_folder", default="build-android")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = TwoConvs().eval()
    modules = {"two_convs": model, "second": model.second}
    inputs = {
        "two_convs": (torch.randn(1, 1, 8, 8),),
        "second": (torch.randn(1, 3, 6, 6),),
    }
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=QcomChipset[args.soc_model],
        backend_options=generate_htp_compiler_spec(
            use_fp16=args.fp16,
            use_weight_sharing=True,
        ),
    )
    program = to_edge_transform_and_lower_to_qnn(
        module=modules,
        inputs=inputs,
        compiler_specs={name: compiler_specs for name in modules},
    ).to_executorch()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pte_path = args.output_dir / "shared_weights.pte"
    pte_path.write_bytes(program.buffer)
    print(f"{pte_path}: {pte_path.stat().st_size} bytes")
    for context_path in dump_context_from_pte(pte_path, args.output_dir / "contexts"):
        print(f"{context_path}")

    if not args.device:
        return

    qnn_config = QnnConfig(
        soc_model=args.soc_model,
        build_folder=args.build_folder,
        device=args.device,
        host=args.host,
    )
    adb = SimpleADB(
        qnn_config=qnn_config,
        pte_path=str(pte_path),
        workspace="/data/local/tmp/qnn_weight_sharing_demo",
    )
    for method_index, (name, module) in enumerate(sorted(modules.items())):
        method_inputs = inputs[name]
        expected = module(*method_inputs).detach()
        adb.push(inputs=[method_inputs], init_env=method_index == 0)
        adb.execute(custom_runner_cmd=f"rm -rf {adb.output_folder}")
        adb.execute(method_index=method_index)
        pulled_outputs = args.output_dir / f"{name}_outputs"
        shutil.rmtree(pulled_outputs, ignore_errors=True)
        adb.pull(str(pulled_outputs), device_output_path=adb.output_folder)
        raw_output = next(pulled_outputs.rglob("*.raw"))
        actual = torch.from_file(
            str(raw_output), dtype=expected.dtype, size=expected.numel()
        ).reshape(expected.shape)
        torch.testing.assert_close(actual, expected, rtol=1, atol=1e-1)
        shutil.move(raw_output, args.output_dir / f"{name}.raw")
        print(f"device method {method_index} ({name}): PASS")


if __name__ == "__main__":
    main()

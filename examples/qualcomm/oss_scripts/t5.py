# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from multiprocessing.connection import Client

import numpy as np

import torch
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    make_output_dir,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
)

from transformers import T5Tokenizer, T5Model
from transformers.models.t5.modeling_t5 import T5Attention
import math

# Copy from transformers/models/t5/modeling_t5.py
def _relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance):
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.int32) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    )

    relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets

T5Attention._relative_position_bucket = staticmethod(_relative_position_bucket)


def postprocess(input_tokens, outputs, tokenizer, vocab_size):
    ret = torch.clone(input_tokens).reshape(-1)
    outputs = outputs.reshape(-1, vocab_size)

    masked_index = torch.nonzero(ret == tokenizer.mask_token_id, as_tuple=False)

    logits = outputs[masked_index, :]
    probs = logits.softmax(dim=-1)

    prediction = torch.argmax(probs).to(torch.int32)
    ret[masked_index] = prediction
    return tokenizer.decode(ret)

def main(args):
    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    if not args.compile_only and args.device is None:
        raise RuntimeError(
            "device serial is required if not compile only. "
            "Please specify a device serial by -s/--device argument."
        )

    text = args.input
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    encoded_inputs = tokenizer(text, return_tensors="pt")
    inputs = (
        (
            encoded_inputs["input_ids"].to(torch.int32),
            encoded_inputs["attention_mask"].to(torch.int32),
        ),
    )

    input_text = " ".join([f"input_0_{i}.raw" for i in range(len(inputs[0]))])
    input_list = f"{input_text}\n"

    pte_filename = "t5"
    module = T5Model.from_pretrained("t5-small").eval()
    vocab_size = module.config.vocab_size

    passes_job = get_capture_program_passes()
    build_executorch_binary(
        module,
        inputs[0],
        args.model,
        f"{args.artifact}/{pte_filename}",
        inputs,
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
        quant_dtype=QuantDtype.use_16a16w,
        passes_job=passes_job,
        shared_buffer=args.shared_buffer,
    )

    if args.compile_only:
        return

    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=f"{args.build_folder}",
        pte_path=f"{args.artifact}/{pte_filename}.pte",
        workspace=f"/data/local/tmp/executorch/{pte_filename}",
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
    )
    adb.push(inputs=inputs, input_list=input_list)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    adb.pull(output_path=args.artifact)

    inferenced_output = torch.from_numpy(
        np.fromfile(
            os.path.join(output_data_folder, "output_0_0.raw"), dtype=np.float32
        )
    )

    ref = module(*inputs[0]).logits
    print(
        postprocess(inputs[0][0], ref, tokenizer, vocab_size)
        == postprocess(inputs[0][0], inferenced_output, tokenizer, vocab_size)
    )


if __name__ == "__main__":
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-a",
        "--artifact",
        help="Path for storing generated artifacts by this example. Default ./t5",
        default="./t5",
        type=str,
    )

    parser.add_argument(
        "-i",
        "--input",
        help=(
            "Given a sentance with [MASK], those [MASK] tokens will be changed to output of albert"
            "e.g. Hello I'm a [MASK] model."
        ),
        type=str,
        default="Hello I'm a [MASK] model.",
    )

    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)

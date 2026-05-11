# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Using the ExecuTorch Developer Tools for Numerical Debugging
========================
"""

######################################################################
# The `ExecuTorch Developer Tools <../devtools-overview.html>`__ is a set of tools designed to
# provide users with the ability to profile, debug, and visualize ExecuTorch
# models.
#
# This tutorial will show a full end-to-end flow of how to utilize the Developer Tools to debug a model
# by detecting numerical discrepancies between the original PyTorch model and the ExecuTorch model.
#
# The tutorial will show you how to:
# 1. Check if the lowered ExecuTorch model is numerically correct.
# 2. Gain a deeper understanding of where the numerical discrepancy comes from using the Inspector API.
#
# This is particularly useful when working with delegated models (e.g., XNNPACK) where numerical
# precision may differ. Specifically, it will:
#
# 1. Generate the artifacts consumed by the Developer Tools (`ETRecord <../etrecord.html>`__, `ETDump <../etdump.html>`__).
# 2. Run the model and compare final outputs between eager model and runtime.
# 3. If discrepancies exist, use the Inspector's `calculate_numeric_gap <../model-inspector.html#calculate-numeric-gap>`__ method to identify operator-level issues.
#
# .. note::
#    Currently operator-level debugging support is limited to ET-visible operators,
#    and treat every delegate call as a single operator.
#    We are working on expanding this support to dive into delegate operators.
#
# We provide two example debugging pipelines on xnnpack-delegated Vision Transformer (VIT) model:
#
# - **Python Pipeline**: Export, run, and debug entirely in Python using the ExecuTorch Runtime API.
# - **CMake Pipeline**: Export in Python, run with CMake example runner, then analyze in Python.

######################################################################
# Prerequisites
# -------------
#
# To run this tutorial, you'll first need to
# `Set up your ExecuTorch environment <../getting-started-setup.html>`__.
#
# For the Python pipeline, you'll need the ExecuTorch Python runtime bindings.
# For the CMake pipeline, follow `these instructions <../runtime-build-and-cross-compilation.html#configure-the-cmake-build>`__ to set up CMake.
#

######################################################################
# Pipeline 1: Python Runtime
# =========================================================
#
# This pipeline allows you to export, run, and debug your model entirely in Python,
# making it ideal for rapid iteration during development.

######################################################################
# Step 1: Export Model and Generate ETRecord
# ------------------------------------------
#
# First, we export the model and generate an ``ETRecord``. The ETRecord contains
# model graphs and metadata for linking runtime results to the eager model.
# We use ``to_edge_transform_and_lower`` with ``generate_etrecord=True`` to
# automatically capture the ETRecord during the lowering process.
#
# .. code-block:: python
#
#    import os
#    import tempfile
#
#    import torch
#    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
#    from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
#    from executorch.exir import ExecutorchProgramManager, to_edge_transform_and_lower
#    from torch.export import export, ExportedProgram
#    from torchvision import models  # type: ignore[import-untyped]
#
#    # Create Vision Transformer model
#    vit = models.vision_transformer.vit_b_16(weights="IMAGENET1K_V1")
#    model = vit.eval()
#    model_inputs = (torch.randn(1, 3, 224, 224),)
#
#    temp_dir = tempfile.mkdtemp()
#
#    # Export and lower model to XNNPACK delegate
#    aten_model: ExportedProgram = export(model, model_inputs, strict=True)
#    edge_program_manager = to_edge_transform_and_lower(
#        aten_model,
#        partitioner=[XnnpackPartitioner()],
#        compile_config=get_xnnpack_edge_compile_config(),
#        generate_etrecord=True,
#    )
#
#    et_program_manager: ExecutorchProgramManager = edge_program_manager.to_executorch()
#
#    # Save the .pte file
#    pte_path = os.path.join(temp_dir, "model.pte")
#    et_program_manager.save(pte_path)
#
#    # Get and save ETRecord with representative inputs
#    etrecord = et_program_manager.get_etrecord()
#    etrecord.update_representative_inputs(model_inputs)
#    etrecord_path = os.path.join(temp_dir, "etrecord.bin")
#    etrecord.save(etrecord_path)
#

######################################################################
#
# .. note::
#    The ``update_representative_inputs`` method is crucial for debugging.
#    It stores the inputs that will be used to compute reference outputs
#    from the exported program, which are then compared against the runtime outputs.
#

######################################################################
# Step 2: Run Model and Generate ETDump with Debug Buffer
# -------------------------------------------------------
#
# Next, we run the model using the ExecuTorch Python Runtime API with debug
# output enabled. The debug buffer captures intermediate outputs from the
# runtime execution.
#
# .. code-block:: python
#
#    from executorch.runtime import Method, Program, Runtime, Verification
#
#    # Load and run the model with debug output enabled
#    et_runtime: Runtime = Runtime.get()
#    program: Program = et_runtime.load_program(
#        pte_path,
#        verification=Verification.Minimal,
#        enable_etdump=True,
#        debug_buffer_size=1024 * 1024 * 1024,  # 1GB buffer
#    )
#
#    forward: Method = program.load_method("forward")
#    runtime_outputs = forward.execute(*model_inputs)
#
#    # Save ETDump and debug buffer
#    etdump_path = os.path.join(temp_dir, "etdump.etdp")
#    debug_buffer_path = os.path.join(temp_dir, "debug_buffer.bin")
#    program.write_etdump_result_to_file(etdump_path, debug_buffer_path)
#
# .. warning::
#    The debug buffer size should be large enough to hold all intermediate
#    outputs.
#    If the buffer is too small, some intermediate outputs may be truncated or error might be rasied.
#

######################################################################
# Step 3: Compare Final Outputs (Best Practice)
# ---------------------------------------------
#
# **Best Practice**: Before diving into operator-level debugging, first compare
# the final outputs between the eager model and the runtime model. This helps
# you quickly determine if there are any numerical issues worth investigating.
#
# .. code-block:: python
#
#    # Get eager model output
#    with torch.no_grad():
#        eager_output = model(*model_inputs)
#
#    # Compare with runtime output
#    if isinstance(runtime_outputs, (list, tuple)):
#        runtime_output = runtime_outputs[0]
#    else:
#        runtime_output = runtime_outputs
#
#    # Calculate MSE between eager and runtime outputs
#    mse = torch.mean((eager_output - runtime_output) ** 2).item()
#    print(f"Final output MSE: {mse}")
#
#    # Check if outputs are close enough
#    if torch.allclose(eager_output, runtime_output, rtol=1e-3, atol=1e-5):
#        print("Outputs match within tolerance!")
#    else:
#        print("Outputs differ - proceeding with operator-level analysis...")
#

######################################################################
# Step 4: Operator-Level Debugging with calculate_numeric_gap
# -----------------------------------------------------------
#
# If the final outputs show discrepancies, use the Inspector's ``calculate_numeric_gap``
# method to identify which operators are contributing to the numerical differences.
#
# .. code-block:: python
#
#    import pandas as pd
#    from executorch.devtools import Inspector
#
#    inspector = Inspector(
#        etdump_path=etdump_path,
#        etrecord=etrecord_path,
#        debug_buffer_path=debug_buffer_path,
#        # reference_graph_name defaults to EDGE_DIALECT_GRAPH_KEY; override when
#        # you want to use a different graph (e.g. a post-lowering graph key) as
#        # the reference for debug handle mapping.
#    )
#
#    pd.set_option("display.width", 100000)
#    pd.set_option("display.max_columns", None)
#
#    # Calculate numerical gap using Mean Squared Error
#    df: pd.DataFrame = inspector.calculate_numeric_gap("MSE")
#    print(df)
#
# The returned DataFrame contains columns for each operator including:
#
# - ``aot_debug_handle``: The debug handle tuple identifying the AOT operator(s)
# - ``aot_ops``: The operators in the eager model graph
# - ``aot_intermediate_output``: Intermediate outputs from eager model
# - ``runtime_ops``: The kernel-level operators executed at runtime
# - ``runtime_debug_handle``: The debug handle tuple from the runtime
# - ``runtime_intermediate_output``: Intermediate outputs from runtime
# - ``gap``: The numerical gap (MSE) between eager and runtime outputs; ``nan`` when shapes differ
# - ``stacktraces``: A dictionary mapping each operator name to its source code stack trace
#
# Example output:
#
# .. code-block:: text
#
#    |    | aot_debug_handle | aot_ops                                    | aot_intermediate_output                           | runtime_ops                          | runtime_debug_handle | runtime_intermediate_output                    | gap                      | stacktraces                                    |
#    |----|-----------------|-------------------------------------------|---------------------------------------------------|--------------------------------------|---------------------|------------------------------------------------|--------------------------|------------------------------------------------|
#    | 0  | (4,)            | [conv2d]                                  | [[[tensor([-0.0130,  0.0075, -0.0334,...          | [native_call_convolution_out]        | (4,)                | [[[tensor([-0.0130,  0.0075, -0.0334,...       | [3.2530690555343034e-15] | {'conv2d': 'File "vit.py", line 10...'}        |
#    | 1  | (11,)           | [permute, cat, add, dropout]              | [[[tensor(-0.0024), tensor(0.0054),...            | [native_call_permute_copy_out]       | (11,)               | [[[tensor(-0.0024), tensor(0.0054),...         | [3.2488685838924244e-15] | {'permute': 'File "vit.py", line 15...', ...}  |
#    | ...|                 |                                           |                                                   |                                      |                     |                                                |                          |                                                |
#    | 4  | (62,)           | [linear, unflatten, unsqueeze, transp...] | [[[tensor(0.0045), tensor(-0.0084),...            | [native_call_expand_copy.out]        | (62,)               | [[[tensor([0.5541, 0.0014, 0.0015,...          | [nan]                    | {'linear': 'File "vit.py", line 125...', ...}  |
#    | ...|                 |                                           |                                                   |                                      |                     |                                                |                          |                                                |
#    | 37 | (164,)          | [layer_norm_24]                           | [[[tensor(-0.9172), tensor(0.0853),...            | [native_call_native_layer_norm.out]  | (164,)              | [[[tensor(-0.9172), tensor(0.0853),...         | [2.2175176622973748e-11] | {'layer_norm_24': 'File "vit.py"...'}          |
#
# The ``stacktraces`` column is particularly useful for tracing operators back to the
# original PyTorch source code. Each entry is a dictionary where keys are operator names
# and values are the corresponding stack traces showing where in the model code each
# operator was defined.
#

######################################################################
# Step 5: Analyze and Identify Problematic Operators
# --------------------------------------------------
#
# Once you have the numerical gaps, identify operators with significant
# discrepancies for further investigation. The ``stacktraces`` column helps
# you trace problematic operators back to the original source code.
#
# .. code-block:: python
#
#    # Find operators with the largest discrepancies
#    df_sorted = df.sort_values(by="gap", ascending=False, key=lambda x: x.apply(lambda y: y[0] if isinstance(y, list) else y))
#
#    print("Top 5 operators with largest numerical discrepancies:")
#    print(df_sorted.head(5))
#
#    # Filter for operators with gap above a threshold
#    threshold = 1e-4
#    problematic_ops = df[df["gap"].apply(lambda x: x[0] > threshold if isinstance(x, list) else x > threshold)]
#    print(f"\nOperators with MSE > {threshold}:")
#    print(problematic_ops)
#
#    # Use stacktraces to find the source location of problematic operators
#    for idx, row in problematic_ops.iterrows():
#        print(f"\n--- Operator {idx} ---")
#        print(f"Operators: {row['aot_ops']}")
#        print(f"Gap: {row['gap']}")
#        print("Stack traces:")
#        for op_name, trace in row['stacktraces'].items():
#            if trace:
#                print(f"  {op_name}:")
#                # Print first few lines of stack trace
#                for line in trace.split('\n')[:3]:
#                    print(f"    {line}")
#
# Example output showing problematic operators in a ViT model:
#
# .. code-block:: text
#
#    Top 5 operators with largest numerical discrepancies:
#       aot_debug_handle          aot_ops                            aot_intermediate_output            runtime_ops                   runtime_debug_handle  runtime_intermediate_output                       gap                                stacktraces
#    37          (164,)  [layer_norm_24]  [[[tensor(-0.9172), tensor(0.0853),...  [native_call_native_layer_norm.out]              (164,)  [[[tensor(-0.9172), tensor(0.0853),...  [2.2175176622973748e-11]  {'layer_norm_24': 'File "vit.py"...'}
#    33          (144,)  [layer_norm_21]  [[[tensor(-0.8958), tensor(-0.0307),...  [native_call_native_layer_norm.out]             (144,)  [[[tensor(-0.8958), tensor(-0.0307),...  [1.2286585568717539e-11]  {'layer_norm_21': 'File "vit.py"...'}
#    36          (157,)  [layer_norm_23]  [[[tensor(-0.8750), tensor(-0.0243),...  [native_call_native_layer_norm.out]             (157,)  [[[tensor(-0.8750), tensor(-0.0243),...  [1.2271681610366983e-11]  {'layer_norm_23': 'File "vit.py"...'}
#    30          (131,)  [layer_norm_19]  [[[tensor(-0.4218), tensor(-0.3333),...  [native_call_native_layer_norm.out]             (131,)  [[[tensor(-0.4218), tensor(-0.3333),...  [1.1904724456170941e-11]  {'layer_norm_19': 'File "vit.py"...'}
#    24          (105,)  [layer_norm_15]  [[[tensor(-0.2805), tensor(-0.3079),...  [native_call_native_layer_norm.out]             (105,)  [[[tensor(-0.2805), tensor(-0.3079),...  [1.1866889275499194e-11]  {'layer_norm_15': 'File "vit.py"...'}
#
#    --- Operator 37 ---
#    Operators: ['layer_norm_24']
#    Gap: [2.2175176622973748e-11]
#    Stack traces:
#      layer_norm_24:
#        File "torchvision/models/vision_transformer.py", line 78, in forward
#          x = self.ln(x)
#        File "torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
#
#    Operators with MSE > 0.0001:
#    Empty DataFrame
#    Columns: [aot_debug_handle, aot_ops, aot_intermediate_output, runtime_ops, runtime_debug_handle, runtime_intermediate_output, gap, stacktraces]
#    Index: []
#
# In this example, the largest numerical gaps come from layer norm operators (gaps ~1e-11),
# which reflects floating-point rounding at float32 precision — well within acceptable tolerance.
# Some attention-related operators (e.g. ``linear, unflatten, unsqueeze, transpose`` groups) show
# ``nan`` gap: this occurs when the AOT op-group output shape does not match the shape of the
# individual runtime kernel output that was captured for the same debug handle. No operators
# exceed the 1e-4 threshold, confirming that XNNPACK float32 delegation is numerically accurate.

######################################################################
# Pipeline 2: CMake Runtime
# ==========================
#
# This pipeline is useful when you want to test your model with the native
# C++ runtime or on platforms where Python bindings are not available.

######################################################################
# Step 1: Export Model and Generate ETRecord
# ------------------------------------------
#
# First, we export the model and generate an ``ETRecord``, same as step 1 of pipeline 1:
#
# .. code-block:: python
#
#    import os
#    import tempfile
#
#    import torch
#    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
#    from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
#    from executorch.exir import ExecutorchProgramManager, to_edge_transform_and_lower
#    from torch.export import export, ExportedProgram
#    from torchvision import models  # type: ignore[import-untyped]
#
#    # Create Vision Transformer model
#    vit = models.vision_transformer.vit_b_16(weights="IMAGENET1K_V1")
#    model = vit.eval()
#    model_inputs = (torch.randn(1, 3, 224, 224),)
#
#    temp_dir = tempfile.mkdtemp()
#
#    # Export and lower model to XNNPACK delegate
#    aten_model: ExportedProgram = export(model, model_inputs, strict=True)
#    edge_program_manager = to_edge_transform_and_lower(
#        aten_model,
#        partitioner=[XnnpackPartitioner()],
#        compile_config=get_xnnpack_edge_compile_config(),
#        generate_etrecord=True,
#    )
#
#    et_program_manager: ExecutorchProgramManager = edge_program_manager.to_executorch()
#
#    # Save the .pte file
#    pte_path = os.path.join(temp_dir, "model.pte")
#    et_program_manager.save(pte_path)
#
#    # Get and save ETRecord with representative inputs
#    etrecord = et_program_manager.get_etrecord()
#    etrecord.update_representative_inputs(model_inputs)
#    etrecord_path = os.path.join(temp_dir, "etrecord.bin")
#    etrecord.save(etrecord_path)
#

######################################################################
# Step 2: Create BundledProgram
# -----------------------------
#
# For the CMake pipeline, we create a ``BundledProgram`` that packages the model
# with sample inputs and expected outputs for testing. We reuse the
# ``et_program_manager`` from Step 1.
#
# .. code-block:: python
#
#    from executorch.devtools import BundledProgram
#    from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
#    from executorch.devtools.bundled_program.serialize import (
#        serialize_from_bundled_program_to_flatbuffer,
#    )
#
#    # Define the method name and test inputs
#    # IMPORTANT: Use the same inputs as etrecord.update_representative_inputs()
#    m_name = "forward"
#    test_inputs = [model_inputs]
#
#    # Create test cases by running the eager model to get expected outputs
#    method_test_suites = [
#        MethodTestSuite(
#            method_name=m_name,
#            test_cases=[
#                MethodTestCase(inputs=inp, expected_outputs=model(*inp)) for inp in test_inputs
#            ],
#        )
#    ]
#
#    # Generate BundledProgram using the existing et_program_manager
#    bundled_program = BundledProgram(et_program_manager, method_test_suites)
#
#    # Serialize BundledProgram to flatbuffer
#    serialized_bundled_program = serialize_from_bundled_program_to_flatbuffer(
#        bundled_program
#    )
#    bundled_program_path = os.path.join(temp_dir, "bundled_program.bpte")
#    with open(bundled_program_path, "wb") as f:
#        f.write(serialized_bundled_program)
#

######################################################################
# Step 3: Run with Devtool Example Runner
# -------------------------------------
#
# This step we will verify the final result and generate etdump for next
# step usage by using devtool example runner.
#
# First, build the example runner with XNNPACK backend support:
#
# .. code-block:: bash
#
#    cd /path/to/executorch
#    ./examples/devtools/build_example_runner.sh --xnnpack
#
# where ``--xnnpack`` is a build flag that enables XNNPACK backend support.
# Then run the example runner with output verification and debug output enabled:
#
# .. code-block:: bash
#
#    cmake-out/examples/devtools/example_runner \
#        --bundled_program_path=/path/to/bundled_program.bpte \
#        --output_verification \
#        --dump_intermediate_outputs \
#        --debug_buffer_size=1073741824
#
# The key flags are:
#
# - ``--output_verification``: Compare runtime outputs against the expected
#   outputs stored in the BundledProgram (uses rtol=1e-3, atol=1e-5)
# - ``--dump_intermediate_outputs``: Capture intermediate outputs for
#   operator-level debugging
# - ``--debug_buffer_size=<bytes>``: Size of debug buffer (1GB in this example)
#
# Example output on success:
#
# .. code-block:: text
#
#    I 00:00:00.123456 executorch:example_runner.cpp:135] Model file bundled_program.bpte is loaded.
#    I 00:00:00.123456 executorch:example_runner.cpp:145] Running method forward
#    I 00:00:00.234567 executorch:example_runner.cpp:250] Model executed successfully.
#    I 00:00:00.234567 executorch:example_runner.cpp:287] Model verified successfully.
#
# If verification fails (outputs don't match within tolerance), you'll see an error:
#
# .. code-block:: text
#
#    E 00:00:00.234567 executorch:example_runner.cpp:287] Bundle verification failed with status 0x10
#
# This will also generate:
#
# - ``etdump.etdp``: The ETDump file containing execution trace (default path, configurable via ``--etdump_path``)
# - ``debug_output.bin``: The debug buffer containing intermediate outputs (default path, configurable via ``--debug_output_path``)

######################################################################
# Step 4: Analyze Results in Python
# ---------------------------------
#
# After running the model with the CMake runner, load the generated artifacts
# back into Python for analysis using the Inspector.
#
# .. code-block:: python
#
#    from executorch.devtools import Inspector
#
#    etrecord_path = "/path/to/etrecord.bin"
#    etdump_path = "/path/to/etdump.etdp"
#    debug_buffer_path = "/path/to/debug_output.bin"
#
#    inspector = Inspector(
#        etdump_path=etdump_path,
#        etrecord=etrecord_path,
#        debug_buffer_path=debug_buffer_path,
#    )
#
# Then use the same analysis techniques as in Pipeline 1:
#
# .. code-block:: python
#
#    import pandas as pd
#
#    # Calculate numerical gaps
#    df = inspector.calculate_numeric_gap("MSE")
#
#    # Find problematic operators
#    df_sorted = df.sort_values(by="gap", ascending=False,
#        key=lambda x: x.apply(lambda y: y[0] if isinstance(y, list) else y))
#    print("Top operators with largest gaps:")
#    print(df_sorted.head(5))
#

######################################################################
# Best Practices for Debugging
# ============================
#
# 1. **Start with final outputs**: Always compare the final model output first
#    before diving into operator-level analysis. This saves time if outputs match.
#
# 2. **Use appropriate thresholds**: Small numerical differences (< 1e-6) are
#    typically acceptable. Focus on operators with gaps > 1e-4.
#
# 3. **Focus on delegated operators**: Numerical discrepancies are most common
#    in delegated operators (shown as ``DELEGATE_CALL``) due to different
#    precision handling in delegate backends.
#
# 4. **Check accumulation patterns**: In transformer models, attention layers
#    often show larger gaps due to accumulated numerical differences across
#    many operations.
#
# 5. **Use stack traces**: With ETRecord, you can trace operators back to the
#    original PyTorch source code for easier debugging using
#    ``event.stack_traces`` and ``event.module_hierarchy``.
#

######################################################################
# Conclusion
# ----------
#
# In this tutorial, we learned how to use the ExecuTorch Developer Tools
# to debug numerical discrepancies in models. The key workflow is:
#
# 1. Export the model with ETRecord generation enabled
# 2. Run the model with debug buffer enabled (Python or CMake)
# 3. **First** compare final outputs between eager and runtime models
# 4. **If issues found**, use ``calculate_numeric_gap`` for operator-level analysis
# 5. Identify and investigate operators with significant gaps
#
# Links Mentioned
# ^^^^^^^^^^^^^^^
#
# - `ExecuTorch Developer Tools Overview <../devtools-overview.html>`__
# - `ETRecord <../etrecord.html>`__
# - `ETDump <../etdump.html>`__
# - `Inspector <../model-inspector.html>`__
# - `Model Debugging Guide <../model-debugging.html>`__
# - `Profiling Tutorial <devtools-integration-tutorial.html>`__

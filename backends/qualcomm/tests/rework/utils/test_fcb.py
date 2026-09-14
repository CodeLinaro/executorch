# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.

from unittest.mock import Mock

import pytest
import torch

from executorch.backends.qualcomm.export_utils import (
    QcomChipset,
    QnnExecuTorchBackendType,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
)
from executorch.backends.qualcomm.serialization.qc_schema_serialize import flatbuffer_to_option
from executorch.backends.qualcomm.tests.rework.conftest import Tolerance, export_and_verify
from executorch.backends.qualcomm.utils import qnn_manager_lifecycle as lifecycle


def test_fcb_compiler_spec_preserves_targets():
    option = flatbuffer_to_option(
        generate_qnn_executorch_compiler_spec(
            soc_model=[QcomChipset.SM8650, QcomChipset.SM8750],
            backend_options=[
                generate_htp_compiler_spec(use_fp16=False),
                generate_htp_compiler_spec(use_fp16=True),
            ],
        )[0].value
    )

    assert [target.soc_info.soc_model for target in option.target_options.targets] == [
        QcomChipset.SM8650,
        QcomChipset.SM8750,
    ]


def test_fcb_manager_cache_is_keyed_by_soc(monkeypatch):
    managers = [Mock(), Mock(), Mock()]
    for manager in managers:
        manager.InitBackend.return_value = Mock(value=0)

    monkeypatch.setattr(lifecycle, "setup_qnn_sdk", lambda: None)
    monkeypatch.setattr(lifecycle, "disable_mkldnn_on_amd", lambda: None)
    create = Mock(side_effect=managers)
    monkeypatch.setattr(lifecycle.PyQnnManager, "QnnManager", create)
    registry = lifecycle.QnnManagerRegistry()
    first = registry.get_or_create_qnn_manager(
        QnnExecuTorchBackendType.kHtpBackend, b"first", QcomChipset.SM8650
    )
    second = registry.get_or_create_qnn_manager(
        QnnExecuTorchBackendType.kHtpBackend, b"second", QcomChipset.SM8750
    )
    third = registry.get_or_create_qnn_manager(
        QnnExecuTorchBackendType.kHtpBackend, b"third", QcomChipset.SM8650
    )

    assert create.call_count == 2
    assert first is not second
    assert first is third


def test_fcb_e2e(qnn_config):
    if qnn_config.build_folder == "build-x86":
        pytest.skip("FCB execution requires an Android HTP target")

    selected_soc = getattr(QcomChipset, qnn_config.soc_model)
    other_soc = QcomChipset.SM8650 if selected_soc != QcomChipset.SM8650 else QcomChipset.SM8750
    compile_specs = generate_qnn_executorch_compiler_spec(
        soc_model=[selected_soc, other_soc],
        backend_options=[
            generate_htp_compiler_spec(use_fp16=False),
            generate_htp_compiler_spec(use_fp16=False),
        ],
    )
    module = torch.nn.ReLU()
    inputs = (torch.randn(1, 3, 4, 4),)
    export_and_verify(
        module,
        inputs,
        qnn_config,
        None,
        compile_specs,
        Tolerance(),
    )

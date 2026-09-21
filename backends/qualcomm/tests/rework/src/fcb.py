# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from unittest.mock import Mock

import pytest
import torch

from executorch.backends.qualcomm.export_utils import (
    QcomChipset,
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
)
from executorch.backends.qualcomm.tests.fcb_utils import (
    fcb_target_socs,
    lower_fcb_weight_sharing_model,
    make_fcb_weight_sharing_model,
)
from executorch.backends.qualcomm.tests.rework.conftest import (
    export_and_verify,
    temp_attribute,
)
from executorch.backends.qualcomm.utils import qnn_manager_lifecycle as lifecycle


def unpack_fixtures(func):
    def wrapper(request, kwargs):
        params = inspect.signature(func).parameters
        extra_fixtures = set(params.keys()) - set(kwargs.keys())
        new_kwargs = {key: request.getfixturevalue(key) for key in extra_fixtures}
        with temp_attribute(
            new_kwargs["qnn_config"], "device_workspace", __name__.replace(".", "_")
        ):
            return func(**new_kwargs, **kwargs)

    return wrapper


class Fcb:
    @staticmethod
    def compiler_spec_preserves_targets(fcb_compile_specs):
        soc_models = (QcomChipset.SM8650, QcomChipset.SM8750)
        option = flatbuffer_to_option(fcb_compile_specs(soc_models)[0].value)

        assert [
            target.soc_info.soc_model for target in option.target_options.targets
        ] == list(soc_models)

    @staticmethod
    def manager_cache_is_keyed_by_soc(monkeypatch):
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

    @staticmethod
    @unpack_fixtures
    def e2e(qnn_config, fcb_compile_specs, expected):
        if qnn_config.build_folder == "build-x86":
            pytest.skip("FCB execution requires an Android HTP target")

        selected_soc = getattr(QcomChipset, qnn_config.soc_model)
        soc_models = tuple(fcb_target_socs(selected_soc))
        module = torch.nn.ReLU()
        inputs = (torch.randn(1, 3, 4, 4),)
        with expected as metrics:
            export_and_verify(
                module=module,
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=None,
                compile_specs=fcb_compile_specs(soc_models),
                metrics=metrics,
            )

    @staticmethod
    @unpack_fixtures
    def reference_weight_sharing_reduces_pte_size(qnn_config, fcb_compile_specs):
        if qnn_config.build_folder == "build-x86":
            pytest.skip("FCB reference-weight sharing requires an Android HTP target")

        module, inputs = make_fcb_weight_sharing_model()
        soc_models = tuple(fcb_target_socs(getattr(QcomChipset, qnn_config.soc_model)))
        shared = lower_fcb_weight_sharing_model(
            module, inputs, fcb_compile_specs(soc_models, True)
        )
        unshared = lower_fcb_weight_sharing_model(
            module, inputs, fcb_compile_specs(soc_models, False)
        )

        assert len(shared.buffer) < len(unshared.buffer)

    @staticmethod
    @unpack_fixtures
    def reference_weight_sharing_e2e(qnn_config, fcb_compile_specs, expected):
        if qnn_config.build_folder == "build-x86":
            pytest.skip("FCB reference-weight sharing requires an Android HTP target")

        module, inputs = make_fcb_weight_sharing_model()
        soc_models = tuple(fcb_target_socs(getattr(QcomChipset, qnn_config.soc_model)))
        with expected as metrics:
            export_and_verify(
                module=module,
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=None,
                compile_specs=fcb_compile_specs(soc_models, True),
                metrics=metrics,
            )

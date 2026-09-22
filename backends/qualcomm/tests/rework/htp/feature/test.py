# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from executorch.backends.qualcomm.tests.rework.conftest import Tolerance
from executorch.backends.qualcomm.tests.rework.src.fcb import Fcb


def test_fcb_compiler_spec_preserves_targets(fcb_compile_specs):
    Fcb.compiler_spec_preserves_targets(fcb_compile_specs)


def test_fcb_manager_cache_is_keyed_by_soc(monkeypatch):
    Fcb.manager_cache_is_keyed_by_soc(monkeypatch)


def test_fcb_dlc_handle_enforces_lifetime_and_owner(fcb_compile_specs):
    Fcb.dlc_handle_enforces_lifetime_and_owner(fcb_compile_specs)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"expected": Tolerance()}, id="e2e"),
    ],
)
def test_fcb_e2e(request, kwargs):
    Fcb.e2e(request, kwargs)


def test_fcb_reference_weight_sharing_reduces_pte_size(request):
    Fcb.reference_weight_sharing_reduces_pte_size(request, {})


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"expected": Tolerance()}, id="e2e"),
    ],
)
def test_fcb_reference_weight_sharing_e2e(request, kwargs):
    Fcb.reference_weight_sharing_e2e(request, kwargs)

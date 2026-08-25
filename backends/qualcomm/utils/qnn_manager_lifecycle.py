import contextlib
import copy
import logging
import threading
from typing import Dict, List

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

from executorch.backends.qualcomm.partition.utils import generate_qnn_executorch_option
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
    option_to_flatbuffer,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec

# Thread-local storage for QnnManager instances
_current_qnn_managers = threading.local()


class QnnManagerRegistry:
    def __init__(self):
        self._registry = {}

    def get_or_create_qnn_manager(
        self, backend_type: QnnExecuTorchBackendType, option: bytes, soc_model=None
    ) -> PyQnnManager.QnnManager:
        key = (backend_type, soc_model)
        if key not in self._registry:
            qnn_manager = PyQnnManager.QnnManager(option)
            err = qnn_manager.InitBackend()
            if err.value != 0:
                raise RuntimeError(
                    f"Failed to initialize QNN backend for {backend_type.name}. "
                    "Ensure QNN SDK libraries are available "
                    "(e.g. LD_LIBRARY_PATH includes $QNN_SDK_ROOT/lib/x86_64-linux-clang/)."
                )
            self._registry[key] = qnn_manager
        return self._registry[key]

    def destroy_all(self):
        for qnn_manager in self._registry.values():
            qnn_manager.Destroy()
        self._registry.clear()


@contextlib.contextmanager
def QnnManagerContext(compile_specs: Dict[str, List[CompileSpec]]):
    # Create a new registry for the current context
    current_context_registry = QnnManagerRegistry()
    _current_qnn_managers.active_registry = current_context_registry

    try:
        for compile_spec_list in compile_specs.values():
            option = generate_qnn_executorch_option(compile_spec_list)
            python_options = flatbuffer_to_option(option)
            backend_type = python_options.backend_options.backend_type
            fcb_options = python_options.fcb_options
            if fcb_options is None:
                current_context_registry.get_or_create_qnn_manager(backend_type, option)
            else:
                for soc_info in fcb_options.soc_infos:
                    per_soc_options = copy.deepcopy(python_options)
                    per_soc_options.soc_info = soc_info
                    current_context_registry.get_or_create_qnn_manager(
                        backend_type,
                        option_to_flatbuffer(per_soc_options),
                        soc_info.soc_model,
                    )
        yield
    finally:
        current_context_registry.destroy_all()

        # Clear the active registry reference
        _current_qnn_managers.active_registry = None


def get_current_qnn_managers(
    backend_type: QnnExecuTorchBackendType, compile_specs: List[CompileSpec]
) -> List[PyQnnManager.QnnManager]:
    active_registry = getattr(_current_qnn_managers, "active_registry", None)
    if active_registry is None:
        option = generate_qnn_executorch_option(compile_specs)
        return [QnnManagerRegistry().get_or_create_qnn_manager(backend_type, option)]
    managers = [
        manager
        for (registered_backend, _), manager in active_registry._registry.items()
        if registered_backend == backend_type
    ]
    if not managers:
        raise RuntimeError(f"No QNN manager active for {backend_type.name}")
    return managers


def get_current_qnn_manager(
    backend_type: QnnExecuTorchBackendType, compile_specs: List[CompileSpec]
) -> PyQnnManager.QnnManager:
    return get_current_qnn_managers(backend_type, compile_specs)[0]

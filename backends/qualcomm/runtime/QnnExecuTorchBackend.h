/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

#include <mutex>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace qnn {

class QnnExecuTorchBackend final
    : public ::executorch::runtime::BackendInterface {
 public:
  ~QnnExecuTorchBackend(){};

  executorch::runtime::Result<executorch::runtime::DelegateHandle*> init(
      executorch::runtime::BackendInitContext& context,
      executorch::runtime::FreeableBuffer* processed,
      executorch::runtime::ArrayRef<executorch::runtime::CompileSpec>
          compile_specs) const override;

  executorch::runtime::Error execute(
      ET_UNUSED executorch::runtime::BackendExecutionContext& context,
      executorch::runtime::DelegateHandle* handle,
      executorch::runtime::EValue** args) const override;

  void destroy(executorch::runtime::DelegateHandle* handle) const override;

  bool is_available() const override;

 private:
  static void add_cached_delegate(
      uint32_t hash_val,
      executorch::runtime::DelegateHandle* handle);
  static void erase_cached_delegate(
      executorch::runtime::DelegateHandle* handle);

  static std::mutex mutex_;
  static std::unordered_map<uint32_t, executorch::runtime::DelegateHandle*>
      delegate_map_;
  static std::unordered_map<executorch::runtime::DelegateHandle*, uint32_t>
      delegate_map_rev_;
  mutable std::string method_name_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch

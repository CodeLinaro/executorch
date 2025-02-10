/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnLogger.h>

#include <vector>

#include "HTP/QnnHtpCommon.h"
#include "IR/QnnIrCommon.h"
#include "QnnBackend.h"
#include "QnnCommon.h"
#include "QnnTypes.h"
#include "Saver/QnnSaverCommon.h"
namespace executorch {
namespace backends {
namespace qnn {
// qnn backend
class QnnBackend {
 public:
  explicit QnnBackend(
      const QnnImplementation& implementation,
      QnnLogger* logger)
      : handle_(nullptr), implementation_(implementation), logger_(logger) {}

  virtual ~QnnBackend();
  virtual bool IsProfileEventTypeParentOfNodeTime(
      QnnProfile_EventType_t /*event_type*/) {
    return false;
  }

  executorch::runtime::Error Configure();

  Qnn_ErrorHandle_t BackendValidateOpConfig(const Qnn_OpConfig_t& op_config) {
    return implementation_.GetQnnInterface().qnn_backend_validate_op_config(
        handle_, op_config);
  };

  Qnn_BackendHandle_t GetHandle() {
    return handle_;
  }

  executorch::runtime::Error VerifyQNNSDKVersion();

 protected:
  virtual Qnn_Version_t GetExpectedBackendVersion() {
    Qnn_Version_t backend_version;
    backend_version.major = QNN_IR_API_VERSION_MAJOR;
    backend_version.minor = QNN_IR_API_VERSION_MINOR;
    backend_version.patch = QNN_IR_API_VERSION_PATCH;
    return backend_version;
  }
  virtual executorch::runtime::Error MakeConfig(
      std::vector<const QnnBackend_Config_t*>& config) {
    return executorch::runtime::Error::Ok;
  };

 private:
  Qnn_BackendHandle_t handle_;
  const QnnImplementation& implementation_;
  QnnLogger* logger_;
  executorch::runtime::Error VersionChecker(
      const Qnn_Version_t& qnn_version,
      const Qnn_Version_t& expected,
      const std::string& prefix);
};
} // namespace qnn
} // namespace backends
} // namespace executorch

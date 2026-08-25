# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestFcbMultiSocWeightSharingDemo(unittest.TestCase):
    def test_demo_reports_artifact_summary_for_two_socs(self):
        soc_models = ["SM8650", "SM8750"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "executorch.examples.qualcomm.util_scripts.fcb_multi_soc_weight_sharing_demo",
                    "--soc_models",
                    *soc_models,
                    "--output_dir",
                    tmp_dir,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )

            output_dir = Path(tmp_dir)
            self.assertEqual(len(list(output_dir.glob("shared*.pte"))), 1)
            self.assertEqual(len(list(output_dir.glob("unshared*.pte"))), 1)

            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(summary["requested_socs"], soc_models)
            self.assertLess(
                summary["shared_pte_bytes"], summary["unshared_pte_bytes"]
            )
            self.assertEqual(summary["cache_record_count"], len(soc_models))
            self.assertEqual(summary["intermediate_context_binary_bytes_in_python"], 0)
            self.assertEqual(summary["max_live_contexts"], 1)


if __name__ == "__main__":
    unittest.main()

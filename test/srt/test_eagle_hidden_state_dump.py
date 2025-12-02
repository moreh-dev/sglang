import os
import time
import glob
import tempfile
import unittest

import torch

import sglang as sgl
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_EAGLE3,
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST_EAGLE3,
    CustomTestCase,
)


class TestEAGLEHiddenStateDump(CustomTestCase):
    def test_hidden_state_dump_eagle3(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = sgl.Engine(
                model_path=DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST_EAGLE3,
                speculative_draft_model_path=DEFAULT_MODEL_NAME_FOR_TEST_EAGLE3,
                speculative_algorithm="EAGLE3",
                speculative_num_steps=3,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=4,
                speculative_eagle_enable_dump_hidden_states=True,
                speculative_eagle_hidden_states_dump_path=tmpdir,
                speculative_eagle_dump_worker_num=1,
                speculative_eagle_dump_buffer_pool_size=64,
                dtype="bfloat16",
                cuda_graph_max_bs=2,
            )

            prompt = "Today is a sunny day and I like"
            sampling_params = {"temperature": 0, "max_new_tokens": 8}

            output = engine.generate(prompt, sampling_params)

            # wait for dump outputs
            deadline = time.time() + 10
            dump_files = []
            while time.time() < deadline:
                dump_files = glob.glob(os.path.join(tmpdir, "*_data.ckpt"))
                if dump_files:
                    break
                time.sleep(0.2)
            engine.shutdown()

            # After engine shutdown, dump files should exist
            dump_files = sorted(glob.glob(os.path.join(tmpdir, "*_data.ckpt")))
            self.assertGreater(len(dump_files), 0, "Dump files not found")

            # Load one dump file and check contents
            dump_path = dump_files[0]
            save_dict = torch.load(dump_path, map_location="cpu")

            # Expect key fields
            self.assertIn("input_ids", save_dict)
            self.assertIn("loss_mask", save_dict)
            self.assertIn("hidden_state", save_dict)  # last hidden
            self.assertIn("aux_hidden_state", save_dict)  # aux hidden

            # Basic shape checks
            self.assertTrue(torch.is_tensor(save_dict["hidden_state"]))
            self.assertTrue(torch.is_tensor(save_dict["aux_hidden_state"]))


if __name__ == "__main__":
    unittest.main()

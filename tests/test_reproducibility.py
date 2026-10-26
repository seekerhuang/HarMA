import random
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from utils.reproducibility import seed_before_model_import


class RetrievalSeedTests(unittest.TestCase):
    def test_yaml_seed_is_applied_before_parameter_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.yaml'
            config.write_text('seed: 777\n')
            values = []
            for _ in range(2):
                self.assertEqual(seed_before_model_import(['--config', str(config)]), 777)
                values.append((random.random(), np.random.rand(), torch.nn.Linear(8, 8).weight.detach().clone()))
            self.assertEqual(values[0][0], values[1][0])
            self.assertEqual(values[0][1], values[1][1])
            self.assertTrue(torch.equal(values[0][2], values[1][2]))

    def test_cli_seed_overrides_yaml_in_both_argument_forms(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.yaml'
            config.write_text('seed: 777\n')
            for extra in (['--seed', '42'], ['--seed=42']):
                self.assertEqual(seed_before_model_import(['--config=' + str(config), *extra]), 42)
                actual = torch.rand(4)
                torch.manual_seed(42)
                self.assertTrue(torch.equal(actual, torch.rand(4)))

    def test_default_seed_without_config(self):
        self.assertEqual(seed_before_model_import([]), 3407)

    def test_config_without_seed_keeps_legacy_default(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.yaml'
            config.write_text('model: vit\n')
            self.assertEqual(seed_before_model_import(['--config', str(config)]), 3407)


if __name__ == '__main__':
    unittest.main()

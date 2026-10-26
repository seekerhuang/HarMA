"""Seed adapters before import and run retrieval with FP32 parameters and TF32."""
from pathlib import Path
import runpy
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.reproducibility import seed_before_model_import

# Shared adapters are constructed at import time, before Retrieval.main().
seed = seed_before_model_import()
torch.set_float32_matmul_precision('high')
print(f'FP32 parameters; TF32 enabled; adapter initialization seed={seed}', flush=True)
runpy.run_path(str(ROOT / 'Retrieval.py'), run_name='__main__')

"""Seed retrieval before modules construct their global adapters."""
import argparse
from pathlib import Path
import random

import numpy as np
import torch
from ruamel.yaml import YAML


def seed_before_model_import(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--config')
    parser.add_argument('--seed', type=int)
    args, _ = parser.parse_known_args(argv)
    seed = args.seed
    if seed is None:
        config = YAML(typ='safe').load(Path(args.config).read_text()) if args.config else {}
        seed = int((config or {}).get('seed', 3407))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

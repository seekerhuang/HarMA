"""Keep one global best training checkpoint for each of the four groups."""
import json
import os
from pathlib import Path
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'artifacts/checkpoint-registry.json'
GROUP_DIRS = {'rsicd_vit': 'RSICD_CLIP', 'rsicd_geo': 'RSICD_Geo',
              'rsitmd_vit': 'RSITMD_CLIP', 'rsitmd_geo': 'RSITMD_Geo'}


def retain_best(save_obj, config, output_dir):
    group = config.get('checkpoint_group')
    if group and group not in GROUP_DIRS:
        raise ValueError(f'Unknown checkpoint group: {group}')
    registry = (json.loads(REGISTRY.read_text()) if REGISTRY.exists() else
                {'schema_version': 1, 'policy': 'four_groups_global_best_only', 'groups': {}}) if group else None
    score = save_obj['test_result']['r_mean']
    if group and score <= registry['groups'].get(group, {}).get('best_mR', float('-inf')):
        return False
    destination = ROOT / 'checkpoints' / GROUP_DIRS[group] if group else Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / 'checkpoint_best.tmp'
    path = destination / 'checkpoint_best.pth'
    save_obj['trial_name'] = config.get('trial_name', Path(output_dir).name)
    torch.save(save_obj, temporary)
    os.replace(temporary, path)
    if group:
        registry['groups'][group] = {
            'best_mR': score, 'metrics': save_obj['test_result'],
            'epoch': save_obj['epoch'], 'seed': save_obj['seed'],
            'source_run': save_obj['trial_name'], 'checkpoint': str(path),
            'loss_balance': 'self_detached' if config.get('normalize_losses') else 'original_sum',
            'has_optimizer_state': 'optimizer' in save_obj, 'updated_at': time.time()}
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        tmp = REGISTRY.with_suffix('.tmp')
        tmp.write_text(json.dumps(registry, indent=2))
        os.replace(tmp, REGISTRY)
        print(f'Updated global best {group}: {score:.2f}', flush=True)
    return True

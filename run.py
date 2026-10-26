import os
import sys
import time
import random
import argparse
import shlex

from utils.hdfs_io import HADOOP_BIN, hexists, hmkdir, hcopy

def get_dist_launch(args):
    devices = {'f4': '0,1,2,3', 'f2': '0,1', 'f3': '0,1,2',
               'f12': '1,2', 'f02': '0,2', 'f03': '0,3', 'l2': '2,3'}
    if args.dist.startswith('gpu'):
        device = int(args.dist[3:])
        if not 0 <= device <= 8:
            raise ValueError('Invalid GPU index')
        selected = str(device)
    else:
        selected = devices[args.dist]
    count = len(selected.split(','))
    return (f'CUDA_VISIBLE_DEVICES={selected} WORLD_SIZE={count} '
            f'{shlex.quote(sys.executable)} -W ignore -m torch.distributed.run '
            f'--standalone --nnodes=1 --nproc_per_node={count} ')


def get_from_hdfs(file_hdfs):
    """
    compatible to HDFS path or local path
    """
    if file_hdfs.startswith('hdfs'):
        file_local = os.path.split(file_hdfs)[-1]

        if os.path.exists(file_local):
            print(f"rm existing {file_local}")
            os.system(f"rm {file_local}")

        hcopy(file_hdfs, file_local)

    else:
        file_local = file_hdfs
        assert os.path.exists(file_local)

    return file_local


def run_retrieval(args):
    dist_launch = get_dist_launch(args)

    seed_arg = f"--seed {args.seed} " if args.seed is not None else ""
    os.system(f"{dist_launch}"
              f"scripts/train_entry.py {seed_arg}--config {args.config} "
              f"--output_dir {args.output_dir} --bs {args.bs} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''}")


def run(args):
    if args.task == 'itr_rsicd_vit':
        # assert os.path.exists("../X-VLM-pytorch/images/rsicd")
        args.config = 'configs/Retrieval_rsicd_vit.yaml'
        run_retrieval(args)

    elif args.task == 'itr_rsitmd_vit':
        # assert os.path.exists("../X-VLM-pytorch/images/rsitmd")
        args.config = 'configs/Retrieval_rsitmd_vit.yaml'
        run_retrieval(args)
    elif args.task == 'itr_rsitmd_geo':
        # assert os.path.exists("../X-VLM-pytorch/images/rsitmd")
        args.config = 'configs/Retrieval_rsitmd_geo.yaml'
        run_retrieval(args)
    elif args.task == 'itr_rsicd_geo':
        # assert os.path.exists("../X-VLM-pytorch/images/rsicd")
        args.config = 'configs/Retrieval_rsicd_geo.yaml'
        run_retrieval(args)

    elif args.task == 'itr_coco':
        assert os.path.exists("../X-VLM-pytorch/images/coco")
        args.config = 'configs/Retrieval_coco.yaml'
        run_retrieval(args)

    elif args.task == 'itr_nwpu':
        assert os.path.exists("../X-VLM-pytorch/images/NWPU")
        args.config = 'configs/Retrieval_nwpu.yaml'
        run_retrieval(args)
    else:
        raise NotImplementedError(f"task == {args.task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='itr_rsitmd_vit')
    parser.add_argument('--dist', type=str, default='gpu0', help="see func get_dist_launch for details")
    parser.add_argument('--config', default='configs/Retrieval_rsitmd_vit.yaml', type=str, help="if not given, use default")
    parser.add_argument('--bs', default=428, type=int, help="for each gpu, batch_size = bs // num_gpus; "
                                                  "this option only works for fine-tuning scripts.")
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--checkpoint', default='-1', type=str, help="for fine-tuning")
    parser.add_argument('--load_ckpt_from', default=' ', type=str, help="load domain pre-trained params")
    # write path: local or HDFS
    parser.add_argument('--output_dir', type=str, default='./outputs/test', help='for fine-tuning, local path; '
                                                                      'for pre-training, local and HDFS are both allowed.')
    parser.add_argument('--evaluate', action='store_true', default=False, help="evaluation on downstream tasks")
    args = parser.parse_args()
    hmkdir(args.output_dir)
    run(args)


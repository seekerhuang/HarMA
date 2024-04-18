import os
import sys
import time
import random
import argparse

from utils.hdfs_io import HADOOP_BIN, hexists, hmkdir, hcopy

def get_dist_launch(args):  # some examples

    if args.dist == 'f4':
        return "CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 /home/pjc/.conda/envs/xlvm/bin/python -W ignore -m torch.distributed.launch --master_port 9999 --nproc_per_node=4 " \
               "--nnodes=1 "

    elif args.dist == 'f2':
        return "CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 /root/miniconda3/bin/python -W ignore -m torch.distributed.launch --master_port 9999 --nproc_per_node=2 " \
               "--nnodes=1 "

    elif args.dist == 'f3':
        return "CUDA_VISIBLE_DEVICES=0,1,2 WORLD_SIZE=3 /home/pjc/.conda/envs/xlvm/bin/python -W ignore -m torch.distributed.launch --master_port 9999 --nproc_per_node=3 " \
               "--nnodes=1 "

    elif args.dist == 'f12':
        return "CUDA_VISIBLE_DEVICES=1,2 WORLD_SIZE=2 /home/pjc/.conda/envs/xlvm/bin/python -W ignore -m torch.distributed.launch --master_port 9999 --nproc_per_node=2 " \
               "--nnodes=1 "

    elif args.dist == 'f02':
        return "CUDA_VISIBLE_DEVICES=0,2 WORLD_SIZE=2 /home/pjc/.conda/envs/xlvm/bin/python -W ignore -m torch.distributed.launch --master_port 9999 --nproc_per_node=2 " \
               "--nnodes=1 "

    elif args.dist == 'f03':
        return "CUDA_VISIBLE_DEVICES=0,3 WORLD_SIZE=2 /home/pjc/.conda/envs/xlvm/bin/python -W ignore -m torch.distributed.launch --master_port 9999 --nproc_per_node=2 " \
               "--nnodes=1 "

    elif args.dist == 'l2':
        return "CUDA_VISIBLE_DEVICES=2,3 WORLD_SIZE=2 /home/pjc/.conda/envs/xlvm/bin/python -W ignore -m torch.distributed.launch --master_port 9998 --nproc_per_node=2 " \
               "--nnodes=1 "

    elif args.dist.startswith('gpu'):  # use one gpu, --dist "gpu0"
        num = int(args.dist[3:])
        assert 0 <= num <= 8
        return "CUDA_VISIBLE_DEVICES={:} WORLD_SIZE=1 /home/pjc/.conda/envs/xlvm/bin/python -W ignore -m torch.distributed.launch --master_port 9999 --nproc_per_node=1 " \
               "--nnodes=1 ".format(num)

    else:
        raise ValueError


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

    os.system(f"{dist_launch} "
              f"--use_env Retrieval.py --config {args.config} "
              f"--output_dir {args.output_dir} --bs {args.bs} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''}")


def run(args):
    if args.task == 'itr_rsicd':
        # assert os.path.exists("../X-VLM-pytorch/images/rsicd")
        args.config = 'configs/Retrieval_rsicd.yaml'
        run_retrieval(args)

    elif args.task == 'itr_rsitmd':
        # assert os.path.exists("../X-VLM-pytorch/images/rsitmd")
        args.config = 'configs/Retrieval_rsitmd.yaml'
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
    parser.add_argument('--task', type=str, default='itr_rsitmd')
    parser.add_argument('--dist', type=str, default='f2', help="see func get_dist_launch for details")
    parser.add_argument('--config', default='configs/Retrieval_rsitmd.yaml', type=str, help="if not given, use default")
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus; "
                                                  "this option only works for fine-tuning scripts.")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--checkpoint', default='-1', type=str, help="for fine-tuning")
    parser.add_argument('--load_ckpt_from', default=' ', type=str, help="load domain pre-trained params")
    # write path: local or HDFS
    parser.add_argument('--output_dir', type=str, default='./outputs/test', help='for fine-tuning, local path; '
                                                                      'for pre-training, local and HDFS are both allowed.')
    parser.add_argument('--evaluate', action='store_true', default=False, help="evaluation on downstream tasks")
    args = parser.parse_args()
    assert hexists(os.path.dirname(args.output_dir))
    hmkdir(args.output_dir)
    run(args)


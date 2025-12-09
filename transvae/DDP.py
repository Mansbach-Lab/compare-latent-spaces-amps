import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

import torch.distributed as dist
import torch.utils.data.distributed
#lgging from https://discuss.pytorch.org/t/ddp-training-log-issue/125808

 ### Build model architecture
def DDP_init(self):
    ### prepare distributed data parallel (added by Samuel Renaud)
    os.system("echo GPUs per node: {}".format(torch.cuda.device_count()))
    print("echo GPUs per node: {}".format(torch.cuda.device_count()))
    ngpus_per_node = torch.cuda.device_count()

    """ This next line is the key to getting DistributedDataParallel working on SLURM:
        SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
        current process inside a node and is also 0 or 1 in this example."""
    print("local: ",os.environ.get("SLURM_LOCALID")," node: ",os.environ.get("SLURM_NODEID"))
    if os.environ.get("SLURM_LOCALID") ==None and os.environ.get("SLURM_NODEID")==None:#for local testing before CC training
        os.environ["SLURM_LOCALID"] = '0'
        os.environ["SLURM_NODEID"] = '0'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

    local_rank = int(os.environ.get("SLURM_LOCALID")) 
    rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank

    """ This next block parses CUDA_VISIBLE_DEVICES to find out which GPUs have been allocated to the job, then sets torch.device to the GPU
        corresponding       to the local rank (local rank 0 gets the first GPU, local rank 1 gets the second GPU etc) """
    print('cuda visible: ',os.environ.get('CUDA_VISIBLE_DEVICES'))
    if os.environ.get('CUDA_VISIBLE_DEVICES') == None:
        available_gpus = 1
        current_device = torch.device('cuda:0')
        torch.cuda.set_device(current_device)
    else: 
        available_gpus = list(os.environ.get('CUDA_VISIBLE_DEVICES').replace(',',""))
        current_device = int(available_gpus[local_rank])
        torch.cuda.set_device(current_device)

    self.build_model()

    """ this block initializes a process group and initiate communications
        between all processes running on all nodes """
    #print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    os.system("os.system:echo From Rank: {}, ==> Initializing Process Group...".format(rank))
    print("echo From Rank: {}, ==> Initializing Process Group...".format(rank))
    print(f"echo rank={rank},local_rank={local_rank},ngpus_per_node={ngpus_per_node},current_device={current_device}")
    print(f"backend={self.params['DIST_BACKEND']},init_method={self.params['INIT_METHOD']},world_size={self.params['WORLD_SIZE']}")
    print("SLURM_LOCALID:", os.environ.get("SLURM_LOCALID"))
    print("SLURM_NODEID:", os.environ.get("SLURM_NODEID"))
    print("MASTER_ADDR:", os.environ.get("MASTER_ADDR"))
    print("MASTER_PORT:", os.environ.get("MASTER_PORT"))
    #init the process group
    dist.init_process_group(backend=self.params['DIST_BACKEND'], init_method=self.params['INIT_METHOD'],
                            world_size=self.params['WORLD_SIZE'], rank=rank)
    os.system("echo process group ready!")
    os.system('echo From Rank: {}, ==> Making model..'.format(rank))
    print("process group ready!")
    print('From Rank: {}, ==> Making model..'.format(rank))
    print("echo final check; ngpus_per_node={},local_rank={},rank={},available_gpus={},current_device={}"
              .format(ngpus_per_node,local_rank,rank,available_gpus,current_device))
    os.system("echo final check; ngpus_per_node={},local_rank={},rank={},available_gpus={},current_device={}"
              .format(ngpus_per_node,local_rank,rank,available_gpus,current_device))


    self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[current_device])
    print('passed distributed data parallel call')


def reduce_tensor(tensor):
    rt = tensor.clone().detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    return rt
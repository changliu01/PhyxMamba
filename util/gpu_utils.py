import os
import numpy as np
import logging
import wandb
import torch
# import torch_geometric
import subprocess
# from torch_geometric.utils import to_dense_adj, to_dense_batch
import GPUtil

def get_free_memory():
    gpus = GPUtil.getGPUs()
    return [int(gpu.memoryFree) for gpu in gpus]

class CheckpointSaver:
    def __init__(self, dirpath, decreasing=True, top_n=5, save_every=False, not_save=False):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n 
        self.decreasing = decreasing
        self.top_model_paths = []
        self.save_every = save_every
        self.not_save = not_save
        
    def __call__(self, model, epoch, metric_val, final_epoch=False):
        model_path = os.path.join(self.dirpath, model.__class__.__name__ + f'_epoch{epoch}.pt')

        if self.not_save:
            save = False
        else:
            if self.save_every:
                save = True
            elif final_epoch:
                save = True
            else:
                if len(self.top_model_paths) < self.top_n:
                    save = True
                else:
                    worst_score = self.top_model_paths[-1]['score'] if self.decreasing else self.top_model_paths[0]['score']
                    save = metric_val < worst_score if self.decreasing else metric_val > worst_score

        if save: 
            print(f"Saving model at {model_path}, current metric value {metric_val}.")
            torch.save(model.state_dict(), model_path)
            self.log_artifact(f'model-ckpt-epoch-{epoch}.pt', model_path, metric_val)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)

            if len(self.top_model_paths) > self.top_n:
                self.cleanup()

    def log_artifact(self, filename, model_path, metric_val):
        artifact = wandb.Artifact(filename, type='model', metadata={'Validation score': metric_val})
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)        
    
    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        print(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]

class MetricMonitor:

    def __init__(self):

        self.f1 = -np.inf
        self.acc = -np.inf
        self.mcc = -np.inf
        self.auc = -np.inf
        self.epoch = 0
    
    def update(self, f1, acc, mcc, auc, epoch):

        if f1 > self.f1:
            self.f1 = f1
            self.acc = acc
            self.mcc = mcc
            self.auc = auc
            self.epoch = epoch
            
    def read(self):

        return self.f1, self.acc, self.mcc, self.auc, self.epoch

class PlaceHolder:

    def __init__(self, X, E, y):

        self.X = X
        self.E = E
        self.y = y
    
    def type_as(self, x: torch.Tensor):

        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)

        return self

    def mask(self, node_mask, collapse=False):

        x_mask = node_mask.unsqueeze(-1).unsqueeze(-1) # (bs, n, 1, 1)
        e_mask1 = node_mask.unsqueeze(-1).unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = node_mask.unsqueeze(-1).unsqueeze(1)             # bs, 1, n, 1

        self.X = self.X * x_mask
        self.E = self.E * (e_mask1 * e_mask2).squeeze(-1) # bs, n, n

        return self


def make_model_dirs(path):

    model_path = os.path.join(path, os.path.pardir, 'checkpoints')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

# def to_dense(x, )
        
def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E

# def to_dense_dt(x, edge_index, edge_attr, batch):

#     X, node_mask = to_dense_batch(x, batch)

    

#     edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)

#     max_num_nodes = X.size(1)

#     E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)

#     # E = encode_no_edge(E)

#     # print(E.shape)
#     # print(E[0])

#     return PlaceHolder(X=X, E=E, y=None), node_mask

class AutoGPU():

    def __init__(self, memory_size):

        self.memory_size = memory_size

        self.strategy = 'most-free'
        # cmd = "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
        # output = subprocess.check_output(cmd, shell=True).decode().strip().split("\n")
        gpus = GPUtil.getGPUs()
        output = [int(gpu.memoryFree) for gpu in gpus]

        self.free_memory = []
        for i, free_memory_str in enumerate(output):
            self.free_memory.append(int(free_memory_str))

    def update_free_memory(self):
        cmd = "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode().strip().split("\n")
        self.free_memory = [int(free_memory_str) for free_memory_str in output]

    def choice_gpu(self):

        flag = False

        if self.strategy == 'most-free':

            max_free_memory = max(self.free_memory)
            max_free_memory_idx = self.free_memory.index(max_free_memory)

            if max_free_memory >= self.memory_size:
                
                flag = True
                self.free_memory[max_free_memory_idx] -= self.memory_size

                print(f"GPU-{max_free_memory_idx}: {max_free_memory}MB -> {self.free_memory[max_free_memory_idx]}MB")

                return max_free_memory_idx

        else:

            for i, free_memory in enumerate(self.free_memory):

                if free_memory >= self.memory_size:
                    
                    flag = True
                    self.free_memory[i] -= self.memory_size
                    print(f"GPU-{i}: {free_memory}MB -> {self.free_memory[i]}MB")
                    
                    return i
        
        if not flag:
            print(f"SubProcess[{os.getpid()}]: No GPU can use, switch to CPU!")
            return -1


def set_cpu_num(cpu_num):

    if cpu_num <= 0: 
        return

    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    torch.set_num_interop_threads(1)


def check_gpus():
    '''
    GPU available check
    http://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-cuda/
    '''
    if not torch.cuda.is_available():
        print('This script could only be used to manage NVIDIA GPUs,but no GPU found in your device')
        return False
    elif not 'NVIDIA System Management' in os.popen('nvidia-smi -h').read():
        print("'nvidia-smi' tool not found.")
        return False
    return True

if check_gpus():
    def parse(line,qargs):
        '''
        line:
            a line of text
        qargs:
            query arguments
        return:
            a dict of gpu infos
        Pasing a line of csv format text returned by nvidia-smi
        解析一行nvidia-smi返回的csv格式文本
        '''
        numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']#可计数的参数
        power_manage_enable=lambda v:(not 'Not Support' in v)#lambda表达式，显卡是否滋瓷power management（笔记本可能不滋瓷）
        to_numberic=lambda v:float(v.upper().strip().replace('MIB','').replace('W',''))#带单位字符串去掉单位
        process = lambda k,v:((int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
        return {k:process(k,v) for k,v in zip(qargs,line.strip().split(','))}
    
    def query_gpu(qargs=[]):
        '''
        qargs:
            query arguments
        return:
            a list of dict
        Querying GPUs infos
        查询GPU信息
        '''
        qargs =['index','gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit']+ qargs
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
        results = os.popen(cmd).readlines()
        return [parse(line,qargs) for line in results]
    
    def by_power(d):
        '''
        helper function fo sorting gpus by power
        '''
        power_infos=(d['power.draw'],d['power.limit'])
        if any(v==1 for v in power_infos):
            print('Power management unable for GPU {}'.format(d['index']))
            return 1
        return float(d['power.draw'])/d['power.limit']
    
    class GPUManager():
        '''
        qargs:
            query arguments
        A manager which can list all available GPU devices
        and sort them and choice the most free one.Unspecified 
        ones pref.
        GPU设备管理器，考虑列举出所有可用GPU设备，并加以排序，自动选出
        最空闲的设备。在一个GPUManager对象内会记录每个GPU是否已被指定，
        优先选择未指定的GPU。
        '''
        def __init__(self,qargs=[], exclude=[]):
            '''
            '''
            self.qargs=qargs
            self.exclude = exclude
            self.gpus=query_gpu(qargs)
            # self.gpus = [gpu for gpu in query_gpu(qargs) if int(gpu['index']) not in exclude]
            # print(self.gpus)
            for gpu in self.gpus:
                gpu['specified']=False
            self.gpu_num=len(self.gpus)
    
        def _sort_by_memory(self,gpus,by_size=False):
            if by_size:
                print('Sorted by free memory size')
                return sorted(gpus,key=lambda d:d['memory.free'],reverse=True)
            else:
                print('Sorted by free memory rate')
                return sorted(gpus,key=lambda d:float(d['memory.free'])/ d['memory.total'],reverse=True)
    
        def _sort_by_power(self,gpus):
            return sorted(gpus,key=by_power)
        
        def _sort_by_custom(self,gpus,key,reverse=False,qargs=[]):
            if isinstance(key,str) and (key in qargs):
                return sorted(gpus,key=lambda d:d[key],reverse=reverse)
            if isinstance(key,type(lambda a:a)):
                return sorted(gpus,key=key,reverse=reverse)
            raise ValueError("The argument 'key' must be a function or a key in query args,please read the documention of nvidia-smi")

        def auto_choice(self,mode=0):
            '''
            mode:
                0:(default)sorted by free memory size
            return:
                a TF device object
            Auto choice the freest GPU device,not specified
            ones 
            自动选择最空闲GPU,返回索引
            '''
            for old_infos,new_infos in zip(self.gpus,query_gpu(self.qargs)):
                old_infos.update(new_infos)
            unspecified_gpus=[gpu for gpu in self.gpus if not gpu['specified']] or self.gpus
            
            if mode==0:
                print('Choosing the GPU device has largest free memory...')
                chosen_gpu=self._sort_by_memory(unspecified_gpus,True)[0]
            elif mode==1:
                print('Choosing the GPU device has highest free memory rate...')
                chosen_gpu=self._sort_by_power(unspecified_gpus)[0]
            elif mode==2:
                print('Choosing the GPU device by power...')
                chosen_gpu=self._sort_by_power(unspecified_gpus)[0]
            else:
                print('Given an unaviliable mode,will be chosen by memory')
                chosen_gpu=self._sort_by_memory(unspecified_gpus)[0]
            chosen_gpu['specified']=True
            index=chosen_gpu['index']
            print('Using GPU {i}:\n{info}'.format(i=index,info='\n'.join([str(k)+':'+str(v) for k,v in chosen_gpu.items()])))
            return int(index)
        
        def choose_no_task_gpu(self, use_memory=2000, threshold=0):
            for old_infos,new_infos in zip(self.gpus,query_gpu(self.qargs)):
                old_infos.update(new_infos)

            unspecified_gpus=[gpu for gpu in self.gpus if not gpu['specified']] or self.gpus

            # gpu_av = dict()
            gpu_av = []

            for i in range(len(unspecified_gpus)):
                if unspecified_gpus[i]['memory.free'] - use_memory > threshold and int(unspecified_gpus[i]['index']) not in self.exclude:
                    
                    gpu_av.append(i)
                    
            return gpu_av
else:
    raise ImportError('GPU available check failed')
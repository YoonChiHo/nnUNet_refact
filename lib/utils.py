import subprocess
import os
import numpy as np
import torch
from time import time, sleep
from datetime import datetime
import sys

def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data

def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d

class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

def get_allowed_n_proc_DA():
    hostname = subprocess.getoutput(['hostname'])

    if 'nnUNet_n_proc_DA' in os.environ.keys():
        return int(os.environ['nnUNet_n_proc_DA'])

    if hostname in ['hdf19-gpu16', 'hdf19-gpu17', 'e230-AMDworkstation']:
        return 16

    if hostname in ['Fabian',]:
        return 12

    if hostname.startswith('hdf19-gpu') or hostname.startswith('e071-gpu'):
        return 12
    elif hostname.startswith('e230-dgx1'):
        return 10
    elif hostname.startswith('hdf18-gpu') or hostname.startswith('e132-comp'):
        return 16
    elif hostname.startswith('e230-dgx2'):
        return 6
    elif hostname.startswith('e230-dgxa100-'):
        return 28
    elif hostname.startswith('lsf22-gpu'):
        return 28
    else:
        return None

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

# def load_pickle(file: str, mode: str = 'rb'):
#     with open(file, mode) as f:
#         a = pickle.load(f)
#     return a
# def save_pickle(obj, file: str, mode: str = 'wb') -> None:
#     with open(file, mode) as f:
#         pickle.dump(obj, f)

# def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
#     with open(file, 'w') as f:
#         json.dump(obj, f, sort_keys=sort_keys, indent=indent)
        
# def load_json(file: str):
#     with open(file, 'r') as f:
#         a = json.load(f)
#     return a
      

# def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
#     if join:
#         l = os.path.join
#     else:
#         l = lambda x, y: y
#     res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
#            and (prefix is None or i.startswith(prefix))
#            and (suffix is None or i.endswith(suffix))]
#     if sort:
#         res.sort()
#     return res

def convert_to_npy(args):
    if not isinstance(args, tuple):
        key = "data"
        npz_file = args
    else:
        npz_file, key = args
    if not os.path.isfile(npz_file[:-3] + "npy"):
        a = np.load(npz_file)[key]
        np.save(npz_file[:-3] + "npy", a)

def print_to_log_file(log_file,output_folder, *args, also_print_to_console=True, add_timestamp=True):

    timestamp = time()
    dt_object = datetime.fromtimestamp(timestamp)

    if add_timestamp:
        args = ("%s:" % dt_object, *args)

    if log_file is None:
        #maybe_mkdir_p()
        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.now()
        log_file = os.path.join(output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                timestamp.second))
        with open(log_file, 'w') as f:
            f.write("Starting... \n")
    successful = False
    max_attempts = 5
    ctr = 0
    while not successful and ctr < max_attempts:
        try:
            with open(log_file, 'a+') as f:
                for a in args:
                    f.write(str(a))
                    f.write(" ")
                f.write("\n")
            successful = True
        except IOError:
            print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
            sleep(0.5)
            ctr += 1
    if also_print_to_console:
        print(*args)
    return log_file


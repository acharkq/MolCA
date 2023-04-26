import time
import random
import numpy as np
import torch
import functools


def print_std(accs, stds, categories, append_mean=False):
    category_line = ' '.join(categories)
    if append_mean:
        category_line += ' Mean'

    line = ''
    if stds is None:
        for acc in accs:
            line += '{:0.1f} '.format(acc)
    else:
        for acc, std in zip(accs, stds):
            line += '{:0.1f}±{:0.1f} '.format(acc, std)

    if append_mean:
        line += '{:0.1f}'.format(sum(accs) / len(accs))
    print(category_line)
    print(line)


def simple_mixup(feat, y, mixup_alpha):
    B = feat.shape[0]
    device = feat.device
    list_f, list_y = feat, y
    permutation = torch.randperm(B)
    lam = np.random.beta(mixup_alpha, mixup_alpha, (B, 1))  # shape = [B,1]
    lam = torch.from_numpy(lam).to(device).float()
    f_ = (1-lam) * feat + lam * feat[permutation]
    y_ = (1-lam) * y + lam * y[permutation]
    list_f = torch.cat((list_f, f_), dim=0)  # shape = [B, D]
    list_y = torch.cat((list_y, y_), dim=0)  # shape = [B, C]
    return list_f, list_y


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        assert n > 0
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def timeit(func):
    @functools.wraps(func)
    def timed(*args, **kw):
        ts = time.time()
        print_time_info('Method: %s started!' % (func.__name__), dash_top=True)
        result = func(*args, **kw)
        te = time.time()
        print_time_info('Method: %s cost %.2f sec!' %
                        (func.__name__, te-ts), dash_bot=True)
        return result
    return timed


def print_time_info(string, end='\n', dash_top=False, dash_bot=False, file=None):
    times = str(time.strftime('%Y-%m-%d %H:%M:%S',
                              time.localtime(time.time())))
    string = "[%s] %s" % (times, str(string))
    if dash_top:
        print(len(string) * '-', file=file)
    print(string, end=end, file=file)
    if dash_bot:
        print(len(string) * '-', file=file)


def worker_seed_init(idx, seed):
    torch_seed = torch.initial_seed()
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    seed = idx + seed + torch_seed
    random.seed(seed)
    np.random.seed(seed)

# SEED
def set_seed(seed, device=None):
    if_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    if if_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_log(print_str, log_file, print_=True):
    if print_:
        print_time_info(print_str)
    if log_file is None:
        return
    with open(log_file, 'a') as f:
        f.write('\n')
        f.write(print_str)

import random
import numpy as np
import torch

def fix_seed(seed):
    """
    Fix all the random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

class AverageMeter(object):
    """
    Compute and store the current value and the average value in a momentum-line way.
    """
    def __init__(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
        
    def update(self, val, factor=0.1, n=1): # factor like momentum
        self.val = val
        self.avg = self.val*factor + self.avg*(1-factor)
        self.sum += val * n
        self.count += n
        if self.count == 0:
            self.avg = self.val

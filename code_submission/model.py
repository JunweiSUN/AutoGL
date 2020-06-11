import numpy as np
import pandas as pd
import torch
import time
import random
import os
import signal
os.system('pip install nni')
os.system('pip install seaborn')
os.system('pip install cython')
os.system('pip install sparsesvd')
from utils.eda import AutoEDA
from utils.tools import fix_seed
from explore import Explore
from data_space import DataSpace
from model_space import ModelSpace
from feat_engine import FeatEngine

fix_seed(1234)
def timeout_handler(signum, frame):
    """
    Signal handler
    Inform the main process when time runs out.
    """
    raise Timeout
signal.signal(signal.SIGTSTP, timeout_handler)

class Timeout(Exception):
    """Timeout"""

class Model:
    """
    Main Class for training and predicting.
    """
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def predict(self):
        self.explore.explore_space()
        preds = self.explore.predict()
        return preds

    def train_predict(self, data, time_budget, n_class, schema):
        # start a timer for timing.
        timer_abs_path = os.path.abspath(__file__).replace('/model.py', '/timer.py')
        pid = os.getpid()
        os.system(f'python {timer_abs_path} {time_budget - 1} {pid} &')

        start = time.time()
        self.auto_eda = AutoEDA(n_class)
        info = self.auto_eda.get_info(data)
        print('EDA Finished, Remaining', time_budget + start - time.time())
        self.feat_engine = FeatEngine(info)
        self.feat_engine.fit_transform(data)
        print('Feature Engine Finished, Remaining', time_budget + start - time.time())
        self.data_space = DataSpace(info, data)
        print('Data Space Constructed, Remaining', time_budget + start - time.time())
        self.model_space = ModelSpace(info)
        print('Model Space Constructed, Remaining', time_budget + start - time.time())
        self.explore = Explore(info, self.model_space, self.data_space)

        # start training
        while True:
            if time_budget + start - time.time() <= 0:
                return self.preds
            try:
                self.preds = self.predict()
            except Timeout:
                return self.preds

        return self.preds

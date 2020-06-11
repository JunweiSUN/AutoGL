import os
import numpy as np
import sys
import gc
import torch
from .tools import fix_seed
from torch_geometric.utils import is_undirected
fix_seed(1234)
class AutoEDA(object):
    """
    A tool box for Exploratory Data Analysis (EDA)
    Parameters:
    ----------
    n_class: int
        number of classes
    ----------
    """
    def __init__(self, n_class):
        self.info = {'n_class': n_class}

    def get_info(self, data):
        self.get_feature_info(data['fea_table'])
        self.get_edge_info(data['edge_file'])
        self.set_priori_knowledges()
        self.get_label_weights(data, reweighting=True)
        return self.info

    def get_feature_info(self, df):
        """
        Get information of the original node features: number of nodes, number of features, etc.
        Remove those features which have only one value.
        """
        unique_counts = df.nunique()
        unique_counts = unique_counts[unique_counts == 1]
        df.drop(unique_counts.index, axis=1, inplace=True)

        self.info['num_nodes'] = df.shape[0]
        self.info['num_features'] = df.shape[1] - 1

        print('Number of Nodes:', self.info['num_nodes'])
        print('Number of Original Features:', self.info['num_features'])

    def get_edge_info(self, df):
        """
        Get information of the edges: number of edges, if weighted, if directed, Max / Min weight, etc.
        """
        self.info['num_edges'] = df.shape[0]
        min_weight, max_weight = df['edge_weight'].min(), df['edge_weight'].max()
        if min_weight != max_weight:
            self.info['weighted'] = True
        else:
            self.info['weighted'] = False

        edge_index = df[['src_idx', 'dst_idx']].to_numpy()
        edge_index = sorted(edge_index, key=lambda d: d[0])
        edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)

        self.info['directed'] = not is_undirected(edge_index, num_nodes=self.info['num_nodes'])

        print('Number of Edges:', self.info['num_edges'])
        print('Is Directed Graph:', self.info['directed'])
        print('Is Weighted Graph:',self.info['weighted'])
        print('Max Weight:', max_weight, 'Min Weight:', min_weight)

    def set_priori_knowledges(self):
        """
        Set some hyper parameters to their initial value according to some priori knowledges.
        """
        if self.info['num_features'] == 0:
            if self.info['directed']:
                self.info['dropedge_rate'] = 0.5
                self.info['chosen_models'] = ['ResGCN', 'GraphConvNet', 'GraphSAGE']
                self.info['ensemble_threshold'] = 0.01
            else:
                self.info['dropedge_rate'] = 0
                self.info['chosen_models'] = ['GraphConvNet','GIN','GraphSAGE']
                self.info['ensemble_threshold'] = 0.01

        else:
            if self.info['directed']:
                self.info['dropedge_rate'] = 0.5
                self.info['chosen_models'] = ['GraphConvNet','GraphSAGE','ResGCN']
                self.info['ensemble_threshold'] = 0.02
            else:
                if self.info['num_edges'] / self.info['num_nodes']>= 10:
                    self.info['dropedge_rate'] = 0.5
                    self.info['chosen_models'] = ['ARMA','GraphSAGE', 'IncepGCN']
                    self.info['ensemble_threshold'] = 0.02
                else:
                    self.info['dropedge_rate'] = 0.5
                    self.info['chosen_models'] = ['ARMA','IncepGCN','GraphConvNet','SG']
                    self.info['ensemble_threshold'] = 0.03

        if  self.info['num_edges'] / self.info['num_nodes'] >= 200:
            self.info['num_layers'] = 1
            self.info['init_hidden_size'] = 5
        elif self.info['num_edges'] / self.info['num_nodes'] >= 100:
            self.info['num_layers'] = 2
            self.info['init_hidden_size'] = 5
        else:
            self.info['num_layers'] = 2
            self.info['init_hidden_size'] = 7

        if self.info['num_edges'] / self.info['num_nodes'] >= 10:
            self.info['use_linear'] = True
            self.info['dropout_rate'] = 0.2
        else:
            self.info['use_linear'] = False
            self.info['dropout_rate'] = 0.5 

        self.info['lr'] = 0.005

        if self.info['num_features'] == 0:
            self.info['feature_type'] = ['svd']  # one_hot / svd / degree / node2vec / adj
        else:
            self.info['feature_type'] = ['original', 'svd']

        self.info['normalize_features'] = 'None'

    def get_label_weights(self, data, reweighting=True):
        """
        Compute the weights of labels as the weight when computing loss.
        """
        if not reweighting:
            self.info['label_weights'] = None
            return

        groupby_data_orginal = data['train_label'].groupby('label').count()
        label_weights = groupby_data_orginal.iloc[:,0]
        
        if len(label_weights) < 10 or max(label_weights) < min(label_weights) * 10:
            self.info['label_weights'] = None
            return

        label_weights = 1 / np.sqrt(label_weights)
        self.info['label_weights'] = torch.tensor(label_weights.values,dtype=torch.float32)
        print('Label Weights:', self.info['label_weights'])






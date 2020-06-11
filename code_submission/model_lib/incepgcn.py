import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score
from utils.tools import fix_seed, AverageMeter
from nni.hyperopt_tuner.hyperopt_tuner import HyperoptTuner
from torch_geometric.utils import dropout_adj
import copy
fix_seed(1234)

class GraphBaseBlock(torch.nn.Module):
    """
    The base block for Multi-layer GCN / ResGCN / Dense GCN 
    """

    def __init__(self, in_features, out_features, nbaselayer,
                 withbn=True, withloop=True, activation=F.relu, dropout=0.5,
                 aggrmethod="concat", dense=False):
        """
        The base block for constructing DeepGCN model.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param dense: enable dense connection
        """
        super(GraphBaseBlock, self).__init__()
        self.in_features = in_features
        self.hiddendim = out_features
        self.nhiddenlayer = nbaselayer
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dense = dense
        self.dropout = dropout

        self.hiddenlayers = nn.ModuleList()
        self.__makehidden()

        if self.aggrmethod == "concat" and dense == False:
            self.out_features = in_features + out_features
        elif self.aggrmethod == "concat" and dense == True:
            self.out_features = in_features + out_features * nbaselayer
        elif self.aggrmethod == "add":
            if in_features != self.hiddendim:
                raise RuntimeError("The dimension of in_features and hiddendim should be matched in add model.")
            self.out_features = out_features
        elif self.aggrmethod == "nores":
            self.out_features = out_features
        else:
            raise NotImplementedError("The aggregation method only support 'concat','add' and 'nores'.")

    def __makehidden(self):
        for i in range(self.nhiddenlayer):
            if i == 0:
                layer = GCNConv(self.in_features, self.hiddendim)
            else:
                layer = GCNConv(self.hiddendim, self.hiddendim)
            self.hiddenlayers.append(layer)

    def _doconcat(self, x, subx):
        if x is None:
            return subx
        if self.aggrmethod == "concat":
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == "add":
            return x + subx
        elif self.aggrmethod == "nores":
            return x

    def forward(self, input, edge_index, edge_weight):
        x = input
        denseout = None
        # Here out is the result in all levels.
        for gc in self.hiddenlayers:
            denseout = self._doconcat(denseout, x)
            x = self.activation(gc(x, edge_index, edge_weight))
            x = F.dropout(x, self.dropout, training=self.training)

        if not self.dense:
            return self._doconcat(x, input)
        return self._doconcat(x, denseout)

    def get_outdim(self):
        return self.out_features

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (self.__class__.__name__,
                                              self.aggrmethod,
                                              self.in_features,
                                              self.hiddendim,
                                              self.nhiddenlayer,
                                              self.out_features)
                                              
class InceptionGCNBlock(torch.nn.Module):
    """
    The multiple layer GCN with inception connection block.
    """

    def __init__(self, in_features, out_features, nbaselayer,
                 dropout=0.5, aggrmethod="concat", dense=False):
        """
        The multiple layer GCN with inception connection block.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param dense: not applied. The default is False, cannot be changed.
        """
        super(InceptionGCNBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hiddendim = out_features
        self.nbaselayer = nbaselayer
        self.aggrmethod = aggrmethod
        self.dropout = dropout
        self.midlayers = torch.nn.ModuleList()
        self.__makehidden()

        if self.aggrmethod == "concat":
            self.out_features = in_features + out_features * nbaselayer
        elif self.aggrmethod == "add":
            if in_features != self.hiddendim:
                raise RuntimeError("The dimension of in_features and hiddendim should be matched in 'add' model.")
            self.out_features = out_features
        else:
            raise NotImplementedError("The aggregation method only support 'concat', 'add'.")

    def __makehidden(self):
        for j in range(self.nbaselayer):
            reslayer = torch.nn.ModuleList()
            for i in range(j + 1):
                if i == 0:
                    layer = GCNConv(self.in_features, self.hiddendim)
                else:
                    layer = GCNConv(self.hiddendim, self.hiddendim)
                reslayer.append(layer)
            self.midlayers.append(reslayer)

    def forward(self, input, edge_index, edge_weight):
        x = input
        for reslayer in self.midlayers:
            subx = input
            for gc in reslayer:
                subx = gc(subx, edge_index, edge_weight)
                subx = F.dropout(subx,  p=self.dropout, training=self.training)
            x = self._doconcat(x, subx)
        return x

    def get_outdim(self):
        return self.out_features

    def _doconcat(self, x, subx):
        if self.aggrmethod == "concat":
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == "add":
            return x + subx

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (self.__class__.__name__,
                                              self.aggrmethod,
                                              self.in_features,
                                              self.hiddendim,
                                              self.nbaselayer,
                                              self.out_features)

class IncepGCN(torch.nn.Module):

    def __init__(self, info):
        super(IncepGCN, self).__init__()
        self.info = info
        self.best_score = 0
        self.hist_score = []

        self.best_preds = None
        self.current_round_best_preds = None
        self.best_valid_score = 0
        self.max_patience = 100
        self.max_epochs = 1600
        
        self.name = 'IncepGCN'

        self.tuner = HyperoptTuner(algorithm_name='tpe', optimize_mode='maximize')
        search_space = {
                "dropedge_rate": {
                    "_type": "choice",
                    "_value": [self.info['dropedge_rate']]
                },
                "dropout_rate": {
                    "_type": "choice",
                    "_value": [self.info['dropout_rate']]
                },
                "num_layers": {
                    "_type": "quniform",
                    "_value": [2, 4, 1]
                },
                "hidden": {
                    "_type": "quniform",
                    "_value": [4, 7, 1]
                },
                "lr":{
                    "_type": "choice",
                    "_value": [0.005]
                }
            }
        self.tuner.update_search_space(search_space)
        self.hyperparameters = {
            'num_layers': self.info['num_layers'],
            'lr': self.info['lr'],
            'dropedge_rate':self.info['dropedge_rate'],
            'dropout_rate':self.info['dropout_rate'],
            'hidden': self.info['init_hidden_size']
        }
        self.best_hp = None

    def init_model(self, n_class, features_num):
        hidden = int(2 ** self.hyperparameters['hidden'])
        num_layers = int(self.hyperparameters['num_layers'])
        self.in_lin = nn.Linear(features_num, hidden)
        self.incep_conv = InceptionGCNBlock(hidden, hidden, nbaselayer=num_layers, dropout=self.hyperparameters['dropout_rate'])
        self.out_lin = nn.Linear(self.incep_conv.get_outdim(), n_class)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hyperparameters['lr'], weight_decay=5e-4)

        self = self.to('cuda')

        torch.cuda.empty_cache()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        if self.hyperparameters['dropedge_rate'] is not None:
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=self.hyperparameters['dropedge_rate'],\
                 force_undirected=False, num_nodes=None, training=self.training)
        x = self.in_lin(x)
        x = F.dropout(x, p=self.hyperparameters['dropout_rate'], training=self.training)
        x = self.incep_conv(x, edge_index, edge_weight)
        x = self.out_lin(x)
        return x

    def trial(self, data, round_num):
        n_class, feature_num = self.info['n_class'], data.x.shape[1]
        if round_num >= 2:
            self.hyperparameters = self.tuner.generate_parameters(round_num-1)
        print(self.hyperparameters)    
           
        while True:
            try:
                self.init_model(n_class, feature_num)
                val_score = self.train_valid(data, round_num)
                if round_num > 1:
                    self.tuner.receive_trial_result(round_num-1,self.hyperparameters,val_score)
                if val_score > self.best_score:
                    self.best_hp = copy.deepcopy(self.hyperparameters)
                break
            except RuntimeError as e:
                print(self.name,e, 'OOM with Hidden Size', self.hyperparameters['hidden'])
                if round_num > 1:
                    self.tuner.receive_trial_result(round_num-1,self.hyperparameters,0)
                return 0
        print("Best Hyperparameters of", self.name, self.best_hp)
        return val_score


    def train_valid(self, data, round_num):
        y, train_mask, valid_mask, test_mask, label_weights = data.y, data.train_mask, data.valid_mask, data.test_mask, data.label_weights

        score_meter = AverageMeter()
        patience = self.max_patience
        best_valid_score = 0
        for epoch in range(self.max_epochs):

            # train
            self.train()
            self.optimizer.zero_grad()
            preds = self.forward(data)
            loss = F.cross_entropy(preds[train_mask], y[train_mask], label_weights)
            loss.backward()
            self.optimizer.step()

            # valid
            self.eval()
            with torch.no_grad():
                preds = F.softmax(self.forward(data), dim=-1)
                valid_preds, test_preds = preds[valid_mask], preds[test_mask]
                valid_score = f1_score(y[valid_mask].cpu(), valid_preds.max(1)[1].flatten().cpu(), average='micro')

            score_meter.update(valid_score)

            # patience
            if score_meter.avg > best_valid_score:
                best_valid_score = score_meter.avg
                self.current_round_best_preds = test_preds
                patience = self.max_patience
            else:
                patience -= 1

            if patience == 0:
                break

        return best_valid_score

    def predict(self):
        if self.current_round_best_preds is not None:
            return self.current_round_best_preds.cpu().numpy()
        else:
            return None

    def __repr__(self):
        return self.__class__.__name__
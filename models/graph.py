import torch
import torch.nn as nn
import dgllife
import dgl

class Structure2VecFirstLayer(nn.Module):

    def __init__(self, num_hidden_features, num_atom_features, num_bond_features, dropout=0):
        super(Structure2VecFirstLayer, self).__init__()
        self.atom_layer = nn.Linear(num_atom_features, num_hidden_features)
        self.bond_layer = nn.Linear(num_bond_features, num_hidden_features)
        self.activation = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(num_hidden_features)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, g):
        g.edata['h'] = self.bond_layer(g.edata['w'])
        g.send(g.edges(), lambda edges: { 'msg': edges.data['h'] })
        g.recv(g.nodes(), lambda nodes: { 'h': torch.sum(nodes.mailbox['msg'], dim=1) })
        h = g.ndata.pop('h') + self.atom_layer(g.ndata['x'])
        h = self.activation(self.bn(h))
        return self.dropout(h)


class Structure2VecLayer(nn.Module):

    def __init__(self, num_hidden_features, num_atom_features, num_bond_features, dropout=0):
        super(Structure2VecLayer, self).__init__()
        self.bond_layer = nn.Linear(num_bond_features, num_hidden_features)
        self.hidden_layer1 = nn.Linear(num_hidden_features, num_hidden_features)
        self.hidden_layer2 = nn.Linear(num_hidden_features, num_hidden_features)
        self.activation = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(num_hidden_features)
        self.bn2 = nn.BatchNorm1d(num_hidden_features)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, g, features):
        g.edata['h'] = self.bond_layer(g.edata['w'])
        g.ndata['h'] = features
        g.send(g.edges(), lambda edges: { 'msg_n': edges.src['h'], 
                                          'msg_e': edges.data['h'] })
        g.recv(g.nodes(), lambda nodes: { 'h1': torch.sum(nodes.mailbox['msg_n'], dim=1),
                                          'h2': torch.sum(nodes.mailbox['msg_e'], dim=1) })
        h1, h2 = g.ndata.pop('h1'), g.ndata.pop('h2')
        h = self.activation(self.bn1(self.hidden_layer1(h1) + h2))
        h = self.activation(self.bn2(self.hidden_layer2(h) + features))
        g.ndata.pop('h')
        g.edata.pop('h')
        return self.dropout(h)


class Structure2Vec(nn.Module):
    
    def __init__(self, num_layers, num_hidden_features, num_atom_features, num_bond_features, dropout=0):
        super(Structure2Vec, self).__init__()
        self.num_hidden_features = num_hidden_features
        self.first_layer = Structure2VecFirstLayer(num_hidden_features,
                                                   num_atom_features,
                                                   num_bond_features,
                                                   dropout=dropout)
        self.layers = nn.ModuleList([Structure2VecLayer(num_hidden_features,
                                                        num_atom_features,
                                                        num_bond_features,
                                                        dropout=dropout) for _ in range(num_layers)])
        self.last_layer = nn.Sequential(nn.Linear(num_hidden_features, num_hidden_features),
                                        nn.BatchNorm1d(num_hidden_features))

    def forward(self, g, device=None):
        features = self.first_layer(g)
        for layer in self.layers:
            features = layer(g, features)
        features = self.last_layer(features)
        return features


import torch
import torch.nn as nn
import dgllife
import dgl

class Structure2VecFirstLayer(nn.Module):

    def __init__(self, num_hidden_features, num_atom_features, num_bond_features, bn_first=False):
        super(Structure2VecFirstLayer, self).__init__()
        self.atom_layer = nn.Linear(num_atom_features, num_hidden_features)
        self.bond_layer = nn.Linear(num_bond_features, num_hidden_features)
        self.activation = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(num_hidden_features)
        self.bn_first = bn_first

    def forward(self, g):
        g.edata['h'] = self.bond_layer(g.edata['w'])
        g.send(g.edges(), lambda edges: { 'msg': edges.data['h'] })
        g.recv(g.nodes(), lambda nodes: { 'h': torch.sum(nodes.mailbox['msg'], dim=1) })
        h = g.ndata.pop('h') + self.atom_layer(g.ndata['x'])
        if not self.bn_first:
            h = self.bn(self.activation(h))
        else:
            h = self.activation(self.bn(h))
        return h


class Structure2VecLayer(nn.Module):

    def __init__(self, num_hidden_features, num_atom_features, num_bond_features, bn_first=False):
        super(Structure2VecLayer, self).__init__()
        self.bond_layer = nn.Linear(num_bond_features, num_hidden_features)
        self.hidden_layer1 = nn.Linear(num_hidden_features, num_hidden_features)
        self.hidden_layer2 = nn.Linear(num_hidden_features, num_hidden_features)
        self.activation = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(num_hidden_features)
        self.bn2 = nn.BatchNorm1d(num_hidden_features)
        self.bn_first = bn_first

    def forward(self, g, features):
        g.edata['h'] = self.bond_layer(g.edata['w'])
        g.ndata['h'] = features
        g.send(g.edges(), lambda edges: { 'msg_n': edges.src['h'], 
                                          'msg_e': edges.data['h'] })
        g.recv(g.nodes(), lambda nodes: { 'h1': torch.sum(nodes.mailbox['msg_n'], dim=1),
                                          'h2': torch.sum(nodes.mailbox['msg_e'], dim=1) })
        h1, h2 = g.ndata.pop('h1'), g.ndata.pop('h2')
        if not self.bn_first:
            h = self.bn1(self.activation(self.hidden_layer1(h1) + h2))
            h = self.bn2(self.activation(self.hidden_layer2(h) + features))
        else:
            h = self.activation(self.bn1(self.hidden_layer1(h1) + h2))
            h = self.activation(self.bn2(self.hidden_layer2(h) + features))
        g.ndata.pop('h')
        g.edata.pop('h')
        return h


class Structure2Vec(nn.Module):
    
    def __init__(self, num_layers, num_hidden_features, num_atom_features, num_bond_features, bn_first=False):
        super(Structure2Vec, self).__init__()
        self.first_layer = Structure2VecFirstLayer(num_hidden_features,
                                                   num_atom_features,
                                                   num_bond_features,
                                                   bn_first=bn_first)
        self.layers = nn.ModuleList([Structure2VecLayer(num_hidden_features,
                                                        num_atom_features,
                                                        num_bond_features,
                                                        bn_first=bn_first) for _ in range(num_layers)])
        self.bn_first = bn_first
        if bn_first:
            self.last_layer = nn.Sequential(nn.Linear(num_hidden_features, num_hidden_features),
                                            nn.BatchNorm1d(num_hidden_features))

    def forward(self, g, device=None):
        features = self.first_layer(g)
        for layer in self.layers:
            features = layer(g, features)
        if self.bn_first:
            features = self.last_layer(features)
        return features


class AttentionFP(dgllife.model.gnn.AttentiveFPGNN):

    def forward(self, g):
        node_feats = g.ndata['x']
        edge_feats = g.edata['w']
        return super(AttentionFP, self).forward(g, node_feats, edge_feats)

class MPNN(dgllife.model.gnn.MPNNGNN):

    def forward(self, g):
        node_feats = g.ndata['x']
        edge_feats = g.edata['w']
        return super(MPNN, self).forward(g, node_feats, edge_feats)

class WLN(dgllife.model.gnn.WLN):

    def forward(self, g):
        node_feats = g.ndata['x']
        edge_feats = g.edata['w']
        return super(WLN, self).forward(g, node_feats, edge_feats)

class EdgewiseGNNLayer(nn.Module):

    def __init__(self, num_hidden_features, num_edge_types):
        super(EdgewiseGNNLayer, self).__init__()
        self.num_edge_types = num_edge_types
        self.num_hidden_features = num_hidden_features
        self.linear1 = nn.Linear(num_hidden_features, num_hidden_features*num_edge_types)
        self.linear2 = nn.Linear(num_hidden_features, num_hidden_features*num_edge_types)
        self.bn1 = nn.BatchNorm1d(num_hidden_features)
        self.bn2 = nn.BatchNorm1d(num_hidden_features)
        self.activation = nn.ReLU(inplace=True)

    def msg_fn(self, edges):
        msg = edges.src['h'].view(-1, self.num_edge_types, self.num_hidden_features)
        msg = msg.mul(edges.data['type'].unsqueeze(-1)).sum(1)
        return { 'msg': msg }

    def reduce_fn(self, nodes):
        return { 'h_new': nodes.mailbox['msg'].sum(1) }

    def forward(self, g, features):
        g.ndata['h'] = self.linear1(self.activation(self.bn1(features)))
        g.send(g.edges(), self.msg_fn)
        g.recv(g.nodes(), self.reduce_fn)
        h_new, _ = g.ndata.pop('h_new'), g.ndata.pop('h')

        g.ndata['h'] = self.linear2(self.activation(self.bn2(h_new)))
        g.send(g.edges(), self.msg_fn)
        g.recv(g.nodes(), self.reduce_fn)
        h_new, _ = g.ndata.pop('h_new'), g.ndata.pop('h')

        return h_new + features

class EdgewiseGNN(nn.Module):

    def __init__(self, num_layers, num_hidden_features, num_atom_features, num_bond_types):
        super(EdgewiseGNN, self).__init__()
        self.num_bond_types = num_bond_types
        self.first_layer = nn.Linear(num_atom_features, num_hidden_features)
        self.layers = nn.ModuleList([EdgewiseGNNLayer(num_hidden_features,
                                                      num_bond_types) for _ in range(num_layers)])
        self.activation = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(num_hidden_features)

    def forward(self, g):
        g.edata['type'] = g.edata['w'][:, :self.num_bond_types]
        features = self.first_layer(g.ndata['x'])
        for layer in self.layers:
            features = layer(g, features)
        return self.bn(self.activation(features))


class Structure2VecOursLayer(nn.Module):

    def __init__(self, num_hidden_features, num_atom_features, num_bond_features):
        super(Structure2VecOursLayer, self).__init__()
        self.linear1 = nn.Linear(num_hidden_features+num_atom_features+num_bond_features,
                                 num_hidden_features)
        self.linear2 = nn.Linear(num_hidden_features, num_hidden_features)
        self.activation = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(num_hidden_features)
        self.bn2 = nn.BatchNorm1d(num_hidden_features)

    def msg_fn(self, edges):
        x = torch.cat([edges.src['h'], edges.src['x'], edges.data['w']], 1)
        x = self.activation(self.bn1(self.linear1(x)) + edges.src['h'])
        return { 'msg': x }

    def reduce_fn(self, nodes):
        return { 'h_new': nodes.mailbox['msg'].sum(dim=1) }

    def forward(self, g, features):
        g.ndata['h'] = features
        g.send(g.edges(), self.msg_fn)
        g.recv(g.nodes(), self.reduce_fn)
        _, h_new = g.ndata.pop('h'), g.ndata.pop('h_new')
        return self.activation(self.bn2(self.linear2(h_new))+features)


class Structure2VecOurs(nn.Module):

    def __init__(self, num_layers, num_hidden_features, num_atom_features, num_bond_features):
        super(Structure2VecOurs, self).__init__()
        self.first_layer = nn.Sequential(nn.Linear(num_atom_features, num_hidden_features),
                                         nn.BatchNorm1d(num_hidden_features),
                                         nn.ReLU(inplace=True))
        self.layers = nn.ModuleList([Structure2VecOursLayer(num_hidden_features,
                                                            num_atom_features,
                                                            num_bond_features) for _ in range(num_layers)])
        self.last_layer = nn.Sequential(nn.Linear(num_hidden_features, num_hidden_features),
                                        nn.BatchNorm1d(num_hidden_features))

    def forward(self, g, device=None):
        features = self.first_layer(g.ndata['x'])
        for layer in self.layers:
            features = layer(g, features)
        return self.last_layer(features)


class RelGraphNN(nn.Module):

    def __init__(self, num_layers, num_hidden_features, num_atom_features, num_bond_types):
        super(RelGraphNN, self).__init__()
        self.num_bond_types = num_bond_types
        layers = [dgl.nn.pytorch.conv.RelGraphConv(
            num_atom_features,
            num_hidden_features,
            num_bond_types,
            activation=nn.Sequential(nn.BatchNorm1d(num_hidden_features),
                                     nn.ReLU(inplace=True)),
            self_loop=True)]
        for i in range(num_layers):
            layers.append(dgl.nn.pytorch.conv.RelGraphConv(
                num_hidden_features,
                num_hidden_features,
                num_bond_types,
                activation=nn.Sequential(nn.BatchNorm1d(num_hidden_features),
                                         nn.ReLU(inplace=True)),
                self_loop=True))
        self.layers = nn.ModuleList(layers)
        self.last_layer = nn.Sequential(nn.Linear(num_hidden_features, num_hidden_features),
                                        nn.BatchNorm1d(num_hidden_features))

    def forward(self, g):
        x = g.ndata['x']
        etypes = g.edata['w'][:, :self.num_bond_types].argmax(1)
        for layer in self.layers:
            x = layer(g, x, etypes)
        return self.last_layer(x)

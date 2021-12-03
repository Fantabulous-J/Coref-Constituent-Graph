from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class CGATLayer(nn.Module, ABC):
    """ Constituent-Constituent GATLayer """

    def __init__(self, in_dim, feat_embed_size, out_dim, num_heads):
        super(CGATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.num_heads = num_heads
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=2)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, edge_type=None):
        z = self.fc(h)
        num_tokens, emb_size = z.size()
        z = z.reshape([num_tokens, self.num_heads, emb_size // self.num_heads])
        cons_node_ids = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)
        cc_edge_id = g.filter_edges(lambda edges: edges.data["dtype"] == edge_type)
        self_edge_id = g.filter_edges(lambda edges: edges.data["dtype"] == 4)
        cc_edge_id = torch.cat([cc_edge_id, self_edge_id], dim=0)
        g.nodes[cons_node_ids].data['z'] = z
        g.apply_edges(self.edge_attention, edges=cc_edge_id)
        g.pull(cons_node_ids, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('h')
        return h[cons_node_ids]


class CTGATLayer(nn.Module, ABC):
    """ Constituent-Token GATLayer """

    def __init__(self, in_dim, feat_embed_size, out_dim, num_heads):
        super(CTGATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.num_heads = num_heads
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=2)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, edge_type=None):
        z = self.fc(h)
        num_tokens, emb_size = z.size()
        z = z.reshape([num_tokens, self.num_heads, emb_size // self.num_heads])
        token_node_ids = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 0)
        cons_node_ids = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)
        ct_edge_id = g.filter_edges(lambda edges: edges.data["dtype"] == 5)
        g.nodes[cons_node_ids].data['z'] = z
        g.apply_edges(self.edge_attention, edges=ct_edge_id)
        g.pull(token_node_ids, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('h')
        return h[token_node_ids]


class MultiHeadGATLayer(nn.Module, ABC):
    def __init__(self, layer, in_size, out_size, feat_embed_size, num_heads, config, merge='cat', layer_norm_eps=1e-12):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        out_dim = out_size // num_heads
        self.layer = layer(in_size, feat_embed_size, out_dim, num_heads)
        self.merge = merge
        self.dropout = nn.Dropout(p=config['gat_dropout_rate'])
        self.LayerNorm = nn.LayerNorm(out_size, eps=layer_norm_eps)

    def forward(self, g, o, h, edge_type=None):
        head_outs = self.layer(g, self.dropout(h), edge_type)
        num_tokens = head_outs.size()[0]
        if self.merge == 'cat':
            out = head_outs.reshape([num_tokens, -1])
        else:
            out = torch.mean(head_outs, dim=1)
        out = o + F.elu(out)
        out = self.LayerNorm(out)
        return out

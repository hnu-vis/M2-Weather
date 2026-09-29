from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        # F.linear maps the node axis with the same y[v] = sum_w A[v,w]x[w]
        # contraction as the original einsum, while dispatching a GEMM-friendly
        # kernel for large station graphs.
        return F.linear(x.transpose(2, 3), A).transpose(2, 3).contiguous()


class nconv1(nn.Module):
    def __init__(self):
        super(nconv1,self).__init__()

    def forward(self,x, A):
        x = torch.einsum("ncwl,nvw->ncvl",(x,A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class mixprop(nn.Module):
    '''
    this a rnn architecture, can modified by GRU
    '''
    def __init__(self,c_in,c_out,gdep,dropout,alpha,predA=True):
        super(mixprop, self).__init__()
        if not predA:
            self.nconv = nconv1()
        else:
            self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        # self.alpha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device) # A+I
        d = adj.sum(1) # sum of row axis. i.e. degree -> vector not matrix
        h = x
        out = [h]
        a = adj / d.view(-1, 1) # d^-1(a+i)
        for i in range(self.gdep):
            # beta*X + (1-beta)\hat{A}*H
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho


class mixprop_gat(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha, leak_alpha, nheads, num_nodes):
        super(mixprop_gat, self).__init__()
        self.gat = GAT(c_in, c_out, dropout, leak_alpha, nheads, num_nodes)
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)  # A+I
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.gat(h, adj)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class GraphAttentionLayer(nn.Module):
    """Graph attention over the node axis of [B, C, N, T] tensors."""

    def __init__(self, in_features, out_features, nnodes, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.nnodes = nnodes
        self.dropout = dropout
        self.concat = concat
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.attn_src = nn.Parameter(torch.empty(out_features))
        self.attn_dst = nn.Parameter(torch.empty(out_features))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=1.414)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0), gain=1.414)
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0), gain=1.414)

    def forward(self, x, adj):
        if x.ndim != 4:
            raise ValueError(f"expected x with shape [B, C, N, T], got {tuple(x.shape)}")
        if x.size(2) != self.nnodes:
            raise ValueError(f"expected {self.nnodes} nodes, got {x.size(2)}")
        if adj.shape != (self.nnodes, self.nnodes):
            raise ValueError(
                f"expected adjacency shape {(self.nnodes, self.nnodes)}, got {tuple(adj.shape)}"
            )

        features = torch.einsum("bcnt,co->bont", x, self.weight)
        src_scores = torch.einsum("bont,o->bnt", features, self.attn_src)
        dst_scores = torch.einsum("bont,o->bnt", features, self.attn_dst)
        scores = self.leaky_relu(src_scores.unsqueeze(2) + dst_scores.unsqueeze(1))

        edge_mask = adj.to(device=x.device) > 0
        scores = scores.masked_fill(
            ~edge_mask.unsqueeze(0).unsqueeze(-1),
            torch.finfo(scores.dtype).min,
        )
        attention = F.softmax(scores, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        output = torch.einsum("bijt,bojt->boit", attention, features)
        return F.elu(output) if self.concat else output


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads, nnodes):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, nnodes, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.mlp = linear(nheads*nhid, nhid)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1) # 注意力机制层 只有一层
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)
        return x


# class HeterGRaphConv(nn.Module):
#     def __init__(self, mods, aggregate='sum'):
#         super(HeterGRaphConv, self).__init__()
#         self.mods = nn.ModuleDict(mods)
#         if isinstance(aggregate, str):
#             self.agg_fn = get_aggregate_fn(aggregate)
#         else:
#             self.agg_fn = aggregate

class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        # self.kernel_set = [2, 3, 6, 7]
        self.kernel_set = [3, 6]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1, dilation_factor)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class dilated_inception_same(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception_same, self).__init__()
        self.tconv = nn.ModuleList()
        # self.kernel_set = [2, 3, 6, 7]
        self.kernel_set = [3, 6]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1, dilation_factor), padding=(0, 3)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-input.size(3):]
        x = torch.cat(x, dim=1)
        return x


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

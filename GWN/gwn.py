import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter


def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum('ncvl,vw->ncwl', (x, A)).contiguous()


class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.final_conv = Conv2d(c_in, c_out, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.final_conv(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GWNet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, do_graph_conv=True,
                 addaptadj=True, aptinit=None, in_dim=2, out_seq_len=12,
                 residual_channels=32, dilation_channels=32, cat_feat_gc=False,
                 skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2, stride=2,
                 apt_size=10, verbose=0):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.do_graph_conv = do_graph_conv
        self.cat_feat_gc = cat_feat_gc
        self.addaptadj = addaptadj
        self.verbose = verbose

        if self.cat_feat_gc:
            self.start_conv = nn.Conv2d(in_channels=1,  # hard code to avoid errors
                                        out_channels=residual_channels,
                                        kernel_size=(1, 1))
            self.cat_feature_conv = nn.Conv2d(in_channels=in_dim - 1,
                                              out_channels=residual_channels,
                                              kernel_size=(1, 1))
        else:
            self.start_conv = nn.Conv2d(in_channels=in_dim,
                                        out_channels=residual_channels,
                                        kernel_size=(1, 1))

        self.fixed_supports = supports or []
        receptive_field = 1

        self.supports_len = len(self.fixed_supports)
        if do_graph_conv and addaptadj:
            if aptinit is None:
                nodevecs = torch.randn(num_nodes, apt_size), torch.randn(apt_size, num_nodes)
            else:
                nodevecs = self.svd_init(apt_size, aptinit)
            self.supports_len += 1
            self.nodevec1, self.nodevec2 = [Parameter(n.to(device), requires_grad=True) for n in nodevecs]

        depth = list(range(blocks * layers))

        # 1x1 convolution for residual and skip connections (slightly different see docstring)
        self.residual_convs = ModuleList([Conv1d(dilation_channels, residual_channels, (1, 1)) for _ in depth])
        self.skip_convs = ModuleList([Conv1d(dilation_channels, skip_channels, (1, 1)) for _ in depth])
        self.bn = ModuleList([BatchNorm2d(residual_channels) for _ in depth])
        self.graph_convs = ModuleList(
            [GraphConvNet(dilation_channels, residual_channels, dropout, support_len=self.supports_len)
             for _ in depth])

        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()
        for b in range(blocks):
            additional_scope = kernel_size - 1
            D = 1  # dilation
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                self.gate_convs.append(Conv1d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                D *= stride
                receptive_field += additional_scope
                additional_scope *= stride
        self.receptive_field = receptive_field

        self.end_conv_1 = Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_2 = Conv2d(end_channels, out_seq_len, (1, 1), bias=True)

    @staticmethod
    def svd_init(apt_size, aptinit):
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    @classmethod
    def from_args(cls, supports, aptinit, **kwargs):

        dropout = kwargs['dropout']
        do_graph_conv = kwargs['do_graph_conv']
        addaptadj = kwargs['addaptadj']
        in_dim = kwargs['in_dim']
        apt_size = kwargs['apt_size']
        out_seq_len = kwargs['out_seq_len']
        hidden = kwargs['hidden']
        kernel_size = kwargs['kernel_size']
        stride = kwargs['stride']
        blocks = kwargs['blocks']
        layers = kwargs['layers']
        cat_feat_gc = kwargs['cat_feat_gc']
        verbose = kwargs['verbose']
        device = kwargs['device']
        num_nodes = kwargs['num_nodes']

        defaults = dict(dropout=dropout, supports=supports,
                        do_graph_conv=do_graph_conv, addaptadj=addaptadj, aptinit=aptinit,
                        in_dim=in_dim, apt_size=apt_size, out_seq_len=out_seq_len,
                        residual_channels=hidden, dilation_channels=hidden,
                        stride=stride, kernel_size=kernel_size,
                        blocks=blocks, layers=layers,
                        skip_channels=hidden * 8, end_channels=hidden * 16,
                        cat_feat_gc=cat_feat_gc, verbose=verbose, device=device, num_nodes=num_nodes)
        # defaults.update(**kwargs)
        model = cls(**defaults)
        return model

    def load_checkpoint(self, state_dict):
        """It is assumed that ckpt was trained to predict a subset of timesteps."""
        bk, wk = ['end_conv_2.bias', 'end_conv_2.weight']  # only weights that depend on seq_length
        b, w = state_dict.pop(bk), state_dict.pop(wk)
        self.load_state_dict(state_dict, strict=False)
        cur_state_dict = self.state_dict()
        cur_state_dict[bk][:b.shape[0]] = b
        cur_state_dict[wk][:w.shape[0]] = w
        self.load_state_dict(cur_state_dict)

    def forward(self, x):

        # input x (b, seq_x, n, features)
        x = x.transpose(1, 3)

        if self.verbose:
            print('-------------------GWN model----------------------')
            print('Input shape: ', x.shape)
        # x (bs, features, n_nodes, n_timesteps)
        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - in_len, 0, 0, 0))

        # first linear layer
        if self.cat_feat_gc:
            f1, f2 = x[:, [0]], x[:, 1:]
            x1 = self.start_conv(f1)
            x2 = F.leaky_relu(self.cat_feature_conv(f2))
            x = x1 + x2
        else:
            x = self.start_conv(x)

        if self.verbose:
            print('After first linear: ', x.shape)

        skip = 0
        adjacency_matrices = self.fixed_supports
        # calculate the current adaptive adj matrix once per iteration
        if self.addaptadj:  # equation (6) and (7)
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)  # the learnable adj matrix
            adjacency_matrices = self.fixed_supports + [adp]
            if self.verbose:
                for adj in adjacency_matrices:
                    print('adj shape', adj.shape)

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            if self.verbose:
                print('Layer: ', i)
                print('   Input layer: ', x.shape)

            residual = x
            # dilated convolution
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter * gate

            if self.verbose:
                print('   filter shape: ', filter.shape)
                print('   gate shape: ', gate.shape)
                print('   Gated tcn output: ', x.shape)

            # parametrized skip connection
            s = self.skip_convs[i](x)  # what are we skipping??
            try:  # if i > 0 this works
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            if self.verbose:
                print('   Skip shape: ', skip.shape)

            if i == (self.blocks * self.layers - 1):  # last X getting ignored anyway
                break

            if self.do_graph_conv:
                graph_out = self.graph_convs[i](x, adjacency_matrices)
                x = x + graph_out if self.cat_feat_gc else graph_out
            else:
                x = self.residual_convs[i](x)

            if self.verbose:
                print('   Graph output: ', x.shape)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)  # ignore last X?
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # downsample to (bs, seq_length, nodes, 1)
        if self.verbose:
            print('\nSkip shape: ', skip.shape)
            print('Output shape: ', x.shape)
            print('------------------------------------------------')
        return x.squeeze(dim=-1)  # (bs, seq_y, n)

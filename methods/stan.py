import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class stan_model(nn.Module):
    """
    1.attribute embeddig(dimension reduction for one-hot features)
    2.attention
    3.cnn
    4.linear layer
    """

    def __init__(
            self,
            time_windows_dim: int,
            spatio_windows_dim: int,
            feat_dim: int,
            num_classes: int,
            attention_hidden_dim: int,
            cate_unique_num: list = [1664, 216, 2500],
            filter_sizes: tuple = (2, 2, 2),
            num_filters: int = 64,
            in_channels: int = 1
    ) -> None:
        super().__init__()
        self.time_windows_dim = time_windows_dim
        self.spatio_windows_dim = spatio_windows_dim
        self.windows_dim = [self.time_windows_dim, self.spatio_windows_dim]
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.attention_hidden_dim = attention_hidden_dim

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        # cate embedding layer
        # ['Location','Type','Target']
        # self.cate_emdeds = nn.ModuleList([nn.Embedding(
        #     cate_unique_num[idx] + 1, cate_embed_dim) for idx in range(cate_feat_num)])

        # attention layer
        self.attention_W_1 = nn.Parameter(torch.Tensor(self.feat_dim, self.attention_hidden_dim).uniform_(0., 1.))
        self.attention_W_2 = nn.Parameter(torch.Tensor(self.feat_dim*2, self.attention_hidden_dim).uniform_(0., 1.))
        self.attention_U_1 = nn.Parameter(torch.Tensor(self.feat_dim, self.attention_hidden_dim).uniform_(0., 1.))
        self.attention_U_2 = nn.Parameter(torch.Tensor(self.feat_dim*2, self.attention_hidden_dim).uniform_(0., 1.))
        self.attention_V_1 = nn.Parameter(torch.Tensor(self.attention_hidden_dim, 1).uniform_(0., 1.))
        self.attention_V_2 = nn.Parameter(torch.Tensor(self.attention_hidden_dim, 1).uniform_(0., 1.))

        # cnn layer
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=filter_sizes
        )

        # FC layer
        self.flatten = nn.Flatten()
        self.linears = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.LazyLinear(self.num_classes)
        )

    def attention_layer(
            self,
            X: torch.Tensor,
            dim=2,
    ):
        self.output_att = []
        # split along time_windows axis
        input_att = torch.split(X, 1, dim=dim)
        for index, x_i in enumerate(input_att):
            # print(1, x_i.shape)
            x_i = x_i.reshape(-1, self.windows_dim[2 - dim], self.feat_dim*dim)
            # print(2, x_i.shape)
            c_i = self.attention(x_i, input_att, index, dim=dim)
            # print(3, c_i.shape)
            inp = torch.concat([x_i, c_i], axis=2)
            # print(4, inp.shape)
            self.output_att.append(inp)

        input_conv = torch.reshape(torch.concat(self.output_att, axis=dim),
                                   [-1, self.time_windows_dim, self.spatio_windows_dim, self.feat_dim * 2 * dim])
        # print(5, input_conv.shape)
        return input_conv

    def cnn_layer(
            self,
            input: torch.Tensor
    ):
        if len(input.shape) == 4:
            self.input_conv_expanded = torch.unsqueeze(input, 1)
        elif len(input.shape) == 5:
            self.input_conv_expanded = input
        else:
            print("Wrong conv input shape!")

        self.input_conv_expanded = F.relu(self.conv(input))

        return self.input_conv_expanded

    def attention(self, x_i, x, index, dim=1):
        e_i = []
        c_i = []

        for i in range(len(x)):
            output = x[i]
            output = output.reshape(-1, self.windows_dim[2 - dim], self.feat_dim * dim)
            if dim == 1:
                att_hidden = torch.tanh(torch.add(torch.matmul(
                    x_i, self.attention_W_1), torch.matmul(output, self.attention_U_1)))
                e_i_j = torch.matmul(att_hidden, self.attention_V_1)
            else:
                att_hidden = torch.tanh(torch.add(torch.matmul(
                    x_i, self.attention_W_2), torch.matmul(output, self.attention_U_2)))
                e_i_j = torch.matmul(att_hidden, self.attention_V_2)
            e_i.append(e_i_j)
        # print(7, e_i[0].shape)
        e_i = torch.concat(e_i, axis=2)
        # print(8, e_i.shape)
        # print(f"e_i shape: {e_i.shape}")
        alpha_i = F.softmax(e_i, dim=2)
        alpha_i = torch.split(alpha_i, 1, 2)  # !!!

        for j, (alpha_i_j, output) in enumerate(zip(alpha_i, x)):
            if j == index:
                continue
            else:
                output = output.reshape(-1, self.windows_dim[2 - dim], self.feat_dim * dim)
                c_i_j = torch.multiply(alpha_i_j, output)
                c_i.append(c_i_j)
        rshape = [-1, self.time_windows_dim, self.spatio_windows_dim, self.feat_dim * dim]
        rshape[dim] -= 1
        # print(9, c_i[0].shape)
        c_i = torch.reshape(torch.concat(c_i, axis=1), rshape)
        # print(10, c_i.shape)
        c_i = torch.sum(c_i, dim=dim)
        return c_i

    def forward(self, X_nume):
        # X shape be like: (batch_size, time_windows_dim, feat_dim)
        out = self.attention_layer(X_nume, dim=1)

        out = self.attention_layer(out, dim=2)
        out = torch.unsqueeze(out, 1)
        out = self.cnn_layer(out)
        # print(out.shape)
        out = self.flatten(out)
        # print(out.shape)
        out = self.linears(out)
        # print(out.shape)
        # quit()
        return out

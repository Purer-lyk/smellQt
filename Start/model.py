import numpy as np
import torch
import torch.nn as nn
from config import cfg, device
import yaml
from typing import Union
import os


class FC(nn.Module):
    def __init__(self, inc, ouc, act=True, bn=True, drop: Union[float, bool] = False):
        super(FC, self).__init__()
        self.line = nn.Linear(inc, ouc, bias=False)
        self.bn = nn.BatchNorm1d(ouc) if bn else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()
        if drop is True:
            self.drop = nn.Dropout()
        elif drop is False:
            self.drop = nn.Identity()
        else:
            self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 多个identity不会增加参数量
        return self.drop(self.act(self.bn(self.line(x))))


class Conv(nn.Module):
    def __init__(self, inc, ouc, k=3, s=1, p=0, g=1, ispool=False, act: Union[bool, nn.Module] = True,
                 drop: bool = False) -> None:
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inc,
                      out_channels=ouc,
                      kernel_size=k,
                      stride=s,
                      padding=p,
                      groups=g,
                      bias=False),
            nn.Dropout(0.5),
        ) if drop else nn.Conv2d(in_channels=inc,
                                 out_channels=ouc,
                                 kernel_size=k,
                                 stride=s,
                                 padding=p,
                                 groups=g,
                                 bias=False)
        self.bn = nn.BatchNorm2d(ouc)
        if isinstance(act, nn.Module):
            self.act = act()
        elif isinstance(act, bool):
            self.act = nn.ReLU() if act else nn.Identity()
        self.pool = nn.MaxPool2d(cfg.pools)
        self.ispool = ispool

    def forward(self, x):
        return self.act(self.bn(self.conv(x))) if not self.ispool else self.pool(self.act(self.bn(self.conv(x))))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):  # B,C,H,W
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class BottleNeck(nn.Module):
    #  do not change the scale, just alter the channel
    def __init__(self, inc, ouc, s=1, g=1, add=False, act=True, drop=False) -> None:
        super(BottleNeck, self).__init__()
        self.block_neck = nn.Sequential(
            nn.Conv2d(in_channels=inc,
                      out_channels=inc,
                      kernel_size=1,
                      stride=s,
                      groups=g,
                      ),
            nn.Conv2d(in_channels=inc,
                      out_channels=inc,
                      kernel_size=3,
                      stride=s,
                      padding=1,
                      groups=g,
                      ),
            nn.Conv2d(in_channels=inc,
                      out_channels=ouc,
                      kernel_size=1,
                      stride=s,
                      groups=g,
                      ),
        )
        self.act = nn.ReLU() if act else nn.Identity()
        self.dropout = nn.Dropout(0.5 if drop else 0)
        self.bn = nn.BatchNorm2d(ouc)
        self.add = add

    def forward(self, x):
        return x + self.act(self.bn(self.dropout(self.block_neck(x)))) if self.add else \
            self.act(self.bn(self.dropout(self.block_neck(x))))


class DenseBlock(nn.Module):  # only three connect
    def __init__(self, inc, gr, dim=1, g=1):
        super(DenseBlock, self).__init__()
        self.c11 = nn.Sequential(
            Conv(inc=inc, ouc=gr * dim, k=1, g=g),
            Conv(inc=gr * dim, ouc=gr, k=3, p=1, g=gr * dim),
        )
        # concat
        self.c12 = nn.Sequential(
            Conv(inc=gr + inc, ouc=gr * dim, k=1, g=g),
            Conv(inc=gr * dim, ouc=gr, k=3, p=1, g=gr * dim),
        )
        # concat
        self.c13 = nn.Sequential(
            Conv(inc=2 * gr + inc, ouc=gr * dim, k=1, g=g),
            Conv(inc=gr * dim, ouc=gr, k=3, p=1, g=gr * dim),
        )
        # concat
        self.concat = Concat()

    def forward(self, x):
        x = self.concat([x, self.c11(x)])
        x = self.concat([x, self.c12(x)])
        return self.concat([x, self.c13(x)])


class Transition(nn.Module):
    def __int__(self, inc, red=0.5):
        super(Transition, self).__int__()
        self.conv = Conv(inc=inc, ouc=inc * red, k=1)
        self.maxpool = nn.MaxPool2d(cfg.pools)

    def forward(self, x):
        return self.maxpool(self.conv(x))


class Inception(nn.Module):
    def __init__(self, inc, ouc, dw=True):
        #  also do not change the scale, just alter the channel
        super(Inception, self).__init__()
        self.conv1 = Conv(inc=inc, ouc=ouc, k=1)
        self.conv2 = Conv(inc=inc, ouc=ouc, k=3, p=1, g=inc if dw else 1)
        self.conv3 = Conv(inc=inc, ouc=ouc, k=5, p=2, g=inc if dw else 1)
        self.pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.BatchNorm2d(inc),  # use for pool
            nn.ReLU(inplace=True)
        )
        self.concat = Concat()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        p = self.pool(x)
        return self.concat([c1, c2, c3, p])


class TransformerLayer(nn.Module):
    def __init__(self, d, n_head):
        super(TransformerLayer, self).__init__()
        self.q = FC(d, d, bn=False)  # Q
        self.k = FC(d, d, bn=False)  # K
        self.v = FC(d, d, bn=False)  # V
        self.ma = nn.MultiheadAttention(embed_dim=d, num_heads=n_head)  # 多头注意力计算
        self.fc1 = FC(d, d, bn=False)
        self.fc2 = FC(d, d, bn=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x  # 优化残差
        x = self.fc2(self.fc1(x) + x) + x  # 优化残差
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d1, d2, wn, n_head, n_layer, learn_em=False):
        super(TransformerBlock, self).__init__()
        self.word_nums = wn
        self.embed = nn.Embedding(self.word_nums + 1, d2)
        self.learnable_embed = nn.Linear(d1, d2, bias=False)
        self.pfc = nn.Linear(d2, d2, False)  # 位置编码
        self.tr = nn.Sequential(*(TransformerLayer(d2, n_head) for _ in range(n_layer)))
        self.learn_em = learn_em

    def forward(self, x):
        if self.learn_em:
            return self._forward_learn_embed(x)
        return self._forward_normal_embed(x)

    def _forward_learn_embed(self, x):
        cls = torch.LongTensor([[2.]] * len(x)).to(device)
        x = torch.concatenate((x, cls), dim=1)
        x = self.learnable_embed(x)
        x = x.permute(1, 0, 2)
        return self.tr(x + self.pfc(x)).permute(1, 0, 2)

    def _forward_normal_embed(self, x):
        cls = torch.LongTensor([[self.word_nums]] * len(x)).to(device)  # 256代表一个特别的token即class
        x = torch.concatenate((x, cls), dim=1)
        # print("-----------------------------------")
        # print(x)
        x = self.embed(x)
        x = x.permute(1, 0, 2)  # (step, batch, dims)
        # print(x.shape)
        # print("------------------------------------")
        return self.tr(x + self.pfc(x)).permute(1, 0, 2)  # 这个转维度是一定需要有的


class ResNet(nn.Module):
    def __init__(self) -> None:
        super(ResNet, self).__init__()
        self.conv1 = Conv(inc=cfg.cnn.c1_inc, ouc=cfg.cnn.c1_ouc, p=1)  # in:9*9*1, out:9*9*3
        self.conv2 = Conv(inc=cfg.cnn.c1_ouc, ouc=cfg.cnn.c2_ouc, p=1, ispool=True)  # in:9*9*3, out:9*9*16
        # max_pooling
        self.conv3 = Conv(inc=cfg.cnn.c2_ouc, ouc=cfg.cnn.c4_ouc, p=1)  # in:3*3*16, out:3*3*32
        # self.transit = Conv(inc=cfg.cnn.c2_ouc, ouc=cfg.cnn.c4_ouc, k=1)  # in:3*3*32, out:3*3*128
        self.bottle = BottleNeck(inc=cfg.cnn.c4_ouc, ouc=cfg.cnn.c4_ouc, add=True)  # in:3*3*128, out:3*3*128
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # in:3*3*128, out:1*1*128
            Conv(inc=cfg.cnn.c4_ouc, ouc=cfg.cnn.c5_ouc, k=1, drop=True),
            nn.Flatten(),
            nn.Linear(in_features=cfg.cnn.c5_ouc, out_features=cfg.classes, bias=False),
            # nn.Softmax(1),
        )

    def forward(self, x):
        # x=x.view(-1,1,24,24)#front is rows, back is cols
        return self.dense(self.bottle(self.conv3(self.conv2(self.conv1(x)))))


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = Conv(inc=cfg.cnn.c1_inc, ouc=cfg.cnn.c1_ouc, p=1)  # in:9*9*1, out:9*9*3
        self.conv2 = Conv(inc=cfg.cnn.c1_ouc, ouc=cfg.cnn.c2_ouc, p=1, ispool=True)  # in:9*9*3, out:9*9*16
        # max_pooling
        self.conv3 = Conv(inc=cfg.cnn.c2_ouc, ouc=cfg.cnn.c3_ouc, p=1, g=cfg.cnn.c2_ouc)  # in:3*3*16, out:3*3*32
        self.conv4 = Conv(inc=cfg.cnn.c3_ouc, ouc=cfg.cnn.c4_ouc, p=1)  # in:3*3*32, out:3*3*128
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # in:3*3*128, out:1*1*128
            Conv(inc=cfg.cnn.c4_ouc, ouc=cfg.cnn.c5_ouc, k=1, drop=True),
            nn.Flatten(),
            nn.Linear(in_features=cfg.cnn.c5_ouc, out_features=cfg.classes, bias=False),
            # nn.Softmax(1),
        )

    def forward(self, x):
        return self.dense(self.conv4(self.conv3(self.conv2(self.conv1(x)))))


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.conv1 = Conv(inc=cfg.cnn.c1_inc, ouc=cfg.cnn.c1_ouc, p=1)  # in:9*9*1, out:9*9*3
        self.conv2 = Conv(inc=cfg.cnn.c1_ouc, ouc=cfg.cnn.c2_ouc, p=1, ispool=True)  # in:9*9*3, out:9*9*16
        # max_pooling
        self.conv3 = Conv(inc=cfg.cnn.c2_ouc, ouc=cfg.cnn.c3_ouc, p=1, g=cfg.cnn.c2_ouc)  # 3*3*16, 3*3*32
        self.inception = Inception(inc=cfg.cnn.c3_ouc, ouc=cfg.cnn.c3_ouc)
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # in:3*3*128, out:1*1*128
            Conv(inc=cfg.cnn.c4_ouc, ouc=cfg.cnn.c5_ouc, k=1, drop=True),
            nn.Flatten(),
            nn.Linear(in_features=cfg.cnn.c5_ouc, out_features=cfg.classes, bias=False),
            # nn.Softmax(1),
        )

    def forward(self, x):
        return self.dense(self.inception(self.conv3(self.conv2(self.conv1(x)))))


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.conv1 = Conv(inc=cfg.cnn.c1_inc, ouc=cfg.cnn.c1_ouc, p=1)  # in:9*9*1, out:9*9*3
        self.conv2 = Conv(inc=cfg.cnn.c1_ouc, ouc=cfg.cnn.c2_ouc, g=cfg.cnn.c1_ouc, p=1,
                          ispool=True)  # in:9*9*3, out:9*9*16
        # max_pooling
        self.conv3 = Conv(inc=cfg.cnn.c2_ouc, ouc=cfg.cnn.c3_ouc, p=1, g=cfg.cnn.c2_ouc)  # 3*3*16, 3*3*32
        self.denseblock = DenseBlock(inc=cfg.cnn.c3_ouc, gr=cfg.cnn.c3_ouc)
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # in:3*3*128, out:1*1*128
            Conv(inc=cfg.cnn.c4_ouc, ouc=cfg.cnn.c5_ouc, k=1, drop=True),
            nn.Flatten(),
            nn.Linear(in_features=cfg.cnn.c5_ouc, out_features=cfg.classes, bias=False),
            # nn.Softmax(1),
        )

    def forward(self, x):
        return self.dense(self.denseblock(self.conv3(self.conv2(self.conv1(x)))))


class SMT(nn.Module):
    def __init__(self):
        super(SMT, self).__init__()
        self.attention = TransformerBlock(cfg.att.init_dims, cfg.att.word_dims, cfg.att.word_nums,
                                          cfg.att.nhead, cfg.att.nlayer)
        self.dense1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=cfg.att.word_dims, out_features=cfg.att.fc1),
            nn.BatchNorm1d(num_features=cfg.att.fc1),
            nn.ReLU(),
        )
        self.dense2 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(in_features=cfg.att.fc1, out_features=cfg.att.fc2),
            nn.BatchNorm1d(num_features=cfg.att.fc2),
            # nn.ReLU(),
        )
        self.dense3 = nn.Linear(in_features=cfg.att.fc2, out_features=cfg.classes)

    def forward(self, x):
        print(self.attention(x).shape)
        x = self.attention(x)[:, -1, :]  # batch, step, dims
        # print(x)
        return self.dense3(self.dense2(self.dense1(x)))


class SIT(nn.Module):
    def __init__(self):
        super(SIT, self).__init__()
        self.embed = nn.Embedding(cfg.att.word_nums + 1, cfg.att.word_dims)
        self.residual = nn.Linear(cfg.att.word_dims, cfg.att.word_dims)
        # 81个数据，都是从归一化到了0~1，映射到图像像素0~255, shape:(batch, step, dims)
        self.layer = nn.TransformerEncoderLayer(d_model=cfg.att.word_dims, nhead=cfg.att.nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.layer, num_layers=cfg.att.nlayer)
        self.dense1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=cfg.att.word_dims, out_features=cfg.att.fc1),
            nn.BatchNorm1d(num_features=cfg.att.fc1),
            nn.ReLU(),
        )
        self.dense2 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(in_features=cfg.att.fc1, out_features=cfg.att.fc2),
            nn.BatchNorm1d(num_features=cfg.att.fc2),
            # nn.ReLU(),
        )
        self.dense3 = nn.Linear(in_features=cfg.att.fc2, out_features=cfg.classes)

    def forward(self, x):
        cls = torch.LongTensor([[cfg.att.word_nums]] * len(x)).to(device)  # 256代表一个特别的token即class
        temp_x = self.embed(torch.concatenate((x, cls), dim=1))
        x = self.encoder(temp_x + self.residual(temp_x))[:, -1, :]
        # print(x1.shape)
        return self.dense3(self.dense2(self.dense1(x)))


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp1 = nn.Sequential(nn.Linear(cfg.mlp.inc, cfg.mlp.fc1, bias=False),
                                  nn.BatchNorm1d(num_features=cfg.mlp.fc1),
                                  nn.ReLU()
                                  )
        self.mlp2 = nn.Sequential(nn.Linear(cfg.mlp.fc1, cfg.mlp.fc2, bias=False),
                                  nn.BatchNorm1d(num_features=cfg.mlp.fc2),
                                  nn.ReLU(),
                                  nn.Dropout(0.3)
                                  )
        self.mlp3 = nn.Sequential(nn.Linear(cfg.mlp.fc2, cfg.mlp.fc3, bias=False),
                                  nn.BatchNorm1d(num_features=cfg.mlp.fc3),
                                  nn.ReLU(),
                                  )
        self.mlp4 = nn.Linear(cfg.mlp.fc3, cfg.classes, bias=False)

    def forward(self, x):
        return self.mlp4(self.mlp3(self.mlp2(self.mlp1(x))))


class COMBINATION(nn.Module):  # attention + cnn, 先用attention因为防止cnn破坏数据的序列特征
    def __init__(self):
        super(COMBINATION, self).__init__()
        self.attention = TransformerBlock(cfg.com.init_dims, cfg.com.word_dims, cfg.com.word_nums,
                                          cfg.com.nhead, cfg.com.nlayer)
        self.c1 = Conv(cfg.com.c1_inc, cfg.com.c1_ouc, p=1, ispool=True)
        # maxpool
        self.c2 = Conv(cfg.com.c1_ouc, cfg.com.c2_ouc, p=1, ispool=True)
        # maxpool
        self.c3 = Conv(cfg.com.c2_ouc, cfg.com.c3_ouc, p=1)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Flatten(),
                                nn.Linear(cfg.com.c3_ouc, cfg.classes, bias=False))
        self.cnn_size = int(np.sqrt(cfg.com.word_dims))

    def forward(self, x):
        x = self.attention(x)[:, -1, :].reshape(-1, 1, self.cnn_size, self.cnn_size)
        x = self.fc(self.c3(self.c2(self.c1(x))))
        return x


# 通过yaml自定义的模型
class CUSTOM(nn.Module):
    def __init__(self, y, inc):
        super(CUSTOM, self).__init__()
        self.yaml_p = y
        self.indim: int = 4
        self.model = self.parse_model([inc])
        self.__class__.__name__ += os.path.split(self.yaml_p)[-1][:-5]

    def forward(self, x):
        # todo:目前concat还不适配这个推理方式，要用concat建议用bottelneck就行
        assert len(x.shape) == self.indim
        for m in self.model:
            # print(m)
            if m.n is TransformerBlock:
                # print(m(x).shape)
                x = m(x)[:, -1, :]
            else:
                x = m(x)
            # print(x.shape)
        return x

    def parse_model(self, ch: list):
        with open(self.yaml_p, encoding='ascii', errors='ignore') as f:
            struct = yaml.safe_load(f)
        f.close()
        nc = struct['nc']
        self.indim = struct['indim']
        layers, c2 = [], ch[-1]
        for i, (f, n, m, args) in enumerate(struct['structure']):
            m = eval(m)
            for j, a in enumerate(args):
                args[j] = eval(a) if isinstance(a, str) else a
            nl = eval(n) if isinstance(n, str) else n  # 暂时不用
            prel = f
            if m in {BottleNeck, Inception, TransformerBlock,
                     nn.Linear, FC}:
                c2 = args[0]
                args = [ch[f], *args]
            elif m is DenseBlock:
                assert args[0] / ch[f] == 4
                c2 = args[0]
                args = [ch[f], args[0] / 4, *args[1:]]  # 还有修改空间
            elif m is Concat:
                c2 = (sum(ch[p]) for p in f)
            elif m is Transition:
                c2 = args[0] * ch[f] if len(args) else 0.5 * ch[f]
                args = [ch[f], *args]
                assert c2 % 1 == 0
            elif m is nn.BatchNorm1d:
                c2 = ch[f]
                args = [c2]
            elif m in {nn.ReLU, nn.SiLU, nn.Flatten, nn.AdaptiveAvgPool2d, nn.Dropout}:
                c2 = ch[f]
            elif m is Conv:
                c2 = args[0]
                if len(args) >= 5 and args[4] == -1:
                    args[4] = ch[f]
                args = [ch[f], *args]
            m_ = nn.Sequential(*(m(*args) for _ in range(nl))) if nl > 1 else m(*args)
            m_.f, m_.n = f, m
            layers.append(m_)
            ch.append(c2)
        return nn.Sequential(*layers)


if __name__ == "__main__":
    a = torch.randn(100, 10, 32)
    b = nn.BatchNorm2d(32)
    print(b(a))

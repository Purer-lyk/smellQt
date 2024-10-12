import torch
import torch.nn as nn
from cfg import device, conf
from typing import Union
from Start import model


class FlatoSequence(nn.Module):
    def __init__(self):
        super(FlatoSequence, self).__init__()

    def forward(self, x):
        xshape = x.shape
        x = torch.reshape(x, (xshape[0], xshape[1], -1))
        return x.permute(0, 2, 1)


class TransformerBlock(nn.Module):
    def __init__(self, d1, d2, n_head, n_layer):
        super(TransformerBlock, self).__init__()
        self.word_dims = d2
        self.learnable_embed = nn.Linear(d1, d2, bias=False)
        self.pfc = nn.Linear(d2, d2, False)  # 位置编码
        self.tr = nn.Sequential(*(model.TransformerLayer(d2, n_head) for _ in range(n_layer)))

    def forward(self, x):
        if not x.shape[-1] == self.word_dims:
            x = self.learnable_embed(x)
        x = x.permute(1, 0, 2)
        return self.tr(x + self.pfc(x)).permute(1, 0, 2)


class STDSCT(nn.Module):
    def __init__(self):
        super(STDSCT, self).__init__()
        self.cnn1 = model.Conv(inc=2, ouc=8, k=1)
        self.cnn2 = model.Conv(inc=8, ouc=32, k=1)
        self.flatten = FlatoSequence()
        self.attention = TransformerBlock(conf.stsmt.init_dims, conf.stsmt.word_dims,
                                          conf.stsmt.nhead, conf.stsmt.nlayer)

        self.mlp1 = nn.Sequential(
            nn.BatchNorm1d(conf.stsmt.word_dims),
            nn.ReLU(),
            model.FC(conf.stsmt.word_dims, conf.stsmt.fc1, drop=True)
        )
        self.mlp2 = model.FC(conf.stsmt.fc1, conf.stsmt.fc2, drop=True)
        self.mlp3 = model.FC(conf.stsmt.fc2, conf.stsmt.classes, act=False, bn=False)

    def forward(self, x):
        return self.mlp3(self.mlp2(self.mlp1(self.attention(self.flatten(self.cnn2(self.cnn1(x))))[:, -1, :])))


class STDSMT(nn.Module):
    def __init__(self):
        super(STDSMT, self).__init__()
        self.attention = TransformerBlock(conf.stsmt.init_dims, conf.stsmt.word_dims, conf.stsmt.nhead,
                                          conf.stsmt.nlayer)
        self.mlp1 = nn.Sequential(
            nn.BatchNorm1d(conf.stsmt.word_dims),
            nn.ReLU(),
            model.FC(conf.stsmt.word_dims, conf.stsmt.fc2)
        )
        # self.mlp2 = model.FC(conf.stsmt.fc1, conf.stsmt.fc2, drop=True)
        self.mlp3 = model.FC(conf.stsmt.fc2, conf.stsmt.classes, act=False, bn=False)

    def forward(self, x):
        return self.mlp3(self.mlp1(self.attention(x)[:, -1, :]))


if __name__ == "__main__":
    tensor = torch.randn([2, 2, 12, 12])
    # tensor = torch.randn([2, 145, 2])
    model = STDSCT()
    output = model(tensor)
    print(output)

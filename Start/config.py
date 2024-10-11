import torch
from easydict import EasyDict as Edict

# 有些在GPU下报的不清楚的错误用CPU再跑一遍就能清除错误具体是什么了
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
infer_device = torch.device("cpu")
cfg = Edict()

# for Attention
cfg.att = Edict()
# for CNN
cfg.cnn = Edict()
# for MLP
cfg.mlp = Edict()
# for combination
cfg.com = Edict()

cfg.att.init_dims = 1  # use self learnable embedding
cfg.att.word_nums = 256  # use pytorch default embedding
cfg.att.word_dims = 128
cfg.att.nhead = 16
cfg.att.nlayer = 2
# 128*512
cfg.att.fc1 = 512
# 512*16
cfg.att.fc2 = 16
# 16*4

cfg.cnn.c1_inc = 1
cfg.cnn.c1_ouc = 4
cfg.cnn.c2_ouc = 16
cfg.cnn.c3_ouc = 32
cfg.cnn.c4_ouc = 128
cfg.cnn.c5_ouc = 256
# 256*4

cfg.mlp.inc = 100
cfg.mlp.fc1 = 512
cfg.mlp.fc2 = 512
cfg.mlp.fc3 = 16

cfg.com.init_dims = 1
cfg.com.word_nums = 256
cfg.com.word_dims = 64
cfg.com.nhead = 16
cfg.com.nlayer = 2  # output: 1*64
cfg.com.c1_inc = 1
cfg.com.c1_ouc = 16
cfg.com.c2_ouc = 64
cfg.com.c3_ouc = 256

cfg.classes = 4

cfg.pools = 2

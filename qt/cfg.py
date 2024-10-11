from easydict import EasyDict as Edict
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Edict()
conf.stsmt = Edict()

conf.stsmt.word_dims = 32
conf.stsmt.word_nums = 511
conf.stsmt.init_dims = 2
conf.stsmt.nhead = 4
conf.stsmt.nlayer = 4

conf.stsmt.fc1 = 64
conf.stsmt.fc2 = 8
conf.stsmt.classes = 3



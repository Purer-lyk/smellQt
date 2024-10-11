import glob
import os
from Master.Start import Train
import torch
from model import STDSMT, STDSCT
from generate import readFromCsv, readFromXlsx, smt_dataset, sct_dataset, ori_path
from torch.utils import data
from Master.Start import model

# train_ori_path = "/dataset/train/"
# test_ori_path = "/dataset/test/"
mlist = ['SCT', 'SMT']


class sct_worker(Train.nn_base):
    def __init__(self, t_batch, v_batch, train_paths, test_paths, sp, m=None, eps=120, lr=0.001, lossf=None):
        super(sct_worker, self).__init__(t_batch, v_batch, sp, eps=eps, lr=lr, lossf=lossf)
        self.model = STDSCT().to(self.device) if m is None else m.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                           weight_decay=0.0005)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=0.99)
        self.train_dataLoader = get_tensor(train_paths, self.train_batch, 'SCT')
        self.valid_dataLoader = get_tensor(test_paths, self.valid_batch, 'SCT')
        self.best_model = self.model
        print(self.model)
        print(sum(x.numel() for x in self.model.parameters()))


class smt_worker(Train.nn_base):
    def __init__(self, t_batch, v_batch, train_paths, test_paths, sp, m=None, eps=120, lr=0.001, lossf=None):
        super(smt_worker, self).__init__(t_batch, v_batch, sp, eps=eps, lr=lr, lossf=lossf)
        self.model = m.to(self.device) if m else STDSMT().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                           weight_decay=0.0005)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=0.99)
        self.train_dataLoader = get_tensor(train_paths, self.train_batch, 'SMT')
        self.valid_dataLoader = get_tensor(test_paths, self.valid_batch, 'SMT')
        print(len(self.train_dataLoader))
        print(len(self.valid_dataLoader))
        self.best_model = self.model
        print(self.model)
        print(sum(x.numel() for x in self.model.parameters()))


def get_tensor(paths, batch, mtype: str):
    self_datas, self_labels = [], []
    self_dataset = None
    for p in paths:
        if isinstance(p, str):
            assert ".xls" in p or ".xlsx" in p
            datas, labels = readFromXlsx(p)
            self_datas.extend(datas)
            self_labels.extend(labels)
        elif isinstance(p, list):
            assert len(p) == 4
            datas, labels = readFromCsv(p)
            self_datas.extend(datas)
            self_labels.extend(labels)
    if mlist.index(mtype) == 0:
        self_dataset = sct_dataset(self_datas, self_labels)
    elif mlist.index(mtype) == 1:
        self_dataset = smt_dataset(self_datas, self_labels)
        # print(1)
    self_dataloader = data.DataLoader(self_dataset, len(self_dataset) if batch == -1 else batch,
                                      shuffle=True, drop_last=True)
    return self_dataloader


if __name__ == "__main__":
    smt = smt_worker(16, 8, glob.glob(ori_path+'train/*.xlsx'), glob.glob(ori_path+'test/*.xlsx'), './model/')
    smt.train()

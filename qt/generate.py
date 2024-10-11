import glob
import numpy
import torch
import numpy as np
from torch.utils import data
from openpyxl import load_workbook
import csv

ori_path = "./dataset/"
label_list = ['c2h5oh', 'mxylene', 'benzene']


class smt_dataset(data.Dataset):
    def __init__(self, input_tensors, input_labels):
        super(smt_dataset, self).__init__()
        self.data_tensors = input_tensors
        self.get_labels = input_labels

    def __len__(self):
        return len(self.data_tensors)

    def __getitem__(self, item):
        # single_tensor = self.data_tensor[item]
        # image_tensor = np.int32(np.round(single_tensor * 255.0))
        seq_tensor = dataToSequence(self.data_tensors[item])
        # print(image_tensor)
        single_label = self.get_labels[item]
        return seq_tensor, single_label


class sct_dataset(data.Dataset):
    def __init__(self, input_tensors, input_labels):
        super(sct_dataset, self).__init__()
        self.data_tensors = input_tensors
        self.get_labels = input_labels

    def __len__(self):
        return len(self.data_tensors)

    def __getitem__(self, item):
        image_tensor = dataToImage(self.data_tensors[item])
        single_label = self.get_labels[item]
        return image_tensor, single_label


def dataToImage(input_data: list):
    channel1 = normalization(torch.tensor(input_data[0]) - torch.tensor(input_data[1]))
    channel2 = normalization(torch.tensor(input_data[2]) - torch.tensor(input_data[3]))
    channel1 = torch.reshape(channel1, (1, 12, 12))
    channel2 = torch.reshape(channel2, (1, 12, 12))
    needMat = torch.concatenate([channel1, channel2], dim=1)
    return needMat


def dataToSequence(input_data: list):
    channel1 = normalization(torch.tensor(input_data[0]) - torch.tensor(input_data[1]))
    channel2 = normalization(torch.tensor(input_data[2]) - torch.tensor(input_data[3]))
    channel1 = torch.reshape(channel1, (144, 1))
    channel2 = torch.reshape(channel2, (144, 1))
    needSequence = torch.concatenate([channel1, channel2], dim=1)
    classToken = torch.tensor([[-1.1, 1.1]])
    needSequence = torch.concatenate([needSequence, classToken], dim=0)
    return needSequence


def normalization(data: torch.Tensor):
    calculbase = max(abs(data))  # 这个如果为0， 那么整个数组都是0
    return data/calculbase if calculbase else data


# 要求xlsx格式的数据集文件第一个sheet有标签列,输入为单个xlsx文件
def readFromXlsx(xlsx):
    train_datas = []
    train_labels = []

    xls_book = load_workbook(xlsx)
    sheets = xls_book.worksheets

    for index, sheet in enumerate(sheets):
        rows = sheet.rows
        for ir, row in enumerate(rows):
            row_val = [cell.value for cell in row]
            # print(row_val)
            if not index:
                train_datas.append([])
                train_labels.append(label_list.index(row_val[0]))
            if len(row_val) == 145:
                train_datas[ir].append([int(s) for s in row_val[1:]])
            elif len(row_val) == 144:
                train_datas[ir].append([int(s) for s in row_val])
    return train_datas, train_labels


# 要求csv每一次数据集的传入都以四个文件为单位传入，且每个文件都需要有标签列，输入为一个包含四个csv的列表
def readFromCsv(csvs):
    train_datas = []
    train_labels = []
    for index, cfile in enumerate(csvs):
        f = open(cfile, mode='r', encoding='utf-8-sig')
        lines = csv.reader(f, dialect='excel')
        for il, line in enumerate(lines):
            if not index:
                train_datas.append([])
                train_labels.append(label_list.index(line[0]))
            train_datas[il].append([int(s) for s in line[1:]])
    return train_datas, train_labels


if __name__ == "__main__":
    xlsxlist = glob.glob(ori_path+'*.xlsx')
    # print(xlsxlist)
    input_dats, input_labs = [], []
    for xlsxp in xlsxlist:
        dats, labs = readFromXlsx(xlsxp)
        input_dats.extend(dats)
        input_labs.extend(labs)
    # print(input_dats[0])
    # print(input_labs[0])
    ds = smt_dataset(input_dats, input_labs)
    self_dataloader = data.DataLoader(ds, 10, shuffle=True, drop_last=True)
    for d, l in self_dataloader:
        print(d)
        print(l)
    """
    体现特征：前后两个传感器的相对大小，谁比谁的数值大（丢失了）；每个点的差距
    第二种： 保留相对大小，差距大小也能保留，
    每个数据的格式：cell = [list1, list2, list3, list4]，total = [cell1, cell2, cell3...]
    """


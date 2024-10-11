import seaborn as sn
from matplotlib.font_manager import fontManager, FontProperties
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import glob
import torch
from torch.autograd import Variable
from config import infer_device, device
from Train import get_tensor
from generate import test_path, train_path
from Kans import *
import time

model_path = "./model/2024-07-01_CUSTOMResNet_20.pt"
# model_path = "./model/2024-07-01_CUSTOMSMT_13.pt"
# model_path = "./model/2024-07-01_CUSTOMMLP_7.pt"
# model_path = "./model/2024-07-01_KAN_19.pt"
pic_label = ["C$_{2}$H$_{5}$OH", "CH$_{2}$OH", "thin", "strong"]


def cfu_mat(label, output, mt):
    font_list = fontManager.ttflist
    font_found = None
    for font in font_list:
        if 'Times New Roman' in font.name:
            font_found = font
            break
    if font_found:
        font_properties = FontProperties(family=font_found.name, style='normal')

    print(label)
    print(output)
    sn.set()
    fig = plt.figure(figsize=(19, 14.4))
    axis = plt.subplot(111)
    mat = confusion_matrix(label, output, labels=range(len(pic_label)))
    print(mat)
    sn.heatmap(mat, annot=True, cmap='RdPu', annot_kws={"fontsize": 35},
               cbar=True,
               xticklabels=pic_label, yticklabels=pic_label)

    cbar = plt.gcf().axes[-1]
    # 设置color bar刻度标签的字体大小
    cbar.tick_params(labelsize=40)  # 将刻度标签的字体大小设置为30
    cbar.set_yticklabels(cbar.get_yticklabels(), fontweight='bold', family='serif')

    plt.title("KANs", fontsize=40, fontweight='bold', family='serif')
    plt.xlabel('output', fontsize=40, fontweight='bold', labelpad=10, family='serif')
    plt.ylabel('label', fontsize=40, fontweight='bold', labelpad=10, family='serif')
    plt.xticks(fontsize=40, fontweight='bold', family='serif')
    plt.yticks(fontsize=40, fontweight='bold', family='serif')
    k = len(glob.glob('pic/cfu*.png'))

    plt.savefig('pic/cfu' + str(k) + '.png')
    plt.show()


def infer():
    model = torch.load(model_path).eval()
    print(model)
    model_type = None
    if 'SMT' in model_path or \
            'SIT' in model_path:
        model_type = 'Attention'
    elif 'ResNet' in model_path or \
            'GoogleNet' in model_path or \
            'VGG' in model_path or \
            'DenseNet' in model_path:
        model_type = 'CNN'
    elif 'KAN' in model_path or \
            'MLP' in model_path:
        model_type = 'MLP'
    print(model_type)
    input_loader = get_tensor(test_path, 1, model_type)
    print(sum(x.numel() for x in model.parameters()))
    avgTime = 0
    for i in range(101):
        start = time.perf_counter()
        for input_tensor, input_label in input_loader:
            print(type(input_tensor))
            x, y = Variable(input_tensor).to(device), Variable(input_label).to(device)
            output = model(x).to(infer_device)
            value, pred = torch.max(torch.softmax(output, dim=1), 1)
            # cfu_mat(input_label.tolist(), pred.tolist(), model_type)
        end = time.perf_counter()
        print(f"run_time: {end - start}seconds")
        if i != 0:
            avgTime += (end - start)
    print(f"avg_run_time{avgTime/100.*1000./120.}ms")


if __name__ == '__main__':
    infer()

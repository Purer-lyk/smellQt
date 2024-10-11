import torch
import time
from torch.autograd import Variable
from generate import dataToSequence, readFromXlsx
from config import device
from model import STDSMT


def infer(model_path: str, input_data: list) -> int:
    if model_path == "":
        model = STDSMT()
    else:
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
    # print(model_type)
    input_tensor = dataToSequence(input_data).unsqueeze(0)
    print(sum(x.numel() for x in model.parameters()))
    start = time.perf_counter()
    # for input_tensor, input_label in input_loader:
    # print(len(input_label))
    x = Variable(input_tensor).to(device)
    output = model(x)
    value, pred = torch.max(torch.softmax(output, dim=1), 1)
    # print(torch.argmax(value))
    # cfu_mat(input_label.tolist(), pred.tolist(), model_type)
    end = time.perf_counter()
    print(f"run_time: {(end - start) * 1000}ms")
    return int(torch.argmax(value))


if __name__ == "__main__":
    modelp = "./model/2024-09-29_STDSMT_0.pt"
    xlsxp = "./dataset/test/data4.xlsx"
    datalist, labellist = readFromXlsx(xlsxp)
    print(datalist[0])
    result = infer(modelp, datalist[0])
    print(result)

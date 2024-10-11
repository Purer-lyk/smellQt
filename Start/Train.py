import csv
import datetime
from torch.autograd import Variable
from generate import *
from config import device
from torch.utils.tensorboard import SummaryWriter
from model import *
import glob
import os


class nn_base:
    def __init__(self, t_batch, v_batch, sp, eps=100, lr=0.01, lossf=None):
        self.device = device
        self.train_batch = t_batch  # 10
        self.valid_batch = v_batch  # 5
        self.epochs = eps
        self.learning_rate = lr
        self.loss_function = nn.CrossEntropyLoss() \
            if not lossf else lossf
        self.save_path = sp
        self.model = nn.Linear(10, 5)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)  # enforce the gradient down
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=0.9)  # use learning rate optimizer
        # 继承时用到
        self.train_dataLoader = None
        self.valid_dataLoader = None
        self.best_model = None
        self.best_step = 0

    def nn_run(self, dataloader, train: bool = True):
        # print(self.model)
        running_equals = 0
        running_correct = 0
        running_loss = 0
        for gas_data, label in dataloader:
            batch = len(gas_data)  # train is 10, valid is 5
            # print(batch)
            x, y = gas_data, label
            x, y = Variable(x).to(self.device), Variable(y).to(self.device)
            outputs = self.model(x)
            # print(outputs)
            value, pred = torch.max(torch.softmax(outputs, dim=1), 1)
            if train:
                self.optimizer.zero_grad()
                loss = self.loss_function(outputs, y.long())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.data  # 不需要除batch，crossentry默认求均值
            equal_correct = torch.tensor(pred == y.data, dtype=torch.float32).clone().detach()
            value = value * equal_correct
            running_equals += torch.sum(equal_correct) / batch
            running_correct += torch.sum(0.1 * equal_correct + 0.9 * value) / batch
            torch.cuda.empty_cache()
        return (running_correct / len(dataloader), running_loss / len(dataloader), running_equals / len(dataloader)) \
            if train else (running_correct / len(dataloader), running_equals / len(dataloader))

    def train(self):
        best_accuracy = 0
        best_classify = 0
        total_result = []
        logs = glob.glob('runs/result_*')
        log_name = "runs/result_" + str(len(logs))
        writer = SummaryWriter(log_dir=log_name, flush_secs=100)

        for step in range(self.epochs):
            step += 1
            train_outputs = self.nn_run(self.train_dataLoader)
            valid_outputs = self.nn_run(self.valid_dataLoader, False)
            # 处理结果
            total_result.append([float(train_outputs[1]), float(train_outputs[0]), float(valid_outputs[0])])
            show(train_outputs, valid_outputs)
            writer.add_scalar("loss", train_outputs[1], step)
            writer.add_scalar("train_acc", train_outputs[0], step)
            writer.add_scalar("test_acc", valid_outputs[0], step)
            writer.add_scalar("train_equal", train_outputs[2], step)
            writer.add_scalar("test_equal", valid_outputs[1], step)

            if 0.1 * train_outputs[0] + 0.9 * valid_outputs[0] > best_accuracy and \
                    train_outputs[2] + valid_outputs[1] >= best_classify:
                print("more excellent!!!\n")
                best_accuracy = 0.1 * train_outputs[0] + 0.9 * valid_outputs[0]
                best_classify = train_outputs[2] + valid_outputs[1]
                self.best_model = self.model
                self.best_step = step

            self.scheduler.step()
        self.export(total_result)
        print("best_step:", self.best_step)

    def export(self, total_result):
        save_name = self.get_model_name(self.model.__class__.__name__)
        # print(save_name[:-3])
        export_train_data(total_result, save_name[:-3])
        torch.save(self.best_model, save_name)

    def get_model_name(self, model_type: str):
        todate = str(datetime.date.today())
        models = glob.glob(self.save_path + '*_' + model_type + '_*.pt')
        cnt = 0
        for m in models:
            if todate in m:
                cnt += 1
        return self.save_path + todate + '_' + model_type + '_' + str(cnt) + '.pt'


class cnn_worker(nn_base):
    def __init__(self, t_batch, v_batch, sp, m=None, eps=80, lr=0.1, lossf=None):
        super(cnn_worker, self).__init__(t_batch, v_batch, sp, eps=eps, lr=lr, lossf=lossf)
        self.model = ResNet().to(self.device) if m is None else m.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                           weight_decay=0.0005)  # enforce the gradient down
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=0.99)  # use learning rate optimizer
        self.train_dataLoader = get_tensor(train_path, self.train_batch, 'CNN')
        self.valid_dataLoader = get_tensor(test_path, self.valid_batch, 'CNN')
        self.best_model = self.model
        print(self.model)
        print(sum(x.numel() for x in self.model.parameters()))


class transformer_worker(nn_base):
    def __init__(self, t_batch, v_batch, sp, m=None, eps=80, lr=0.1, lossf=None):
        super(transformer_worker, self).__init__(t_batch, v_batch, sp, eps=eps, lr=lr, lossf=lossf)
        self.model = SMT().to(self.device) if m is None else m.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                           weight_decay=0.0005)  # enforce the gradient down
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=0.99)  # use learning rate optimizer
        self.train_dataLoader = get_tensor(train_path, self.train_batch, 'Attention')
        self.valid_dataLoader = get_tensor(test_path, self.valid_batch, 'Attention')
        self.best_model = self.model
        print(self.model)
        print(sum(x.numel() for x in self.model.parameters()))


class mlp_worker(nn_base):
    def __init__(self, t_batch, v_batch, sp, m=None, eps=80, lr=1, lossf=None):
        super(mlp_worker, self).__init__(t_batch, v_batch, sp, eps=eps, lr=lr, lossf=lossf)
        self.model = MLP().to(self.device) if m is None else m.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                           weight_decay=0.0005)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=0.99)
        self.train_dataLoader = get_tensor(train_path, self.train_batch, 'MLP')
        self.valid_dataLoader = get_tensor(test_path, self.valid_batch, 'MLP')
        self.best_model = self.model
        print(self.model)
        print(sum(x.numel() for x in self.model.parameters()))


def export_train_data(train_result, name):
    if not os.path.exists(name):
        os.mkdir(name)
    with open(name + '/train_result.csv', 'a', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(['loss', 'train_s', 'valid_s'])
        for result in train_result:
            csv_write.writerow(result)
    f.close()


def show(t_outputs, v_outputs):
    print("Loss              || {:.12f}\n"
          "Train score       || {:.12f}   Train equal       || {:.12f}\n"
          "Valid score       || {:.12f}   Valid equal       || {:.12f}\n".format
          (float(t_outputs[1]),
           float(t_outputs[0]), float(t_outputs[2]),
           float(v_outputs[0]), float(v_outputs[1])))


def get_tensor(input_xls, batch_size, mtype: str):
    array_total = np.array([])
    label_total = np.array([])
    for i in input_xls:
        array, label = generate_nums(i)
        array_total, label_total = (array, label) if not len(array_total) \
            else (np.concatenate((array_total, array)), np.concatenate((label_total, label)))
    self_dataset = None
    mtype = mtype_list.index(mtype)
    if mtype == 0:
        self_dataset = transformer_dataset(array_total, label_total)
    elif mtype == 1:
        self_dataset = cnn_dataset(array_total, label_total)
    elif mtype == 2:
        self_dataset = mlp_dataset(array_total, label_total)
    self_dataloader = data.DataLoader(self_dataset, len(self_dataset) if batch_size == -1 else batch_size,
                                      shuffle=True, drop_last=True)

    return self_dataloader


if __name__ == '__main__':
    yaml_path = './structure/ResNet.yaml'
    if 'SMT' in yaml_path:
        smt = transformer_worker(10, 5, './model/', CUSTOM(yaml_path, 1))
        smt.train()
    elif 'ResNet' in yaml_path:
        cnn = cnn_worker(10, 5, './model/', CUSTOM(yaml_path, 1))
        cnn.train()
    elif 'MLP' in yaml_path:
        mlp = mlp_worker(10, 5, './model/', CUSTOM(yaml_path, 100))
        mlp.train()


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import *
from generate import generate_nums, train_path, test_path
from Train import nn_base, get_tensor


class KANLayer(nn.Module):
    def __init__(self, inc, ouc, de='cpu', grid_s=5, sp_order=3, scale_noise=0.1, scale_base=1.0,
                 scale_spline=1.0, enable_standalone=True, act=None, grid_eps=0.02, grid_range=[-1, 1]):
        """

        :param inc: input channel
        :param ouc: output channel
        :param grid_s: 基本的区间数量
        :param sp_order: 样条函数的阶数
        :param scale_noise: 初始用于初始化grid的噪声
        :param scale_base: 基本mlp计算的缩放系数
        :param scale_spline: 样条函数输出值的缩放系数
        :param enable_standalone: 是否允许样条函数的最小二乘逼近，todo:这个暂时先不管
        :param act: 默认激活函数
        :param grid_eps: 不根据数据分割网格的坐标在grid更新中的占比比重
        :param grid_range: 初始网格的范围
        """
        super(KANLayer, self).__init__()
        self.inc = inc
        self.ouc = ouc
        self.device = de
        self.grid_size = grid_s
        self.sp_order = sp_order
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.default_act = nn.SiLU() if act is None else act()
        # 以下两者会参与torch的grad优化训练

        # 这个是控制每个神经元函数拟合的精确性的，因而会通过与B样条求最小二乘来赋值，但也会手grad影响
        self.sp_weight = torch.nn.Parameter(torch.Tensor(ouc, inc, sp_order + grid_s)).to(de)
        # 这个是控制每个神经元的作用大小的，即论文里展示的随着训练轮数增加，有些神经元对应的子函数逐渐变得不参与最终端到端的拟合
        self.sp_scale = torch.nn.Parameter(torch.Tensor(ouc, inc)).to(de)

        self.grid_eps = grid_eps
        self.base_fc = nn.Linear(inc, ouc, bias=False, device=de)
        self.grid = torch.nn.Parameter(torch.einsum(
            'i,j->ij', torch.ones(inc), torch.linspace(grid_range[0], grid_range[1], grid_s + 1 + 2 * sp_order)
        )).requires_grad_(False)
        noise = (
                        (torch.rand(grid_s + 1, inc, ouc)) - 0.5
                ) * scale_noise / grid_s
        noise = noise.to(de)
        with torch.no_grad():
            self.sp_weight.data.copy_(self.curve2coeff(self.grid.T[sp_order:-sp_order], noise))
            nn.init.kaiming_uniform_(self.sp_scale)

    def forward(self, x):
        x_shape = x.shape
        # print("ori_x:", x)
        base_y = self.default_act(self.base_fc(x))  # MLP那一套的结果
        # 当前最小方差的系数乘上
        spline_y = F.linear(
            self.b_spline(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.ouc, -1),
        )  # KANs那一套的结果
        final_y = (self.scale_base * base_y + self.scale_spline * spline_y).view(*x_shape[:-1], self.ouc)  # 最后两者相加
        return final_y

    def b_spline(self, x, printf=False):
        """
        Compute the B-spline bases for the given input tensor.
        B-样条计算只是获得一个样条函数的结果，这个函数内没有任何参数参与pytorch的grad优化，但是其本身的结果因为每轮update_grid
        函数的运行而发生改变，因而在训练过程中也有优化其输出。

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
            :param printf: 是否打印查看中间输出
        """
        x, grid = x.to(self.device), self.grid.to(self.device)

        if printf:
            print("ori_grid", grid)
            print("input_x", x)
        x = x.unsqueeze(-1)
        bases = torch.tensor((x >= grid[:, :-1]) & (x < grid[:, 1:]), dtype=x.dtype)
        # if printf:
        #     print(bases)
        #     print('--------------')
        for k in range(1, self.sp_order + 1):
            bases = (
                            bases[:, :, :-1] * (x - grid[:, :-(k + 1)])
                            / (grid[:, k:-1] - grid[:, :-(k + 1)])
                    ) + (
                            bases[:, :, 1:] * (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                    )
        if printf:
            print(bases)
            print('--------------')
        return bases.contiguous().to(self.device)

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.
        self.sp_weight会跟随pytorch的grad优化改变，导致本次

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        x, y = x.to(self.device), y.to(self.device)
        '''
            pytorch不允许在CUDA下使用gels以外的最小二乘解决方法，因为gels的QR分解矩阵方式更适合并行化的CUDA，但gels不允许非满秩的矩阵
            参与计算，因为非满秩的矩阵会导致QR分解计算的结果中R矩阵至少一个零特征值，导致最小二乘结果不稳定，即可能会使计算结果错误。
            pytorch不希望出现不稳定的结果，且在CUDA下没有优化实现gels之外的最小二乘求解方法，因而暂时只能将数据转到cpu下计算最小二乘。
            由于数学功底不好，且自身开发底层原理能力弱，没有办法自行优化实现CUDA下gels*，基本上是只能用转设备这一个方法。
        '''
        A = self.b_spline(x).transpose(0, 1).to('cpu')
        B = y.transpose(0, 1).to('cpu')
        solution = torch.linalg.lstsq(A, B).solution  # 最小二乘问题

        result = solution.permute(2, 0, 1)
        return result.to(self.device)

    @property
    def scaled_spline_weight(self):
        # 这两个参数都会参与grad的优化训练
        return self.sp_weight * (
            self.sp_scale.unsqueeze(-1)
        ).to(self.device)

    @torch.no_grad()  # 更新grid不参与梯度计算
    def update_grid(self, x, margin=0.01):
        batch = x.size(0)
        # coef即grid_size + sp_order
        # 当前输入的x并不对应当前的grid(即x在grid的中间)所以这个样条函数的输出是定义为不准确的
        splines = self.b_spline(x).permute(1, 0, 2)  # (in, batch, coef)
        # 经过torch的grad优化训练后的
        ori_coef = self.scaled_spline_weight.permute(1, 2, 0)  # (in, coef, out)

        last_sp_ouput = torch.bmm(splines, ori_coef).permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]  # return 2 tensor, first is result, second is index
        grid_main = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64)
        ]
        adjust_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        # adjust_step = torch.tensor(np.repeat(adjust_step.unsqueeze(0), 6, axis=0))
        # print(adjust_step.shape)
        grid_adjust = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * adjust_step
                + x_sorted[0]
                - margin
        )
        # 这里的grid才完成更新，才是当前的grid
        grid = self.grid_eps * grid_adjust + (1 - self.grid_eps) * grid_main
        # print(grid.shape)
        # kk = adjust_step * torch.arange(self.sp_order, 0, -1, device=x.device)
        # print(kk.shape)
        new_grid = torch.concatenate(
            [
                grid[0]
                - torch.einsum("i,j->ji", adjust_step, torch.arange(self.sp_order, 0, -1, device=x.device)),

                grid,

                grid[-1]
                + torch.einsum("i,j->ji", adjust_step, torch.arange(1, self.sp_order + 1, device=x.device))
            ],
            dim=0,
        )
        # 样条函数里面用的时候grid区间在第二个维度，但当前为第一维度，故转置
        self.grid.copy_(new_grid.T)
        '''
        通过上一轮的B样条（grid改变前）计算结果和梯度优化后的sp_weight相乘的结果和当前B样条（grid改变后）的结果做最小二乘
        得到最小方差的系数
        '''
        self.sp_weight.data.copy_(self.curve2coeff(x, last_sp_ouput))


class KAN(nn.Module):
    def __init__(
            self,
            layers_hidden,
            de: str = "cpu",
            grid_size=5,
            sp_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            default_act=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        # print(de)
        self.device = de
        self.grid_size = grid_size
        self.sp_order = sp_order

        self.layers = nn.Sequential(
            *(
                KANLayer(
                    inc=in_channel,
                    ouc=ou_channel,
                    de=de,
                    grid_s=grid_size,
                    sp_order=sp_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    act=default_act,
                    grid_eps=grid_eps,
                    grid_range=grid_range
                ) for in_channel, ou_channel in zip(layers_hidden, layers_hidden[1:])
            )
        )

    def forward(self, x):
        for layer in self.layers:
            layer.update_grid(x)
            x = layer(x)
        return x


class kan_worker(nn_base):
    def __init__(self, t_batch, v_batch, sp, m=None, eps=80, lr=0.1, lossf=None):
        super(kan_worker, self).__init__(t_batch, v_batch, sp, eps=eps, lr=lr, lossf=lossf)
        self.model = KAN([100, 64, 4], de=self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                           weight_decay=0.0005)  # enforce the gradient down
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=0.99)  # use learning rate optimizer
        self.train_dataLoader = get_tensor(train_path, self.train_batch, 'MLP')
        self.valid_dataLoader = get_tensor(test_path, self.valid_batch, 'MLP')
        self.best_model = self.model
        print(self.model)
        print(sum(x.numel() for x in self.model.parameters()))


if __name__ == '__main__':
    # k1 = torch.Tensor(torch.ones(3, 100) * 0.5)
    # model = KAN([100, 64, 4])
    # print(model)
    # print(model(k1))
    # model(k1)
    kans = kan_worker(10, 5, './model/')
    kans.train()

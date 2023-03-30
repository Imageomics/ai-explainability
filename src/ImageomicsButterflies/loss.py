import torch
import torch.nn as nn

class TransformLoss(nn.Module):

    def __init__(self, original,  x, beta=2, reg_lambda=0.001, reg_original=0.001):
        super().__init__()
        self.x = x
        self.original = original
        self.beta = beta
        self.reg_lambda = reg_lambda
        self.reg_original = reg_original
        self.mse = nn.MSELoss(size_average=False).cuda()
        self.cs = nn.CosineSimilarity(dim=0).cuda()
        self.l1 = nn.L1Loss(size_average=False).cuda()
        self.eps = 1e-3

    def reg_loss(self, z):
        x_diff = z[:, :-1, :-1] - z[:, :-1, 1:]
        y_diff = z[:, :-1, :-1] - z[:, 1:, :-1]

        sq_diff = torch.clamp(x_diff * x_diff + y_diff * y_diff, self.eps, 10000000)
        return torch.norm(sq_diff, self.beta / 2.0) ** (self.beta / 2.0)

        #right_diff = torch.cat((z[:, :, 1:], z[:, :, -1].unsqueeze(2)), dim=2) - z
        #bottom_diff = torch.cat((z[:, 1:], z[:, -1].unsqueeze(1)), dim=1) - z
        #loss = torch.norm((right_diff**2) + (bottom_diff**2), self.beta / 2) ** (self.beta / 2)
        #return loss

    def forward(self, z, z_act):
        mse = self.mse(z_act, self.x)# - self.cs(z_act, self.x)
        smooth_loss = self.reg_loss(z) * self.reg_lambda
        change_loss = self.l1(z, self.original) * self.reg_original
        return mse, smooth_loss, change_loss
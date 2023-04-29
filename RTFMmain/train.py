import numpy as np
import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss
from tqdm import tqdm


def sparsity(arr, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss

def smooth(arr, lamda1):  # smoothness (15)  input 32 x 32 x 1
    arr1 = arr[:, :-1, :]  # we need to do it only per video.  32 x 31 x 1
    arr2 = arr[:, 1:, :]   # 32 x 31 x 1
    loss = torch.sum((arr2 - arr1) ** 2)
    del arr2
    del arr1
    return lamda1 * loss

class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)

class RTFM_loss(torch.nn.Module):
    def __init__(self, alpha, margin):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = torch.nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal

        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()
        label = label.cuda()
        loss_cls = self.criterion(score, label)  # BCE loss in the score space

        loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))

        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)

        loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)

        # loss_total = loss_cls + self.alpha * loss_rtfm

        return loss_cls, self.alpha * loss_rtfm


def train(nloader, aloader, model, params, optimizer, viz, device):
    with torch.set_grad_enabled(True):
        model.train()
        total_cost, loss_cls_sum, loss_rtfm_sum, loss_sparse_sum, loss_smooth_sum = 0, 0, 0, 0, 0
        for _, ((ninput, nlabel), (ainput, alabel)) in tqdm(enumerate(zip(nloader, aloader))):
            input = torch.cat((ninput, ainput), 0).to(device)

            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
            feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, _ = model(input)  # b*32  x 2048

            loss_smooth = smooth(scores, params["lambda_1"])  # scores is the s_a^(i,j) (loss functions) (32 x 32 x 1)
            loss_sparse = sparsity(scores[:params["batch_size"], :, :].view(-1), params["lambda_2"])
            # sparsity should be with normal scores

            nlabel = nlabel[0:params["batch_size"]]
            alabel = alabel[0:params["batch_size"]]

            loss_criterion = RTFM_loss(alpha=params["alpha"], margin=params["margin"])

            loss_cls, loss_rtfm = \
                loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn)

            cost = loss_smooth + loss_sparse + loss_cls + loss_rtfm

            # viz.plot_lines('loss', cost.item())
            # viz.plot_lines('smooth loss', loss_smooth.item())
            # viz.plot_lines('sparsity loss', loss_sparse.item())
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            total_cost += cost.item()
            loss_sparse_sum += loss_sparse.item()
            loss_smooth_sum += loss_smooth.item()
            loss_cls_sum += loss_cls.item()
            loss_rtfm_sum += loss_rtfm.item()

        return total_cost, loss_cls_sum, loss_sparse_sum, loss_smooth_sum, loss_rtfm_sum


def val(nloader, aloader, model, params, device):
    with torch.set_grad_enabled(False):
        model.eval()
        total_cost, loss_cls_sum, loss_rtfm_sum, loss_sparse_sum, loss_smooth_sum = 0, 0, 0, 0, 0
        for _, ((ninput, nlabel), (ainput, alabel)) in tqdm(enumerate(zip(nloader, aloader))):
            input = torch.cat((ninput, ainput), 0).to(device)

            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
            feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, _ = model(input)  # b*32  x 2048

            loss_smooth = smooth(scores, params["lambda_1"])  # scores is the s_a^(i,j) (loss functions) (32 x 32 x 1)
            loss_sparse = sparsity(scores[:params["batch_size"], :, :].view(-1), params["lambda_2"])
            # sparsity should be with normal scores

            nlabel = nlabel[0:params["batch_size"]]
            alabel = alabel[0:params["batch_size"]]

            loss_criterion = RTFM_loss(alpha=params["alpha"], margin=params["margin"])

            loss_cls, loss_rtfm = \
                loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn)

            cost = loss_smooth + loss_sparse + loss_cls + loss_rtfm

            total_cost += cost.item()
            loss_sparse_sum += loss_sparse.item()
            loss_smooth_sum += loss_smooth.item()
            loss_cls_sum += loss_cls.item()
            loss_rtfm_sum += loss_rtfm.item()

        return total_cost, loss_cls_sum, loss_sparse_sum, loss_smooth_sum, loss_rtfm_sum






#
# class SigmoidCrossEntropyLoss(torch.nn.Module):
#     # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
#     def __init__(self):
#         super(SigmoidCrossEntropyLoss, self).__init__()
#
#     def forward(self, x, target):
#         tmp = 1 + torch.exp(- torch.abs(x))
#         return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))
import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torch import nn
from tqdm import tqdm


def sparsity(arr, lamda2):  # smoothness (14)
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss

def smooth(arr, lamda1):  # Spacity (15)
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    del arr2
    del arr

    return lamda1*loss


class ContrastiveLoss(nn.Module):  # This is used for the three different cases of (9) - (11)
    def __init__(self, margin=200.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# class SigmoidCrossEntropyLoss(nn.Module):  # Maybe (L_SCE)
#     # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
#     def __init__(self):
#         super(SigmoidCrossEntropyLoss, self).__init__()
#
#     def forward(self, x, target):
#         tmp = 1 + torch.exp(- torch.abs(x))
#         return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))

class mgfn_loss(torch.nn.Module):
    def __init__(self):
        super(mgfn_loss, self).__init__()
        # self.sigmoid = torch.nn.Sigmoid()
        self.criterion = torch.nn.BCELoss()  # Sigmoid - should be combined with BCE to create the L_SCE loss.
        self.contrastive = ContrastiveLoss()  # L_MC

    def forward(self, score_normal, score_abnormal, nlabel, alabel, nor_feamagnitude, abn_feamagnitude):
        label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal
        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()
        label = label.cuda()
        seperate = len(abn_feamagnitude) / 2

        loss_cls = self.criterion(score, label)  # L_SCE loss (Maybe) ?????

        loss_con = self.contrastive(torch.norm(abn_feamagnitude, p=1, dim=2), torch.norm(nor_feamagnitude, p=1, dim=2),
                                    1)  # try tp separate normal and abnormal  (11)

        loss_con_n = self.contrastive(torch.norm(nor_feamagnitude[int(seperate):], p=1, dim=2),
                                      torch.norm(nor_feamagnitude[:int(seperate)], p=1, dim=2),
                                      0)  # try to cluster the same class (9)

        loss_con_a = self.contrastive(torch.norm(abn_feamagnitude[int(seperate):], p=1, dim=2),
                                      torch.norm(abn_feamagnitude[:int(seperate)], p=1, dim=2),
                                      0)  # (10)

        loss_total = loss_cls + 0.001 * (0.001 * loss_con + loss_con_a + loss_con_n)
        return loss_total



def train(nloader, aloader, model, batch_size, optimizer, device, iterator = 0):
    """
    :param nloader (DataLoader): A pytorch dataloader that only loads normal videos
    :param aloader (DataLoader): A pytorch dataloader that only loads abnormal videos
    :param model:
    :param batch_size:
    :param optimizer:
    :param device:
    :return:
    """
    with torch.set_grad_enabled(True):
        model.train()
        for _, ((ninput, nlabel), (ainput, alabel)) in tqdm(enumerate(zip(nloader, aloader))):

            inp = torch.cat((ninput, ainput), 0).to(device)

            score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores = model(inp)  # b*32  x 2048
            scores = scores.view(batch_size * 32 * 2, -1)

            scores = scores.squeeze()
            abn_scores = scores[batch_size * 32:]

            nlabel = nlabel[0:batch_size]
            alabel = alabel[0:batch_size]

            loss_criterion = mgfn_loss()
            loss_sparse = sparsity(abn_scores, 8e-3)
            loss_smooth = smooth(abn_scores, 8e-4)

            cost = loss_criterion(score_normal, score_abnormal, nlabel, alabel, nor_feamagnitude, abn_feamagnitude) + \
                   loss_smooth + loss_sparse

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            iterator += 1

        return cost.item(), loss_smooth.item(), loss_sparse.item()

def val(nloader, aloader, model, batch_size, device):
    """
    :param nloader (DataLoader): A pytorch dataloader that only loads normal videos
    :param aloader (DataLoader): A pytorch dataloader that only loads abnormal videos
    :param model:
    :param batch_size:
    :param optimizer:
    :param device:
    :return:
    """
    with torch.set_grad_enabled(False):
        model.eval()
        for _, ((ninput, nlabel), (ainput, alabel)) in tqdm(enumerate(zip(nloader, aloader))):

            inp = torch.cat((ninput, ainput), 0).to(device)

            score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores = model(inp)  # b*32  x 2048
            scores = scores.view(batch_size * 32 * 2, -1)

            scores = scores.squeeze()
            abn_scores = scores[batch_size * 32:]

            nlabel = nlabel[0:batch_size]
            alabel = alabel[0:batch_size]

            loss_criterion = mgfn_loss()
            loss_sparse = sparsity(abn_scores, 8e-3)
            loss_smooth = smooth(abn_scores, 8e-4)

            cost = loss_criterion(score_normal, score_abnormal, nlabel, alabel, nor_feamagnitude, abn_feamagnitude) + \
                    loss_smooth + loss_sparse

        return cost.item()

import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torch import nn
from tqdm import tqdm


def sparsity(arr, lamda2):  # Sparcity (14)
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss

def smooth(arr, lamda1):  # smoothness (15)  input 32 x 32 x 1
    arr1 = arr[:, :-1, :]  # we need to do it only per video.  32 x 31 x 1
    arr2 = arr[:, 1:, :]   # 32 x 31 x 1
    loss = torch.sum((arr2 - arr1) ** 2)

    del arr2
    del arr1

    return lamda1 * loss


class ContrastiveLoss(nn.Module):  # This is used for the three different cases of (9) - (11)
    def __init__(self, margin=200.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        :param output1 (80 x 3):  - 3 might stem from the value of k
        :param output2 (80 x 3):
        :param label (0):
        """
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)  # 80 x 1
                            #  torch.sum((two-one)**2)**(1/2)
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
    def __init__(self, lambda3):
        super(mgfn_loss, self).__init__()
        # self.sigmoid = torch.nn.Sigmoid()
        self.lambda3 = lambda3
        self.criterion = torch.nn.BCELoss()  # Sigmoid - should be combined with BCE to create the L_SCE loss.
        self.contrastive = ContrastiveLoss()  # L_MC

    def forward(self, score_normal, score_abnormal, nlabel, alabel, nor_feamagnitude, abn_feamagnitude):
        """
        score_normal:   Is the representation scores (for the normal videos).
        score_abnormal: Is the representation scores (for the anomaly videos).
        nlabel: Is the targets of the above mentioned scores (for the normal videos).
        alabal: Is the targets of the above mentioned scores (for the abnormal videos).

        nor_feamagnitude (160, 3, 1024)  # normal videos this is the top-k Representation scores  M_n  (k = 3)  # (B*10, k, 1024)
        abn_feamagnitude (160, 3, 1024)  # abnormal videos this is the top-k Representation scores  M_a  (k = 3)  # (B*10, k, 1024)
        """

        label = torch.cat((nlabel, alabel), 0)
        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()
        label = label.cuda()
        seperate = len(abn_feamagnitude) / 2

        loss_cls = self.criterion(score, label)  # L_SCE loss (Maybe) ?????

        loss_con = self.contrastive(torch.norm(abn_feamagnitude, p=1, dim=2),
                                    torch.norm(nor_feamagnitude, p=1, dim=2),
                                    1)  # try tp separate normal and abnormal (11)

        loss_con_n = self.contrastive(torch.norm(nor_feamagnitude[int(seperate):], p=1, dim=2),
                                        torch.norm(nor_feamagnitude[:int(seperate)], p=1, dim=2),
                                        0)  # try to cluster the same class (9)

        loss_con_a = self.contrastive(torch.norm(abn_feamagnitude[int(seperate):], p=1, dim=2),
                                        torch.norm(abn_feamagnitude[:int(seperate)], p=1, dim=2),
                                        0)  # (10)

        # loss_total = loss_cls + self.lambda3 * (loss_con + loss_con_a + loss_con_n)  # Last part is MC loss?
        loss_mc = self.lambda3 * (loss_con + loss_con_a + loss_con_n)
        return loss_cls, loss_mc, loss_con, loss_con_n, loss_con_a



def train(nloader, aloader, model, params, optimizer, device, iterator = 0):
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
        loss_sum, loss_sce_sum, loss_mc_sum, loss_smooth_sum, loss_sparse_sum = 0, 0, 0, 0, 0
        loss_con_sum, loss_con_n_sum, loss_con_a_sum = 0, 0, 0
        for _, ((ninput, nlabel), (ainput, alabel)) in tqdm(enumerate(zip(nloader, aloader))):

            inp = torch.cat((ninput, ainput), 0).to(device)

            score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores = model(inp)  # b * 32 x 2048
            loss_smooth = smooth(scores, params["lambda_1"])  # scores is the s_a^(i,j) (loss functions) (32 x 32 x 1)
            loss_sparse = sparsity(scores[:params["batch_size"], :, :].view(-1), params["lambda_2"])
            # sparsity should be with normal scores

            nlabel = nlabel[0: params["batch_size"]]
            alabel = alabel[0: params["batch_size"]]

            loss_criterion = mgfn_loss(params["lambda_3"])
            loss_sce, loss_mc, loss_con, loss_con_n, loss_con_a = loss_criterion(score_normal, score_abnormal, nlabel,
                                                                                    alabel, nor_feamagnitude,
                                                                                    abn_feamagnitude)
            cost = loss_sce + loss_mc + loss_smooth + loss_sparse

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            iterator += 1

            loss_sum += cost.item()
            loss_sce_sum += loss_sce.item()
            loss_mc_sum += loss_mc.item()
            loss_smooth_sum += loss_smooth.item()
            loss_sparse_sum += loss_sparse.item()
            loss_con_sum += loss_con.item()
            loss_con_n_sum += loss_con_n.item()
            loss_con_a_sum += loss_con_a.item()
            break

        return loss_sum, loss_sce_sum, loss_mc_sum, loss_smooth_sum, loss_sparse_sum, loss_con_sum, loss_con_n_sum, loss_con_a_sum

def val(nloader, aloader, model, params, device):
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
        loss_sum, loss_sce_sum, loss_mc_sum, loss_smooth_sum, loss_sparse_sum = 0, 0, 0, 0, 0
        loss_con_sum, loss_con_n_sum, loss_con_a_sum = 0, 0, 0
        for _, ((ninput, nlabel), (ainput, alabel)) in tqdm(enumerate(zip(nloader, aloader))):

            inp = torch.cat((ninput, ainput), 0).to(device)

            score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores = model(inp)  # b*32  x 2048
            loss_smooth = smooth(scores, params["lambda_1"])  # scores is the s_a^(i,j) (loss functions)
            loss_sparse = sparsity(scores[:params["batch_size"], :, :].view(-1), params["lambda_2"])
            # sparsity should be with normal scores

            nlabel = nlabel[0: params["batch_size"]]
            alabel = alabel[0: params["batch_size"]]

            loss_criterion = mgfn_loss(params["lambda_3"])

            loss_sce, loss_mc, loss_con, loss_con_n, loss_con_a = loss_criterion(score_normal, score_abnormal, nlabel,
                                                                                    alabel, nor_feamagnitude,
                                                                                    abn_feamagnitude)

            cost = loss_sce + loss_mc + loss_smooth + loss_sparse

            loss_sum += cost.item()
            loss_sce_sum += loss_sce.item()
            loss_mc_sum += loss_mc.item()
            loss_smooth_sum += loss_smooth.item()
            loss_sparse_sum += loss_sparse.item()
            loss_con_sum += loss_con.item()
            loss_con_n_sum += loss_con_n.item()
            loss_con_a_sum += loss_con_a.item()
            break

        return loss_sum, loss_sce_sum, loss_mc_sum, loss_smooth_sum, loss_sparse_sum, loss_con_sum, loss_con_n_sum, loss_con_a_sum

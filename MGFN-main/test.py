import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
import numpy as np
from datasets.dataset import Dataset

# import matplotlib.pyplot as plt
# import option
# args = option.parse_args()
# from config import *


def test(dataloader, model, params, device):
    # plt.clf()
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        featurelen = []
        for i, inputs in tqdm(enumerate(dataloader)):

            input = inputs[0].to(device)
            input = input.permute(0, 2, 1, 3)
            _, _, _, _, logits = model(input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits.detach()
            featurelen.append(len(sig))
            pred = torch.cat((pred, sig))

        gt = np.load(params["gt"])
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)  # repeat (eg. x5 will expand each entry of a list by 5 [0, 1] => [0 x5, 1 x5]).
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        print('pr_auc : ' + str(pr_auc))
        print('rec_auc : ' + str(rec_auc))
        return rec_auc, pr_auc

if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader
    import params
    from models.mgfn import mgfn as Model

    def path_inator(params, args):
        if args.user == "marc":
            params["save_dir"] = "/home/marc/Documents/sandbox"  # where to save results + model
            params["rgb_list"] = "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFN-main/" \
                                 "UCF_list/ucf-i3d.list"

            params["gt"] = "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFN-main/" \
                           "results/ucf_gt/gt-ucf.npy"

            params["test_rgb_list"] = "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/" \
                                      "MGFN-main/UCF_list/ucf-i3d-test.list"

            params["pretrained_path"] = "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/" \
                                        "MGFN-main/results/UCF_pretrained/mgfn_ucf.pkl"

            return "/home/marc/Documents/sandbox"  # path where to wave files

        elif args.user == "cluster":
            params["save_dir"] = ""  # where to save results + model
            params["rgb_list"] = ""
            params["gt"] = "/home/cv05f23/data/UCF/test_gt/gt-ucf.npy"
            params["test_rgb_list"] = "/home/cv05f23/data/UCF/lists/ucf-i3d-test.list"
            params["pretrained_path"] = "/home/cv05f23/data/UCF/lists/UCF_pretrained/mgfn_ucf.pkl"
            return ""  # path where to wave files


    parser = argparse.ArgumentParser(description='MGFN')
    parser.add_argument("-u", '--user', default='cluster', choices=['cluster', 'marc'])  # this gives dir to data and save loc
    parser.add_argument("-p", "--params", required=True, help="Params to load")  # which parameters to load
    args = parser.parse_args()
    param = params.HYPERPARAMS[args.params]
    savepath = path_inator(param, args)

    # device = torch.device("cpu")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model()

    test_loader = DataLoader(Dataset(rgb_list=param["test_rgb_list"], datasetname="UCF", modality="RGB", seg_length=32,
                                     test_mode=True),
                             batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    print(params["pretrained_path"])
    model = model.to("cpu")
    model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(params["pretrained_path"], map_location="cpu").items()})
    model = model.to(device)
    auc = test(test_loader, model, param, device)


# --test-rgb-list /home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFN-main/UCF_list/ucf-i3d-test.list --gt /home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFN-main/results/ucf_gt/gt-ucf.npy
# --test-rgb-list /home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFN-main/UCF_list/ucf-i3d-test.list --gt /home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFN-main/results/ucf_gt/gt-ucf.npy


# dir = fr"/home/marc/Downloads/UCF_Test_ten_i3d/"
# import os
# prev_runs = os.walk(dir)
# ls = [i for i in prev_runs]
# # ls[0][2]
#
#
# dir2 = fr"/home/marc/Downloads/test/"
# import os
# prev_runs = os.walk(dir2)
# ls2 = [i for i in prev_runs]
# # ls[0][2]
#
# set(ls[0][2]).difference(set(ls2[0][2]))
# set(ls2[0][2]).difference(set(ls[0][2]))



# import matplotlib.pyplot as plt
# from sklearn.metrics import RocCurveDisplay
# import matplotlib as mpl
# mpl.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer
#
# RocCurveDisplay.from_predictions(
#     list(gt),
#     pred,
#     name=f" vs the rest",
#     color="darkorange",
# )
# plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
# plt.axis("square")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("One-vs-Rest ROC curves:\nVirginica vs (Setosa & Versicolor)")
# plt.legend()
# plt.show()
#
#
#
#
# from sklearn.metrics import PrecisionRecallDisplay
#
# display = PrecisionRecallDisplay.from_predictions([i for i in map(int, list(gt))], pred, name="LinearSVC")
# _ = display.ax_.set_title("2-class Precision-Recall curve")
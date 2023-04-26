import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
import numpy as np
import sys
sys.path.append("../..")  # adds higher directory to python modules path
sys.path.append("..")  # adds higher directory to python modules path
sys.path.insert(1, '/MGFNmain/datasets')
from MGFNmain.datasets.dataset import Dataset
import os
import neptune




# import matplotlib.pyplot as plt
# import option
# args = option.parse_args()
# from config import *


def test(dataloader, model, params, device):
    # plt.clf()
    # model = model.to("cpu")
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        # featurelen = []
        for i, (inputs, name) in tqdm(enumerate(dataloader)):
            inputs = inputs.permute(0, 2, 1, 3)
            # inputs = inputs.to("cpu")
            _, _, _, _, logits = model(inputs)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits.detach()
            # featurelen.append(len(sig))
            pred = torch.cat((pred, sig))
            torch.cuda.empty_cache()

        gt = np.load(params["gt"])
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)  # repeat (eg. x5 will expand each entry of a list by 5 [0, 1] => [0 x5, 1 x5]).
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        print('pr_auc : ' + str(pr_auc))
        print('rec_auc : ' + str(rec_auc))
        # path = params["pretrained_path"][:-4] + "_test.npy"
        # np.save(path, pred)  # save the prediction file
        return rec_auc, pr_auc

if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader
    from MGFNmain import params
    from MGFNmain.models.mgfn import mgfn as Model

    def path_inator(params, args):
        if args.user == "marc":
            params["test_rgb_list"] = "/home/marc/Documents/data/UCF/UCF_list/ucf-i3d-test.list"
            params["gt"] = "/home/marc/Documents/data/UCF/UCF_list/gt-ucf_our.npy"
            # params["pretrained_path"] = "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/" \
            #                             "MGFNmain/results/UCF_pretrained/mgfn_ucf.pkl"
            return ""

        elif args.user == "cluster":
            params["gt"] = "/home/cv05f23/data/UCF/test_gt/gt-ucf_our.npy"
            params["test_rgb_list"] = "/home/cv05f23/data/UCF/lists/ucf-i3d-test.list"
            # params["pretrained_path"] = "/home/cv05f23/data/UCF/lists/UCF_pretrained/mgfn_ucf.pkl"
            return ""  # path where to wave files


    parser = argparse.ArgumentParser(description='MGFN')
    parser.add_argument("-u", '--user', default='cluster', choices=['cluster', 'marc'])  # this gives dir to data and save loc
    parser.add_argument("-p", "--params", required=True, help="Params to load")  # which parameters to load
    args = parser.parse_args()
    param = params.HYPERPARAMS[args.params]
    savepath = path_inator(param, args)

    # device = torch.device("cpu")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    token = os.getenv('NEPTUNE_API_TOKEN')
    run = neptune.init_run(
        project="AAM/anomaly",
        api_token=token,
    )

    for i in range(1, 2):
        param["pretrained_path"] = f"/home/marc/Documents/sandbox/mgfn/nept_id_AN-110/mgfn{i}-i3d.pkl"
        # param["pretrained_path"] = f"/home/cv05f23/data/UCF/results/mgfn/nept_id_MGFN-1/mgfn{i}-i3d.pkl"
        model = Model()

        test_loader = DataLoader(Dataset(rgb_list=param["test_rgb_list"], datasetname="UCF", modality="RGB", seg_length=32,
                                            mode="test", shuffle=False),
                                    batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        model = model.to("cpu")

        di = {k.replace('module.', ''): v for k, v in torch.load(param["pretrained_path"], map_location="cpu").items()}
        # di["to_logits.weight"] = di.pop("to_logits.0.weight")
        # di["to_logits.bias"] = di.pop("to_logits.0.bias")

        model_dict = model.load_state_dict(di)
        model = model.to(device)
        rec_auc, pr_auc = test(test_loader, model, param, device)

        run["test/auc"].log(rec_auc)
        run["test/pr"].log(pr_auc)

    run.stop()


# --test-rgb-list /home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/UCF_list/ucf-i3d-test.list --gt /home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/results/ucf_gt/gt-ucf.npy
# --test-rgb-list /home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/UCF_list/ucf-i3d-test.list --gt /home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/results/ucf_gt/gt-ucf.npy


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
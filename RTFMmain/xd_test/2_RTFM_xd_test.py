import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve, f1_score, accuracy_score, precision_score, \
                            recall_score, average_precision_score
from tqdm import tqdm
import numpy as np
import sys
sys.path.append("../..")  # adds higher directory to python modules path
sys.path.append("..")  # adds higher directory to python modules path
sys.path.insert(1, '/MGFNmain/datasets')
import os
import neptune

# import matplotlib.pyplot as plt
# import option
# args = option.parse_args()


"""
This script is for creating a prediction (numpy file and upload the different metric results to neptune).
Use the params: "params_xd_def_cheat" to get the default parameteres!
"""


def test(dataloader, model, params, device):
    # plt.clf()
    # model = model.to("cpu")
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cpu()
        # featurelen = []
        for i, inputs in tqdm(enumerate(dataloader)):
            inputs = inputs.to(device)
            inputs = inputs.permute(0, 2, 1, 3)

            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
                feat_select_normal_bottom, logits, scores_nor_bottom, scores_nor_abn_bag, \
                feat_magnitudes = model(inputs=inputs)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits.detach().cpu()
            # featurelen.append(len(sig))
            pred = torch.cat((pred, sig))

        gt = np.load(params["gt"])
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)  # repeat (eg. x5 will expand each entry of a list by 5 [0, 1] => [0 x5, 1 x5]).
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        f1 = f1_score(gt, np.rint(pred))
        f1_macro = f1_score(gt, np.rint(pred), average="macro")
        acc = accuracy_score(gt, np.rint(pred))
        prec = precision_score(gt, np.rint(pred))
        recall = recall_score(gt, np.rint(pred))
        ap = average_precision_score(gt, pred)

        # print('pr_auc : ' + str(pr_auc))
        # print('rec_auc : ' + str(rec_auc))
        path = params["pretrained_path"][:-4] + "_test.npy"
        np.save(path, pred)  # save the prediction file
        return rec_auc, pr_auc, f1, f1_macro, acc, prec, recall, ap

if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader
    from RTFMmain import params
    from RTFMmain.model import Model
    from RTFMmain.dataset import Dataset

    def path_inator(params, args):
        if args.user == "marc":
            params["save_dir"] = "/home/marc/Documents/sandbox/rtfm"  # where to save results + model
            params["rgb_list"] = "/home/marc/Documents/data/xd/lists/rgb.list"
            params["test_rgb_list"] = "/home/marc/Documents/data/xd/lists/rgbtest.list"
            params["pretrained_path"] = fr"/home/marc/Documents/data/xd/results/rtfm/nept_id_RTFM-18/rftm30-i3d.pkl"  # params_xd_2
            # params["pretrained_path"] = fr"/home/marc/Documents/data/xd/results/rtfm/nept_id_RTFM-38/rftm280-i3d.pkl"  # params_xd_val
                    # params["pretrained_path"] = fr"/home/marc/Documents/data/xd/results/rtfm/nept_id_RTFM-41/rftm218-i3d.pkl"  # params_xd_val
            # params["pretrained_path"] = fr"/home/marc/Documents/data/xd/results/rtfm/nept_id_RTFM-55/rftm676-i3d.pkl"  # params_xd_val
            params["gt"] = "/home/marc/Documents/data/xd/test_gt/gt-xd_our.npy"
            return params["save_dir"]  # path where to save files

    parser = argparse.ArgumentParser(description='RFTM')
    parser.add_argument("-u", '--user', default='cluster', choices=['cluster', 'marc'])  # this gives dir to data and save loc
    parser.add_argument("-p", "--params", required=True, help="Params to load")  # which parameters to load
    parser.add_argument("-c", "--cuda", required=True, help="gpu number")
    parser.add_argument("-n", "--nept_run", required=True, help="neptune run to load")
    args = parser.parse_args()
    param = params.HYPERPARAMS[args.params]
    savepath = path_inator(param, args)

    # device = torch.device("cpu")
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    token = os.getenv('NEPTUNE_API_TOKEN')
    run = neptune.init_run(
        project="AAM/rtfm",
        api_token=token,
        with_id=f"rtfm-{args.nept_run}"
    )

    for i in range(param["max_epoch"]):
        model = Model(n_features=param["feature_size"], batch_size=param["batch_size"],
                      num_segments=param["num_segments"],
                      ncrop=param["ncrop"], drop=0, k_abn=param["k_abn"], k_nor=param["k_nor"])

        test_loader = DataLoader(Dataset(rgb_list=param["test_rgb_list"], datasetname=param["datasetname"],
                                         seg_length=param["seg_length"], mode="test", shuffle=False),
                                 batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        model = model.to("cpu")

        di = {k.replace('module.', ''): v for k, v in torch.load(param["pretrained_path"], map_location="cpu").items()}
        # di["to_logits.weight"] = di.pop("to_logits.0.weight")
        # di["to_logits.bias"] = di.pop("to_logits.0.bias")

        model_dict = model.load_state_dict(di)
        model = model.to(device)
        rec_auc, pr_auc, f1, f1_macro, acc, prec, recall, ap = test(test_loader, model, param, device)

        run["test/auc"].log(rec_auc)
        run["test/pr"].log(pr_auc)
        run["test/f1"].log(f1)
        run["test/f1_macro"].log(f1_macro)
        run["test/accuracy"].log(acc)
        run["test/precision"].log(prec)
        run["test/recall"].log(recall)
        run["test/average_precision"].log(ap)
        break

    run.stop()


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
# from sklearn.metrics import PrecisionRecallDisplay
#
# display = PrecisionRecallDisplay.from_predictions([i for i in map(int, list(gt))], pred, name="LinearSVC")
# _ = display.ax_.set_title("2-class Precision-Recall curve")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, precision_recall_curve, f1_score, accuracy_score, precision_score, \
                            recall_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib as mpl
mpl.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer

"""
This script is used for creating the plots for a specific xd model (mainly used for the pretrained model).
"""

gt = np.load("/home/marc/Documents/data/xd/test_gt/gt-xd_our.npy")
# pred = np.load(f"/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/results/XD_pretrained/mgfn_xd_test.npy")
# pred = np.load(fr"/home/marc/Documents/data/xd/results/MGFN/MGFNXD10/mgfn8-i3d_test.npy")  # params_xd_reg_11
# pred = np.load(fr"/home/marc/Documents/data/xd/results/MGFN/MGFNXD30/mgfn1-i3d_test.npy")  # params_xd_reg_22
pred = np.load(fr"/home/marc/Documents/data/xd/results/MGFN/MGFNXD113/mgfn7-i3d_test.npy")  # params_xd_reg_105

path = "/home/marc/Documents/data/xd/lists/rgbtest.list"

with open(path, 'r') as f:  # use a context manager to safely opening and closing files
    val = [line.strip() for line in f.readlines()]  # use strip to get rid of the \n at the end of string
val.sort()  # make sure to sort the values!


start = "/home/marc/Documents/data/xd/RGBTest/"
end = ".npy"
nept = True


if nept:
    import neptune
    import os

    token = os.getenv('NEPTUNE_API_TOKEN')
    run = neptune.init_run(
        project="AAM/mgfnxd",
        api_token=token,
        with_id="MGFNXD-136"
    )

leng = 0

save_path = "/home/marc/Documents/data/xd/results/MGFN/Plots_thier2/"

files = os.listdir(save_path)
if files:
    for i in files:
        os.remove(save_path + i)

for i in val:
    string = i[len(start):-len(end)]
    length = np.load(i).shape[0] * 16
    length += leng

    gt_anno = gt[leng: length]
    pred_anno = pred[leng: length]

    plt.plot(range(length - leng), gt_anno, color='b', label='gt')
    plt.plot(range(length - leng), pred_anno, color='r', label='prediction')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.title(f"{string}")
    plt.xlabel('frame number')
    plt.ylabel('anomaly score')

    # path = f"/home/marc/Documents/data/UCF/results/MGFN/Nept_id_MGFN-6/{network}/" + string[:-3] + ".jpg"
    path = save_path + string + ".jpg"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    leng = length

    if nept:
        run[f"plots/MGFN/{string}"].upload(path)

# Get the ROC curve:

RocCurveDisplay.from_predictions(
    list(gt),
    pred,
    name=f"MGFN",
    color="darkorange",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve for MGFN model:\n(Predictions on UCF data)")
plt.legend()
path = save_path + "roc" + ".jpg"
plt.savefig(path, bbox_inches='tight')
plt.close()


# Get the Precision recall curve:

display = PrecisionRecallDisplay.from_predictions([i for i in map(int, list(gt))], pred, name="MGFN")
_ = display.ax_.set_title("2-class Precision-Recall curve")
path = save_path + "pr" + ".jpg"
plt.savefig(path, bbox_inches='tight')
plt.close()


if nept:
    path = save_path + "roc" + ".jpg"
    run[f"plots/MGFN/roc"].upload(path)
    path = save_path + "pr" + ".jpg"
    run[f"plots/MGFN/pr"].upload(path)

if nept:
    run.stop()












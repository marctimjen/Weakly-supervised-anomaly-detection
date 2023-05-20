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
This script is used for creating the plots for a specific UCF model.
"""

gt = np.load("/home/marc/Documents/data/UCF/UCF_list/gt-ucf_our.npy")
# pred = np.load(f"/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/results/UCF_pretrained/mgfn_ucf_test.npy")
# pred = np.load(f"/home/marc/Documents/data/UCF/results/MGFN/nept_id_MGFN-38/mgfn95-i3d_test.npy")
pred = np.load(f"/home/marc/Documents/data/UCF/results/MGFN/nept_id_MGFN-63/mgfn50-i3d_test.npy")

path = rf"/home/marc/Documents/data/UCF/UCF_list/"
with open(path + rf"ucf-i3d-test.list", 'r') as f:  # use a context manager to safely opening and closing files
    val = [line.strip() for line in f.readlines()]  # use strip to get rid of the \n at the end of string
val.sort()  # make sure to sort the values!

start = "/home/marc/Documents/data/UCF/test/"
end = "_i3d.npy"
nept = True

if nept:
    import neptune
    import os

    token = os.getenv('NEPTUNE_API_TOKEN')
    run = neptune.init_run(
        project="AAM/mgfn",
        api_token=token,
        with_id="MGFN-75"
    )

leng = 0

for i in val:
    string = i[len(start):-len(end)] + ".mp4"
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
    path = f"/home/marc/Documents/data/UCF/results/MGFN/Plots_their2/" + string[:-3] + ".jpg"
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    leng = length



    if nept:
        run[f"plots/MGFN/{string[:-3]}"].upload(path)

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
path = f"/home/marc/Documents/data/UCF/results/MGFN/Plots_their2/" + "roc" + ".jpg"
plt.savefig(path, bbox_inches='tight')
plt.close()


# Get the Precision recall curve:

display = PrecisionRecallDisplay.from_predictions([i for i in map(int, list(gt))], pred, name="MGFN")
_ = display.ax_.set_title("2-class Precision-Recall curve")
path = f"/home/marc/Documents/data/UCF/results/MGFN/Plots_their2/" + "pr" + ".jpg"
plt.savefig(path, bbox_inches='tight')
plt.close()


if nept:
    path = f"/home/marc/Documents/data/UCF/results/MGFN/Plots_their2/" + "roc" + ".jpg"
    run[f"plots/MGFN/roc"].upload(path)
    path = f"/home/marc/Documents/data/UCF/results/MGFN/Plots_their2/" + "pr" + ".jpg"
    run[f"plots/MGFN/pr"].upload(path)

if nept:
    run.stop()












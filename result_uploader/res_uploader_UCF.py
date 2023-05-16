import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve, f1_score, accuracy_score, precision_score, \
                            recall_score, average_precision_score

import os
import neptune

"""
Is used to upload results from a model. This is specifically for the AUC score on the different classes of the UCF-
dataset.
"""

use_thresholding = True

gt_file = np.load("/home/marc/Documents/data/UCF/UCF_list/gt-ucf_our.npy")



pred_file = np.load(f"/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/results/UCF_pretrained/mgfn_ucf_test.npy")
model = "Pretrained_MGFN_thr"

auc_res = {}


classes = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "Normal_Videos_",
            "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]

mask_path = rf"/home/marc/Documents/data/UCF/UCF_list/masks/"

for i in classes:
    print(i)
    gt_class = np.load(mask_path + i + ".npy")
    gt = gt_file[np.where(gt_class)]
    pred = pred_file[np.where(gt_class)]
    if i == "Normal_Videos_":
        gt = gt + 1
        pred = 1 - pred
        f1 = f1_score(gt, pred > 0.5)
        f1_macro = f1_score(gt, pred > 0.5, average="macro")
        acc = accuracy_score(gt, pred > 0.5)
        prec = precision_score(gt, pred > 0.5)
        recall = recall_score(gt, pred > 0.5)
        ap = average_precision_score(gt, pred)
        print('f1_macro : ' + str(f1_macro))
        print('f1 : ' + str(f1))
        print('acc : ' + str(acc))
        print('prec : ' + str(prec))
        print('recall : ' + str(recall))
        print('ap : ' + str(ap))
        print()

    elif use_thresholding:
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (threshold[ix], gmeans[ix]))

        f1 = f1_score(gt, pred > threshold[ix])
        f1_macro = f1_score(gt, pred > threshold[ix], average="macro")
        acc = accuracy_score(gt, pred > threshold[ix])
        prec = precision_score(gt, pred > threshold[ix])
        recall = recall_score(gt, pred > threshold[ix])
        ap = average_precision_score(gt, pred)

        print('AUC : ' + str(rec_auc))
        print('pr_auc : ' + str(pr_auc))
        print('f1_macro : ' + str(f1_macro))
        print('f1 : ' + str(f1))
        print('acc : ' + str(acc))
        print('prec : ' + str(prec))
        print('recall : ' + str(recall))
        print('ap : ' + str(ap))
        print()
        auc_res[i] = rec_auc

    else:
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (threshold[ix], gmeans[ix]))

        f1 = f1_score(gt, pred > 0.5)
        f1_macro = f1_score(gt, pred > 0.5, average="macro")
        acc = accuracy_score(gt, pred > 0.5)
        prec = precision_score(gt, pred > 0.5)
        recall = recall_score(gt, pred > 0.5)
        ap = average_precision_score(gt, pred)

        print('AUC : ' + str(rec_auc))
        print('pr_auc : ' + str(pr_auc))
        print('f1_macro : ' + str(f1_macro))
        print('f1 : ' + str(f1))
        print('acc : ' + str(acc))
        print('prec : ' + str(prec))
        print('recall : ' + str(recall))
        print('ap : ' + str(ap))
        print()
        auc_res[i] = rec_auc


token = os.getenv('NEPTUNE_API_TOKEN')
run = neptune.init_run(
    project="AAM/results",
    api_token=token,
)

run["model"] = model
run["auc"] = auc_res
run["use_thresholding"] = use_thresholding

run.stop()

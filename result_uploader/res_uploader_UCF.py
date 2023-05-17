import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve, f1_score, accuracy_score, precision_score, \
                            recall_score, average_precision_score

import os
import neptune

"""
Is used to upload results from a model. This is specifically for the AUC score on the different classes of the UCF-
dataset.
"""



gt_file = np.load("/home/marc/Documents/data/UCF/UCF_list/gt-ucf_our.npy")
# gt_file = np.load("/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/RTFMmain/list/gt-ucf.npy")

model = "Shanghai"

if model[-3:] == "thr":
    use_thresholding = True
else:
    use_thresholding = False

model_data = {
    "Pretrained_MGFN": "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/results/UCF_pretrained/mgfn_ucf_test.npy",
    "Pretrained_MGFN_thr": "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/results/UCF_pretrained/mgfn_ucf_test.npy",
    "RTFM_UCF_22": "/home/marc/Documents/data/UCF/results/rftm/nept_id_RTFMUC-22/rftm91-i3d_test.npy",
    "RTFM_UCF_22_thr": "/home/marc/Documents/data/UCF/results/rftm/nept_id_RTFMUC-22/rftm91-i3d_test.npy",
    "RTFM_UCF_22_0": "/home/marc/Documents/data/UCF/results/rftm/nept_id_RTFMUC-22/rftm0-i3d_test.npy",
    "RTFM_UCF_22_0_thr": "/home/marc/Documents/data/UCF/results/rftm/nept_id_RTFMUC-22/rftm0-i3d_test.npy",
    "MGFN_AN46": "/home/marc/Documents/data/UCF/results/MGFN/nept_id_AN-43/mgfnfinal_test.npy",
    "MGFN_AN46_thr": "/home/marc/Documents/data/UCF/results/MGFN/nept_id_AN-43/mgfnfinal_test.npy",
    "MGFN_63_val": "/home/marc/Documents/data/UCF/results/MGFN/nept_id_MGFN-63/mgfn50-i3d_test.npy",
    "MGFN_63_val_thr": "/home/marc/Documents/data/UCF/results/MGFN/nept_id_MGFN-63/mgfn50-i3d_test.npy",
    "RTFM_38_val": "/home/marc/Documents/data/UCF/results/rftm/nept_id_RTFMUC-38/rftm168-i3d_test.npy",
    "RTFM_38_val_thr": "/home/marc/Documents/data/UCF/results/rftm/nept_id_RTFMUC-38/rftm168-i3d_test.npy",
    "random": "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/result_uploader/rando.npy",
    "random_thr": "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/result_uploader/rando.npy",
    "Shanghai": "/home/marc/Documents/data/shanghai/shanghai_best_ckpt_test.npy"
}

pred_file = np.load(model_data.get(model))


auc_res = {}
pr_res = {}
ap_res = {}
recall_res = {}
prec_res = {}
f1_macro_res = {}
f1_res = {}
accuarcy_res = {}
fdr_res = {}

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
        fdr = np.sum(pred[np.where(gt == 0)] > 0.5) / np.sum(pred > 0.5)  # FP / (FP + TP)

        print('AUC : ' + str(rec_auc))
        print('pr_auc : ' + str(pr_auc))
        print('f1_macro : ' + str(f1_macro))
        print('f1 : ' + str(f1))
        print('acc : ' + str(acc))
        print('prec : ' + str(prec))
        print('recall : ' + str(recall))
        print('ap : ' + str(ap))
        print("frd :" + str(fdr))
        print()
        auc_res[i] = rec_auc
        pr_res[i] = pr_auc
        ap_res[i] = ap
        recall_res[i] = recall
        prec_res[i] = prec
        f1_macro_res[i] = f1_macro
        f1_res[i] = f1
        accuarcy_res[i] = acc
        fdr_res[i] = fdr


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
        fdr = np.sum(pred[np.where(gt == 0)] > 0.5) / np.sum(pred > 0.5)  # FP / (FP + TP)

        print('AUC : ' + str(rec_auc))
        print('pr_auc : ' + str(pr_auc))
        print('f1_macro : ' + str(f1_macro))
        print('f1 : ' + str(f1))
        print('acc : ' + str(acc))
        print('prec : ' + str(prec))
        print('recall : ' + str(recall))
        print('ap : ' + str(ap))
        print("frd :" + str(fdr))
        print()
        auc_res[i] = rec_auc
        pr_res[i] = pr_auc
        ap_res[i] = ap
        recall_res[i] = recall
        prec_res[i] = prec
        f1_macro_res[i] = f1_macro
        f1_res[i] = f1
        accuarcy_res[i] = acc
        fdr_res[i] = fdr



token = os.getenv('NEPTUNE_API_TOKEN')
run = neptune.init_run(
    project="AAM/results",
    api_token=token,
)

run["model"] = model
run["auc"] = auc_res
run["pr"] = pr_res
run["ap"] = ap_res
run["recall"] = recall_res
run["precision"] = prec_res
run["f1_macro"] = f1_macro_res
run["f1"] = f1_res
run["accuarcy"] = accuarcy_res
run["fdr"] = fdr_res
run["use_thresholding"] = use_thresholding

run.stop()

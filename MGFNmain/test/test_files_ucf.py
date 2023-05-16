import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve, f1_score, accuracy_score, precision_score, \
                            recall_score, average_precision_score

"""
This script is for comparing two files: The Ground Truth file (GT) and a prediction file from a model. This is for the
UCF-dataset!
"""

use_thresholding = True

gt_file = np.load("/home/marc/Documents/data/UCF/UCF_list/gt-ucf_our.npy")
pred_file = np.load(f"/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/results/UCF_pretrained/mgfn_ucf_test.npy")
# pred = np.load(f"/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/results/XD_pretrained/mgfn_xd_test.npy")

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

# print("IS all anomaly scores below 20 %?", (pred < 0.2).all())
#
#
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
# plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
# plt.axis("square")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("One-vs-Rest ROC curves:\nVirginica vs (Setosa & Versicolor)")
# plt.legend()
# plt.show()
#
# # # plot the roc curve for the model
# # plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
# # plt.plot(fpr, tpr, marker='.', label='Logistic')
# # plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
# # # axis labels
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.legend()
# # # show the plot
# # plt.show()
#
#
# print("Fraction of anomaly frames:", np.sum(gt)/len(gt))
#
#
# import matplotlib.pyplot as plt
# from sklearn.metrics import RocCurveDisplay
# import matplotlib as mpl
# mpl.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer
# from sklearn.metrics import PrecisionRecallDisplay
#
# display = PrecisionRecallDisplay.from_predictions([i for i in map(int, list(gt))], pred, name="LinearSVC")
# _ = display.ax_.set_title("2-class Precision-Recall curve")
# plt.show()
#
# print("#no")
#

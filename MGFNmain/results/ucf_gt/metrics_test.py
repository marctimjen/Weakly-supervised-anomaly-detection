from sklearn.metrics import auc, roc_curve, precision_recall_curve, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, average_precision_score
import numpy as np


# Using the correct "gt"-file
# gt = np.load("gt-ucf_our.npy")

# Using the old (wrong) "gt"-file
gt = np.load("gt-ucf.npy")

# Get the preidcitons from the pretrained model
pred = np.load("MGFN_ucf_pre_test.npy")



print("Number of frames in test-set:", len(gt), "Number of frames with anomalies:", np.sum(gt))

fpr, tpr, threshold = roc_curve(list(gt), pred)
rec_auc = auc(fpr, tpr)
precision, recall, th = precision_recall_curve(list(gt), pred)
pr_auc = auc(recall, precision)
print('pr_auc : ' + str(pr_auc))
print('rec_auc : ' + str(rec_auc))

# print(roc_auc_score(gt, pred))

print("Average precision : ", average_precision_score(gt, pred))

print("f1 score : ", f1_score(gt, np.rint(pred)))
print("macro f1 score : ", f1_score(gt, np.rint(pred), average="macro"))
print("accuracy : ", accuracy_score(gt, np.rint(pred)))
print("precision : ", precision_score(gt, np.rint(pred)))
# np.sum(np.in1d(np.where(np.rint(pred))[0], np.where(gt)[0]))/len(np.where(np.rint(pred))[0])

print("recall : ", recall_score(gt, np.rint(pred)))
# np.sum(np.in1d(np.where(np.rint(pred))[0], np.where(gt)[0])) / len(np.where(gt)[0])


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
# from sklearn.metrics import PrecisionRecallDisplay
#
# display = PrecisionRecallDisplay.from_predictions([i for i in map(int, list(gt))], pred, name="LinearSVC")
# _ = display.ax_.set_title("2-class Precision-Recall curve")
#
# print("no")
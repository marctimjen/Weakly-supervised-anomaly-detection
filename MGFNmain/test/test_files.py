import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve, f1_score, accuracy_score, precision_score, \
                            recall_score, average_precision_score

"""
This script is for comparing two files: The Ground Truth file (GT) and a prediction file from a model:
"""

# gt = np.load("/home/marc/Documents/data/xd/test_gt/gt-xd_our.npy")
# pred = np.load(f"/home/marc/Documents/data/xd/results/MGFN/MGFNXD10/mgfn8-i3d_test.npy")
# pred = np.load(f"/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/results/XD_pretrained/mgfn_xd_test.npy")



model_path = f"/home/marc/Documents/data/Model_files"

gt = np.load(model_path + "/XD/gt-xd_our.npy")
# pred = np.load(model_path + "/MGFN_xd_pre_test.npy")
pred = np.load(model_path + "/MGFN_ucf_cheat_A_test.npy")
# pred = np.load(model_path + "/MGFN_ucf_cheat_B_test.npy")
# pred = np.load(model_path + "/MGFN_ucf_cheat_C_test.npy")
# pred = np.load(model_path + "/RTFM_xd_val_test.npy")
# pred = np.load(model_path + "/RTFM_xd_cheat_test.npy")


fpr, tpr, threshold = roc_curve(list(gt), pred)
rec_auc = auc(fpr, tpr)
precision, recall, th = precision_recall_curve(list(gt), pred)
pr_auc = auc(recall, precision)

# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr * (1-fpr))

ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (threshold[ix], gmeans[ix]))

# f1 = f1_score(gt, pred > threshold[ix])
# f1_macro = f1_score(gt, pred > threshold[ix], average="macro")
# acc = accuracy_score(gt, pred > threshold[ix])
# prec = precision_score(gt, pred > threshold[ix])
# recall = recall_score(gt, pred > threshold[ix])
# ap = average_precision_score(gt, pred)

f1 = f1_score(gt, pred > 0.5)
f1_macro = f1_score(gt, pred > 0.5, average="macro")
acc = accuracy_score(gt, pred > 0.5)
prec = precision_score(gt, pred > 0.5)
recall = recall_score(gt, pred > 0.5)
ap = average_precision_score(gt, pred)

print('ap : ' + str(ap))
print('rec_auc : ' + str(rec_auc))
print('f1 : ' + str(f1))
print('f1_macro : ' + str(f1_macro))
print('pr_auc : ' + str(pr_auc))
print('acc : ' + str(acc))
print('recall : ' + str(recall))
print('prec : ' + str(prec))



print("IS all anomaly scores below 20 %?", (pred < 0.2).all())


import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import matplotlib as mpl
mpl.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer

RocCurveDisplay.from_predictions(
    list(gt),
    pred,
    name=f" vs the rest",
    color="darkorange",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\nVirginica vs (Setosa & Versicolor)")
plt.legend()
plt.show()

# # plot the roc curve for the model
# plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
# plt.plot(fpr, tpr, marker='.', label='Logistic')
# plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
# # axis labels
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
# # show the plot
# plt.show()


print("Fraction of anomaly frames:", np.sum(gt)/len(gt))


import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import matplotlib as mpl
mpl.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer
from sklearn.metrics import PrecisionRecallDisplay

display = PrecisionRecallDisplay.from_predictions([i for i in map(int, list(gt))], pred, name="LinearSVC")
_ = display.ax_.set_title("2-class Precision-Recall curve")
plt.show()

print("#no")


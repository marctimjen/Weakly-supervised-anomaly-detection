import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve, f1_score, accuracy_score, precision_score, \
                            recall_score, average_precision_score

"""
This script is for comparing two files: The Ground Truth file (GT) and a prediction file from a model:
"""

gt = np.load("/home/marc/Documents/data/xd/test_gt/gt-ucf_our.npy")
pred = np.load(f"/home/marc/Documents/data/xd/results/MGFN/MGFNXD10/mgfn8-i3d_test.npy")


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

print('pr_auc : ' + str(pr_auc))
print('rec_auc : ' + str(rec_auc))
print('f1_macro : ' + str(f1_macro))
print('f1 : ' + str(f1))
print('acc : ' + str(acc))
print('prec : ' + str(prec))
print('recall : ' + str(recall))
print('ap : ' + str(ap))


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
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\nVirginica vs (Setosa & Versicolor)")
plt.legend()
plt.show()


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


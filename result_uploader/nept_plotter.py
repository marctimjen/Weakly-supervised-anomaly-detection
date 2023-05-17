import matplotlib.pyplot as plt
import os
import neptune
import pandas as pd
import matplotlib as mpl
mpl.use('Qt5Agg')

"""
This script is used to get information of the models for comparison of results of the different 
classes in the UCF crime dataset.
"""


ids = {
    # "Pretrained_RTFM": "RES-2",
    # "Pretrained_MIL": "RES-3",
    # "Pretrained_MGFN": "RES-5",
    # "Pretrained_MGFN_thr": "RES-7",
    # "RTFM_UCF_22": "RES-9",
    # "RTFM_UCF_22_0": "RES-10",
    # "MGFN_AN46": "RES-12"
}


# metric = "auc"
# metric = "pr"
# metric = "ap"
metric = "recall"
# metric = "precision"
# metric = "f1_macro"
# metric = "f1"
# metric = "accuarcy"
# metric = "fdr"

ids = ids | {
    # "MGFN_AN46": "RES-13",
    # "MGFN_AN46_thr": "RES-21",
    "Pretrained_MGFN": "RES-36",
    # "Pretrained_MGFN_thr": "RES-16",
    # "RTFM_UCF_22": "RES-17",
    # "RTFM_UCF_22_thr": "RES-18",
    # "RTFM_UCF_22_0": "RES-19",
    # "RTFM_UCF_22_0_thr": "RES-20",
    "MGFN_63_val": "RES-37",
    # "MGFN_63_val_thr": "RES-25",
    # "RTFM_38_val": "RES-26",
    # "RTFM_38_val_thr": "RES-27",
    "random": "RES-35",
    "random_thr": "RES-38",
    "Shanghai": "RES-40",
}


data = {}

token = os.getenv('NEPTUNE_API_TOKEN')

for key in ids.keys():

    run = neptune.init_run(
        project="AAM/results",
        api_token=token,
        with_id=ids.get(key)
    )

    nept_log = run[f"{metric}"].fetch()
    model = run["model"].fetch()
    thr = run["use_thresholding"]

    if model != key:
        raise KeyError("The key given in ids does not match the key on neptune")

    data[key] = {"model": model, f"{metric}": nept_log, "Threshold": thr}

    run.stop()

df = pd.DataFrame()

for key in data.keys():
    df = pd.concat([df, pd.DataFrame({key: data.get(key).get(f"{metric}")}).T])

df = df.T

ax = df.plot.bar(rot=0)

for p in ax.patches:
   height = p.get_height()
   ax.text(x=p.get_x() + p.get_width() / 2, y=height+.01,
      s="{}%".format(round(height*100, 1)),
      ha='center', rotation=45)

ax.set_xlabel("Anomaly type")
ax.set_ylabel(f"{metric}")
ax.set_title(f"{metric.upper()} of UCF anomaly classes")
plt.plot()
plt.show()

print(data)







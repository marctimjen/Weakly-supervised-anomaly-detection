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
    "Pretrained_RTFM": "RES-2",
    "Pretrained_MIL": "RES-3",
    "Pretrained_MGFN": "RES-5",
    "Pretrained_MGFN_thr": "RES-7",
}

data = {}

token = os.getenv('NEPTUNE_API_TOKEN')

for key in ids.keys():

    run = neptune.init_run(
        project="AAM/results",
        api_token=token,
        with_id=ids.get(key)
    )

    auc = run["auc"].fetch()
    model = run["model"].fetch()

    if model != key:
        raise KeyError("The key given in ids does not match the key on neptune")

    data[key] = {"model": model, "auc": auc}

    run.stop()


df = pd.DataFrame()

for key in data.keys():
    df = pd.concat([df, pd.DataFrame({key: data.get(key).get("auc")}).T])

df = df.T

ax = df.plot.bar(rot=0)

for p in ax.patches:
   height = p.get_height()
   ax.text(x=p.get_x() + p.get_width() / 2, y=height+.01,
      s="{}%".format(round(height*100, 1)),
      ha='center', rotation=45)

ax.set_xlabel("Anomaly type")
ax.set_ylabel("AUC")
ax.set_title("AUC of UCF anomaly classes")
plt.plot()

print(data)







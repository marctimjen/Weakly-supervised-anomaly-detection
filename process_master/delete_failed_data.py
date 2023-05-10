import neptune
import os

"""
This script is for deleting failed neptune runs...
"""


token = os.getenv('NEPTUNE_API_TOKEN')
run = neptune.init_run(
    project="AAM/mgfn",
    api_token=token,
    with_id=f"MGFN-56"  # 45  56
)

del run["test/auc"]

del run["test/pr"]

del run["test/f1"]

del run["test/f1_macro"]

del run["test/accuracy"]

del run["test/precision"]

del run["test/recall"]

del run["test/average_precision"]

run.stop()

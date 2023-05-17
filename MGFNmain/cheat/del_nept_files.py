import os
import neptune
import argparse

parser = argparse.ArgumentParser(description='MGFN')
parser.add_argument("-n", "--nept_run", required=True, help="neptune run to load")
args = parser.parse_args()

token = os.getenv('NEPTUNE_API_TOKEN')
run = neptune.init_run(
    project="AAM/mgfnxd",
    api_token=token,
    with_id=f"MGFNXD-{args.nept_run}"
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
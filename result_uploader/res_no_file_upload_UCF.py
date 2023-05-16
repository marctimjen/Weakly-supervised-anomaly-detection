import neptune
import os

"""
This script is used to upload information of the RTFM model and MIL model for comparison of results of the different 
classes in the UCF crime dataset.
"""


token = os.getenv('NEPTUNE_API_TOKEN')
run = neptune.init_run(
    project="AAM/results",
    api_token=token,
)

auc_scores_RTFM = {"Abuse": 0.559,
                   "Arrest": 0.591,
                   "Arson": 0.653,
                   "Assault": 0.707,
                   "Burglary": 0.701,
                   "Explosion": 0.452,
                   "Fighting": 0.700,
                   "RoadAccidents": 0.559,
                   "Robbery": 0.698,
                   "Shooting": 0.736,
                   "Shoplifting": 0.708,
                   "Stealing": 0.757,
                   "Vandalism": 0.663}

auc_scores_MIL = {"Abuse": 0.703,
                   "Arrest": 0.591,
                   "Arson": 0.618,
                   "Assault": 0.546,
                   "Burglary": 0.595,
                   "Explosion": 0.487,
                   "Fighting": 0.703,
                   "RoadAccidents": 0.598,
                   "Robbery": 0.633,
                   "Shooting": 0.790,
                   "Shoplifting": 0.494,
                   "Stealing": 0.751,
                   "Vandalism": 0.563}


# run["model"] = "Pretrained_RTFM"
# run["auc"] = auc_scores_RTFM

run["model"] = "Pretrained_MIL"
run["auc"] = auc_scores_MIL

run.stop()
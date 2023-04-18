original = {"margin": 100, "w_decay": 0.0005, "lr": [0.001]*15000, "batch_size": 32}  # hyper params

dataset_params = {"rgb_list": "data/ucf_tencrop_1d/ucf-i3d.list",
                  "val_rgb_list": "data/ucf_tencrop_1d/ucf-i3d-val.list",
                  "test_rgb_list": "data/ucf_tencrop_1d/ucf-i3d-test.list",
                  "datasetname": "UCF",
                  "dataset": "UCF",
                  "modality": "RGB",
                  }

test = {"gt": "data/ucf_tencrop_1d/gt-ucf.npy"}

main = {"max_epoch": 1000,
        "pretrained_ckpt": None,
        "model_name": "rftm"
        }

params_def = {
    "feat_extractor": "i3d",
    "feature_size": 2048,
    "gpus": [0],
    "workers": 4,
    "num_classes": 1,
    "plot_freq": 10,
}


HYPERPARAMS = {
    'params_def': params_def | main | test | dataset_params | original,

}

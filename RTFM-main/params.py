original = {"margin": 100, "w_decay": 0.0005, "lr": 0.001, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001}  # hyper params

dataset_params = {"rgb_list": "data/ucf_tencrop_1d/ucf-i3d.list",
                  "val_rgb_list": "data/ucf_tencrop_1d/ucf-i3d-val.list",
                  "test_rgb_list": "data/ucf_tencrop_1d/ucf-i3d-test.list",
                  "datasetname": "UCF",
                  "dataset": "UCF",
                  "modality": "RGB",
                  "UCF_train_len": 1449,
                  "UCF_val_len": 161,
                  "UCF_test_len": 290
                  }

test = {"gt": "data/ucf_tencrop_1d/gt-ucf.npy"}

main = {"max_epoch": 1000,
        "pretrained_ckpt": False,
        "model_name": "rftm"
        }

params_def = {
    "feat_extractor": "i3d",
    "feature_size": 2048,
    "gpus": 0,
    "workers": 4,
    "num_classes": 1,
    "plot_freq": 10,
}


HYPERPARAMS = {
    'params_def': params_def | main | test | dataset_params | original,

}

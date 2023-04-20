original = {"T": 32, "P": 10, "alpha": 0.1, "k": 3, "lambda_1": 1, "lambda_2": 1, "lambda_3": 0.001,
            "w_decay": 0.0005, "lr": [0.001]*15000, "batch_size": 8}  # hyper params

dataset_params = {"seg_length": 32,
                  "add_mag_info": False,  # Do not quite know what this does...
                  "rgb_list": "data/ucf_tencrop_1d/ucf-i3d.list",
                  "val_rgb_list": "data/ucf_tencrop_1d/ucf-i3d-val.list",
                  "test_rgb_list": "data/ucf_tencrop_1d/ucf-i3d-test.list",
                  "datasetname": "UCF",
                  "dataset": "UCF",
                  "modality": "RGB",
                  "UCF_train_len": 1449,
                  "UCF_val_len": 161,
                  "UCF_test_len": 290
                  }

mgfn_params = {"depths1": 3,
               "depths2": 3,
               "depths3": 2,
               "mgfn_type1": "gb",
               "mgfn_type2": "fb",
               "mgfn_type3": "fb",
               "dropout_rate": 0.7,
               "mag_ratio": 0.1
               }

test = {"gt": "data/ucf_tencrop_1d/gt-ucf.npy"}

main = {"max_epoch": 100,
        "pretrained_ckpt": None,
        "model_name": "mgfn"
        }

params_def = {
    "feat_extractor": "i3d",
    "feature_size": 2048,
    "hiddensize": 512,
    "comment": "mgfn",
    "local_con": "static",
    "head_K": 4,
    "gpus": [0],
    "workers": 0,
    "num_classes": 2,
    "plot_freq": 10,
}


HYPERPARAMS = {
    'params_def': params_def | main | test | mgfn_params | dataset_params | original,

}

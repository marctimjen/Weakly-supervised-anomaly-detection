original = {"T": 32, "P": 10, "alpha": 0.1, "k": 3, "lambda_1": 1, "lambda_2": 1, "lambda_3": 0.001,
            "w_decay": 0.0005, "lr": 0.001, "batch_size": 8}  # hyper params

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

main = {"max_epoch": 250,  # normally 1000
        "pretrained_ckpt": False,
        "model_name": "mgfn"
        }

params_def = {
    "feat_extractor": "i3d",
    "feature_size": 2048,
    "hiddensize": 512,
    "comment": "mgfn",
    "local_con": "static",
    "head_K": 4,
    "gpus": 0,
    "workers": 0,
    "num_classes": 2,
    "plot_freq": 10,
}

params_1 = {"T": 32, "P": 10, "alpha": 0.1, "k": 3, "lambda_1": 0.5, "lambda_2": 1, "lambda_3": 0.5,
            "w_decay": 0.0005, "lr": 0.001, "batch_size": 8}  # hyper params

paths = {
    "rgb_list": "/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFN-main/UCF_list/ucf-i3d-train.list",
    "val_rgb_list": "/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFN-main/UCF_list/ucf-i3d-val.list",
    "test_rgb_list": "/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFN-main/UCF_list/ucf-i3d-test.list",
    "gt": "data/ucf_tencrop_1d/gt-ucf.npy"
}

paths_cheat = {
    "rgb_list": "/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFN-main/UCF_list/ucf-i3d-train_cheat.list",
    "val_rgb_list": "/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFN-main/UCF_list/ucf-i3d-test.list",
    "UCF_train_cheat_len": 1610,
}


HYPERPARAMS = {
    'params_def': params_def | main | mgfn_params | dataset_params | original | paths,
    'params_1': params_def | main | mgfn_params | dataset_params | params_1 | paths,
    'params_cheat': params_def | main | mgfn_params | dataset_params | original | paths_cheat,
}






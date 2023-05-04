original = {"margin": 100, "w_decay": 0.0005, "lr": 0.001, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001}  # hyper params

dataset_params = {"datasetname": "UCF",
                  "dataset": "UCF",
                  "modality": "RGB",
                  "UCF_train_len": 1449,
                  "UCF_val_len": 161,
                  "UCF_test_len": 290,
                  "seg_length": 32,
                  "add_mag_info": False,
                  }

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

paths = {
    "rgb_list": "/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFNmain/UCF_list/ucf-i3d-train.list",
    "val_rgb_list": "/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFNmain/UCF_list/ucf-i3d-val.list",
    "test_rgb_list": "/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFNmain/UCF_list/ucf-i3d-test.list",
    "gt": "data/ucf_tencrop_1d/gt-ucf.npy"
}


HYPERPARAMS = {
    'params_def': params_def | main | dataset_params | original | paths,
}


params_def_xd = {
    "feat_extractor": "i3d",
    "feature_size": 1024,
    "gpus": 0,
    "workers": 4,
    "num_classes": 1,
    "plot_freq": 10,
}

main_xd = {"max_epoch": 1000,
        "pretrained_ckpt": False,
        "model_name": "rftm"
        }

dataset_params_xd = {"seg_length": 32,
                  "add_mag_info": False,  # Do not quite know what this does...
                  "datasetname": "XD",
                  "dataset": "XD",
                  "modality": "RGB",
                  "xd_train_len": 3954,
                  "xd_test_len": 800
                  }

original_xd = {"margin": 100, "w_decay": 0.0005, "lr": 0.001, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001}  # hyper params

paths_xd = {
    "rgb_list": "/home/cv05f23/data/XD/lists/rgb.list",
    "test_rgb_list": "/home/cv05f23/data/XD/lists/rgbtest.list",
    "gt": "/home/cv05f23/data/XD/test_gt/gt-ucf_our.npy"
}

HYPERPARAMS |= {
    'params_xd': params_def_xd | main_xd | dataset_params_xd | original_xd | paths_xd,
}
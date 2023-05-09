original = {"margin": 100, "w_decay": 0.0005, "lr": 0.001, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001, "drop": 0.7, "k_abn": 3, "k_nor": 3}  # hyper params

dataset_params = {"datasetname": "UCF",
                  "dataset": "UCF",
                  "modality": "RGB",
                  "UCF_train_len": 1449,
                  "UCF_val_len": 161,
                  "UCF_test_len": 290,
                  "seg_length": 32,
                  "add_mag_info": False,
                  "ncrop": 10,
                  "num_segments": 32,
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
                  "xd_test_len": 800,
                  "ncrop": 5,
                  "num_segments":32,
                  }

original_xd = {"margin": 100, "w_decay": 0.0005, "lr": 0.001, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001, "drop": 0.7, "k_abn": 3, "k_nor": 3}  # hyper params

paths_xd = {
    "rgb_list": "/home/cv05f23/data/XD/lists/rgb.list",
    "test_rgb_list": "/home/cv05f23/data/XD/lists/rgbtest.list",
    "gt": "/home/cv05f23/data/XD/test_gt/gt-ucf_our.npy"
}

HYPERPARAMS |= {
    'params_xd': params_def_xd | main_xd | dataset_params_xd | original_xd | paths_xd,
}

main_xd_reg = {"max_epoch": 100,
        "pretrained_ckpt": False,
        "model_name": "rftm"
        }

original_xd1 = {"margin": 100, "w_decay": 0.0005, "lr": 0.001, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001, "drop": 0.7, "k_abn": 3, "k_nor": 3}  # hyper params

original_xd2 = {"margin": 100, "w_decay": 0.005, "lr": 0.001, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001, "drop": 0.7, "k_abn": 3, "k_nor": 3}  # hyper params

original_xd3 = {"margin": 100, "w_decay": 0.0025, "lr": 0.001, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001, "drop": 0.7, "k_abn": 3, "k_nor": 3}  # hyper params

original_xd4 = {"margin": 100, "w_decay": 0.0005, "lr": 0.001, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001, "drop": 0.0, "k_abn": 3, "k_nor": 3}  # hyper params

original_xd5 = {"margin": 100, "w_decay": 0.005, "lr": 0.001, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001, "drop": 0.0, "k_abn": 3, "k_nor": 3}  # hyper params

original_xd6 = {"margin": 100, "w_decay": 0.0025, "lr": 0.001, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001, "drop": 0.0, "k_abn": 3, "k_nor": 3}  # hyper params

original_xd7 = {"margin": 100, "w_decay": 0.0005, "lr": 0.001, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001, "drop": 0.35, "k_abn": 3, "k_nor": 3}  # hyper params

original_xd8 = {"margin": 100, "w_decay": 0.005, "lr": 0.001, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001, "drop": 0.7, "k_abn": 3, "k_nor": 3}  # hyper params

original_xd9 = {"margin": 100, "w_decay": 0.0025, "lr": 0.001, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001, "drop": 0.35, "k_abn": 3, "k_nor": 3}  # hyper params


original_xd10 = {"margin": 100, "w_decay": 0.0005, "lr": 0.01, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001, "drop": 0.7, "k_abn": 3, "k_nor": 3}  # hyper params

original_xd11 = {"margin": 100, "w_decay": 0.005, "lr": 0.01, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001, "drop": 0.7, "k_abn": 3, "k_nor": 3}  # hyper params

original_xd12 = {"margin": 100, "w_decay": 0.0025, "lr": 0.01, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001, "drop": 0.7, "k_abn": 3, "k_nor": 3}  # hyper params


original_xd13 = {"margin": 100, "w_decay": 0.0005, "lr": 0.0005, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001, "drop": 0.7, "k_abn": 3, "k_nor": 3}  # hyper params

original_xd14 = {"margin": 100, "w_decay": 0.005, "lr": 0.0005, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001, "drop": 0.7, "k_abn": 3, "k_nor": 3}  # hyper params

original_xd15 = {"margin": 100, "w_decay": 0.0025, "lr": 0.0005, "batch_size": 32, "lambda_2": 8e-3,
            "lambda_1": 8e-4, "alpha": 0.0001, "drop": 0.7, "k_abn": 3, "k_nor": 3}  # hyper params


HYPERPARAMS |= {
    'params_xd_1': params_def_xd | main_xd_reg | dataset_params_xd | original_xd1 | paths_xd,
    'params_xd_2': params_def_xd | main_xd_reg | dataset_params_xd | original_xd2 | paths_xd,
    'params_xd_3': params_def_xd | main_xd_reg | dataset_params_xd | original_xd3 | paths_xd,
    'params_xd_4': params_def_xd | main_xd_reg | dataset_params_xd | original_xd4 | paths_xd,
    'params_xd_5': params_def_xd | main_xd_reg | dataset_params_xd | original_xd5 | paths_xd,
    'params_xd_6': params_def_xd | main_xd_reg | dataset_params_xd | original_xd6 | paths_xd,
    'params_xd_7': params_def_xd | main_xd_reg | dataset_params_xd | original_xd7 | paths_xd,
    'params_xd_8': params_def_xd | main_xd_reg | dataset_params_xd | original_xd8 | paths_xd,
    'params_xd_9': params_def_xd | main_xd_reg | dataset_params_xd | original_xd9 | paths_xd,
    'params_xd_10': params_def_xd | main_xd_reg | dataset_params_xd | original_xd10 | paths_xd,
    'params_xd_11': params_def_xd | main_xd_reg | dataset_params_xd | original_xd11 | paths_xd,
    'params_xd_12': params_def_xd | main_xd_reg | dataset_params_xd | original_xd12 | paths_xd,
    'params_xd_13': params_def_xd | main_xd_reg | dataset_params_xd | original_xd13 | paths_xd,
    'params_xd_14': params_def_xd | main_xd_reg | dataset_params_xd | original_xd14 | paths_xd,
    'params_xd_15': params_def_xd | main_xd_reg | dataset_params_xd | original_xd15 | paths_xd,
}


paths_cheat = {
    "rgb_list": "/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFNmain/UCF_list/ucf-i3d-train_cheat.list",
    "test_rgb_list": "/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFNmain/UCF_list/ucf-i3d-test.list",
    "UCF_train_cheat_len": 1610,
}

paths |= paths_cheat

HYPERPARAMS |= {
    'params_ucf_1': params_def | main_xd_reg | dataset_params | original_xd1 | paths,
    'params_ucf_2': params_def | main_xd_reg | dataset_params | original_xd2 | paths,
    'params_ucf_3': params_def | main_xd_reg | dataset_params | original_xd3 | paths,
    'params_ucf_4': params_def | main_xd_reg | dataset_params | original_xd4 | paths,
    'params_ucf_5': params_def | main_xd_reg | dataset_params | original_xd5 | paths,
    'params_ucf_6': params_def | main_xd_reg | dataset_params | original_xd6 | paths,
    'params_ucf_7': params_def | main_xd_reg | dataset_params | original_xd7 | paths,
    'params_ucf_8': params_def | main_xd_reg | dataset_params | original_xd8 | paths,
    'params_ucf_9': params_def | main_xd_reg | dataset_params | original_xd9 | paths,
    'params_ucf_10': params_def | main_xd_reg | dataset_params | original_xd10 | paths,
    'params_ucf_11': params_def | main_xd_reg | dataset_params | original_xd11 | paths,
    'params_ucf_12': params_def | main_xd_reg | dataset_params | original_xd12 | paths,
    'params_ucf_13': params_def | main_xd_reg | dataset_params | original_xd13 | paths,
    'params_ucf_14': params_def | main_xd_reg | dataset_params | original_xd14 | paths,
    'params_ucf_15': params_def | main_xd_reg | dataset_params | original_xd15 | paths,
}

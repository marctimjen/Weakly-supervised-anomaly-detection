import copy

original = {"T": 32, "P": 10, "alpha": 0.1, "k": 3, "lambda_1": 1, "lambda_2": 1, "lambda_3": 0.001,
            "w_decay": 0.0005, "lr": 0.001, "batch_size": 16}  # hyper params

dataset_params = {"seg_length": 32,
                  "add_mag_info": False,  # Do not quite know what this does...
                  "datasetname": "UCF",
                  "dataset": "UCF",
                  "modality": "RGB",
                  "UCF_train_len": 1449,
                  "UCF_val_len": 161,
                  "UCF_test_len": 290
                  }

mgfn_params = {"dims1": 64,
               "dims2": 128,
               "dims3": 1024,
               "depths1": 3,
               "depths2": 3,
               "depths3": 2,
               "mgfn_type1": "gb",
               "mgfn_type2": "fb",
               "mgfn_type3": "fb",
               "channels": 2048,
               "ff_repe": 4,
               "dim_head": 64,
               "dropout": 0.0,
               "attention_dropout": 0.0,
               "dropout_rate": 0.7,
               "mag_ratio": 0.1
               }

main = {"max_epoch": 1000,  # normally 1000
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
    "rgb_list": "/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFNmain/UCF_list/ucf-i3d-train.list",
    "val_rgb_list": "/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFNmain/UCF_list/ucf-i3d-val.list",
    "test_rgb_list": "/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFNmain/UCF_list/ucf-i3d-test.list",
    "gt": "data/ucf_tencrop_1d/gt-ucf.npy"
}

HYPERPARAMS = {
    'params_def': params_def | main | mgfn_params | dataset_params | original | paths,
    'params_1': params_def | main | mgfn_params | dataset_params | params_1 | paths,
}


# Try with regulization of the network to aviod overfitting
paths_cheat = {
    "rgb_list": "/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFNmain/UCF_list/ucf-i3d-train_cheat.list",
    "val_rgb_list": "/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFNmain/UCF_list/ucf-i3d-test.list",
    "UCF_train_cheat_len": 1610,
}

mgfn_params_reg = copy.deepcopy(mgfn_params)

mgfn_params_reg |= {
                    "dropout_rate": 0.4,
                    "dropout": 0.4,
                    "attention_dropout": 0.4
                    }


# original_reg |= {"w_decay": 0.0005}  # more weight decay = L2 norm :).


HYPERPARAMS |= {
    'params_cheat': params_def | main | mgfn_params | dataset_params | original | paths_cheat,
    'params_cheat_reg': params_def | main | mgfn_params_reg | dataset_params | original | paths_cheat
}

# Make a series of tests with more and less regulization:

# original_reg = original
original_reg = copy.deepcopy(original)
original_reg |= {"w_decay": 0.005}  # more weight decay = L2 norm :).

mgfn_params_2 = copy.deepcopy(mgfn_params)
mgfn_params_3 = copy.deepcopy(mgfn_params)
mgfn_params_4 = copy.deepcopy(mgfn_params)
mgfn_params_5 = copy.deepcopy(mgfn_params)
mgfn_params_6 = copy.deepcopy(mgfn_params)


# Try different hyper-params of dropout
mgfn_params_2 |= {
    "dropout": 0.0,
    "attention_dropout": 0.5,
    "dropout_rate": 0.7,
                    }

mgfn_params_3 |= {
    "dropout": 0.5,
    "attention_dropout": 0.0,
    "dropout_rate": 0.7,
                    }

mgfn_params_4 |= {
    "dropout": 0.5,
    "attention_dropout": 0.5,
    "dropout_rate": 0.0,
                    }

mgfn_params_5 |= {
    "dropout": 0.5,
    "attention_dropout": 0.5,
    "dropout_rate": 0.5,
                    }

mgfn_params_6 |= {
    "dropout": 0.5,
    "attention_dropout": 0.5,
    "dropout_rate": 0.7,
                    }

# Try with smaller network:
mgfn_params_net = copy.deepcopy(mgfn_params)

mgfn_params_net |= {
               "depths1": 3,
               "depths2": 2,
               "depths3": 1,
               "mgfn_type1": "gb",
               "mgfn_type2": "fb",
               "mgfn_type3": "fb",
               }

mgfn_params_net2 = copy.deepcopy(mgfn_params_5)
mgfn_params_net2 |= {
               "depths1": 3,
               "depths2": 2,
               "depths3": 1,
               "mgfn_type1": "gb",
               "mgfn_type2": "fb",
               "mgfn_type3": "fb",
               }

main_reg = copy.deepcopy(main)
main_reg |= {"max_epoch": 100}

HYPERPARAMS |= {
    'params_cheat_1': params_def | main_reg | mgfn_params | dataset_params | original_reg | paths_cheat,  # try only change weight decay
    'params_cheat_2': params_def | main_reg | mgfn_params_2 | dataset_params | original | paths_cheat,
    'params_cheat_3': params_def | main_reg | mgfn_params_3 | dataset_params | original | paths_cheat,
    'params_cheat_4': params_def | main_reg | mgfn_params_4 | dataset_params | original | paths_cheat,
    'params_cheat_5': params_def | main_reg | mgfn_params_5 | dataset_params | original | paths_cheat,
    'params_cheat_6': params_def | main_reg | mgfn_params_6 | dataset_params | original | paths_cheat,
    'params_cheat_7': params_def | main_reg | mgfn_params_2 | dataset_params | original_reg | paths_cheat,
    'params_cheat_8': params_def | main_reg | mgfn_params_3 | dataset_params | original_reg | paths_cheat,
    'params_cheat_9': params_def | main_reg | mgfn_params_4 | dataset_params | original_reg | paths_cheat,
    'params_cheat_10': params_def | main_reg | mgfn_params_5 | dataset_params | original_reg | paths_cheat,
    'params_cheat_11': params_def | main_reg | mgfn_params_6 | dataset_params | original_reg | paths_cheat,

    'params_cheat_12': params_def | main_reg | mgfn_params_net | dataset_params | original | paths_cheat,
    'params_cheat_13': params_def | main_reg | mgfn_params_net | dataset_params | original_reg | paths_cheat,

    'params_cheat_14': params_def | main_reg | mgfn_params_net2 | dataset_params | original | paths_cheat,
    'params_cheat_15': params_def | main_reg | mgfn_params_net2 | dataset_params | original_reg | paths_cheat,
}



####################################### XD-violence data:

original_xd = {"T": 32, "P": 10, "alpha": 0.1, "k": 3, "lambda_1": 1, "lambda_2": 1, "lambda_3": 0.001,
            "w_decay": 0.0005, "lr": 0.001, "batch_size": 16}  # hyper params

dataset_params_xd = {"seg_length": 32,
                  "add_mag_info": False,  # Do not quite know what this does...
                  "datasetname": "XD",
                  "dataset": "XD",
                  "modality": "RGB",
                  "UCF_train_len": 1449,
                  "UCF_val_len": 161,
                  "UCF_test_len": 290
                  }

mgfn_params_xd = {"dims1": 64,
               "dims2": 128,
               "dims3": 1024,
               "depths1": 3,
               "depths2": 3,
               "depths3": 2,
               "mgfn_type1": "gb",
               "mgfn_type2": "fb",
               "mgfn_type3": "fb",
               "channels": 1024,
               "ff_repe": 4,
               "dim_head": 64,
               "dropout": 0.0,
               "attention_dropout": 0.0,
               "dropout_rate": 0.7,
               "mag_ratio": 0.1
               }

main_xd = {"max_epoch": 1000,  # normally 1000
            "pretrained_ckpt": False,
            "model_name": "mgfn"
            }

params_def_xd = {
    "feat_extractor": "i3d",
    "feature_size": 1024,
    "hiddensize": 512,
    "comment": "mgfn",
    "local_con": "static",
    "head_K": 4,
    "gpus": 0,
    "workers": 0,
    "num_classes": 2,
    "plot_freq": 10,
}

paths_xd = {
    "rgb_list": "/home/cv05f23/data/XD/i3d-features/lists/rgb.list",
    "test_rgb_list": "/home/cv05f23/data/XD/i3d-features/lists/rgbtest.list",
    "gt": ""
}

HYPERPARAMS |= {"params_xd": params_def_xd | main_xd | mgfn_params_xd | dataset_params_xd | original_xd | paths_xd}






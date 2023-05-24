from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import sys
sys.path.append("../..")  # adds higher directory to python modules path
sys.path.append("..")  # adds higher directory to python modules path
from RTFMmain.model import Model
from RTFMmain.dataset import Dataset
from RTFMmain.train import train, val

from tqdm import tqdm
import argparse
# from torch.multiprocessing import set_start_method

import datetime
import RTFMmain.params as params
import os
import neptune

# import option
# args = option.parse_args()

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

"""
This script is used to upload the performance of the RTFM model on the UCF dataset. 
This will give us the loss of a specific model (used for the pretrained model for now).

Use the params: "params_def_cheat" to get the default parameteres!
"""

def save_config(save_path, nept_id, params):
    path = save_path + '/nept_id_' + nept_id + "/"
    os.makedirs(path, exist_ok=True)
    f = open(path + f"config_{datetime.datetime.now()}.txt", 'w')
    for key in params.keys():
        f.write(f'{key}: {params[key]}')
        f.write('\n')
    return path

def path_inator(params, args):
    if args.user == "marc":
        params["save_dir"] = "/home/marc/Documents/sandbox/rtfm"  # where to save results + model
        params["rgb_list"] = "/home/marc/Documents/data/UCF/UCF_list/ucf-i3d-train_cheat.list"
        # params["rgb_list"] = "/home/marc/Documents/data/UCF/UCF_list/ucf-i3d-train.list"

        params["test_rgb_list"] = "/home/marc/Documents/data/UCF/UCF_list/ucf-i3d-test.list"

        params["pretrained_ckpt"] = "/home/marc/Documents/data/UCF/results/rftm/nept_id_RTFMUC-22/rftm91-i3d.pkl"  # params_ucf_1

        # params["pretrained_ckpt"] = "/home/marc/Documents/data/UCF/results/rftm/nept_id_RTFMUC-38/rftm771-i3d.pkl"  # params_def
        return params["save_dir"]  # path where to save files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RTFM')
    parser.add_argument("-u", '--user', default='cluster', choices=['cluster', 'marc'])  # this gives dir to data and save loc
    parser.add_argument("-p", "--params", required=True, help="Params to load")  # which parameters to load
    parser.add_argument("-c", "--cuda", required=True, help="gpu number")
    args = parser.parse_args()

    token = os.getenv('NEPTUNE_API_TOKEN')
    run = neptune.init_run(
        project="AAM/rtfmucf",
        api_token=token,
    )
    run_id = run["sys/id"].fetch()

    param = params.HYPERPARAMS[args.params]
    param["user"] = args.user
    param["params"] = args.params
    run["params"] = param
    run["model"] = "RTFM"

    save_path = path_inator(param, args)
    save_path = save_config(save_path, run_id, params=param)

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_nloader = DataLoader(Dataset(rgb_list=param["rgb_list"], datasetname=param["datasetname"],
                                        seg_length=param["seg_length"], mode="train", is_normal=True, shuffle=True),
                                batch_size=param["batch_size"], shuffle=False, num_workers=param["workers"],
                                pin_memory=False, drop_last=True)

    train_aloader = DataLoader(Dataset(rgb_list=param["rgb_list"], datasetname=param["datasetname"],
                                        seg_length=param["seg_length"], mode="train", is_normal=False, shuffle=True),
                                batch_size=param["batch_size"], shuffle=False, num_workers=param["workers"],
                                pin_memory=False, drop_last=True)

    val_nloader = DataLoader(Dataset(rgb_list=param["test_rgb_list"], datasetname=param["datasetname"],
                                        seg_length=param["seg_length"], mode="val", is_normal=True, shuffle=True),
                                batch_size=param["batch_size"], shuffle=False, num_workers=param["workers"],
                                pin_memory=False, drop_last=True)

    val_aloader = DataLoader(Dataset(rgb_list=param["test_rgb_list"], datasetname=param["datasetname"],
                                        seg_length=param["seg_length"], mode="val", is_normal=False, shuffle=True),
                                batch_size=param["batch_size"], shuffle=False, num_workers=param["workers"],
                                pin_memory=False, drop_last=True)


    model = Model(n_features=param["feature_size"], batch_size=param["batch_size"], num_segments=param["num_segments"],
                    ncrop=param["ncrop"], drop=param["drop"], k_abn=param["k_abn"], k_nor=param["k_nor"])

    # params["pretrained_ckpt"] = "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/" \
    #                             "MGFNmain/results/UCF_pretrained/mgfn_ucf.pkl"

    if param["pretrained_ckpt"]:
        di = {k.replace('module.', ''): v for k, v in torch.load(param["pretrained_ckpt"], map_location="cpu").items()}
        # di["to_logits.weight"] = di.pop("to_logits.0.weight")
        # di["to_logits.bias"] = di.pop("to_logits.0.bias")
        model_dict = model.load_state_dict(di)
        print("pretrained loaded")

    model = model.to(device)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    optimizer = optim.Adam(model.parameters(), lr=param["lr"], weight_decay=param["w_decay"])

    iterator = 0
    for step in tqdm(range(0, param["max_epoch"]), total=param["max_epoch"], dynamic_ncols=True):

        total_cost, loss_cls_sum, loss_sparse_sum, loss_smooth_sum, loss_rtfm_sum \
            = val(train_nloader, train_aloader,  model, param, device)

        run["train/loss"].log(total_cost/(param["UCF_train_cheat_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["train/loss_cls"].log(loss_cls_sum/(param["UCF_train_cheat_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["train/loss_sparse"].log(loss_sparse_sum/(param["UCF_train_cheat_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["train/loss_smooth"].log(loss_smooth_sum/(param["UCF_train_cheat_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["train/loss_rtfm"].log(loss_rtfm_sum/(param["UCF_train_cheat_len"]//(param["batch_size"]*2)*param["batch_size"]))

        val_loss, loss_cls_sum, loss_sparse_sum, loss_smooth_sum, loss_rtfm_sum \
            = val(val_nloader, val_aloader, model, param, device)

        run["test/loss"].log(val_loss/(param["UCF_test_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["test/loss_cls"].log(loss_cls_sum/(param["UCF_test_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["test/loss_sparse"].log(loss_sparse_sum/(param["UCF_test_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["test/loss_smooth"].log(loss_smooth_sum/(param["UCF_test_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["test/loss_rtfm"].log(loss_rtfm_sum/(param["UCF_test_len"]//(param["batch_size"]*2)*param["batch_size"]))
        # break

    run.stop()





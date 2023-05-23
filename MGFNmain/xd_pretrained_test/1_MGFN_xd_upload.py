from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import sys
sys.path.append("../..")  # adds higher directory to python modules path
sys.path.append("..")  # adds higher directory to python modules path

from MGFNmain.utils.utils import save_best_record
from tqdm import tqdm
import argparse
# from torch.multiprocessing import set_start_method
from MGFNmain.models.mgfn import mgfn
from MGFNmain.datasets.dataset import Dataset
from MGFNmain.train import train, val
import datetime
import MGFNmain.params as params
import os
import neptune

# import option
# args = option.parse_args()
# from config import *
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

"""
This script is used to upload the performance of the MGFN model on the UCF dataset. 
This will give us the loss of a specific model (used for the pretrained model for now).

Use the params: "params_xd_def_cheat" to get the default parameteres (for the model used for the xd dataset)!
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
        params["save_dir"] = "/home/marc/Documents/sandbox/mgfn"  # where to save results + model
        params["rgb_list"] = "/home/marc/Documents/data/xd/lists/rgb.list"
        params["test_rgb_list"] = "/home/marc/Documents/data/xd/lists/rgbtest.list"
        # params["pretrained_ckpt"] = fr"/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/results/XD_pretrained/mgfn_xd.pkl"  # params_xd_def_cheat
        # params["pretrained_ckpt"] = fr"/home/marc/Documents/data/xd/results/MGFN/MGFNXD10/mgfn8-i3d.pkl"  # params_xd_reg_11
        params["pretrained_ckpt"] = fr"/home/marc/Documents/data/xd/results/MGFN/MGFNXD30/mgfn1-i3d.pkl"  # params_xd_reg_22
        # params["pretrained_ckpt"] = fr"/home/marc/Documents/data/xd/results/MGFN/MGFNXD113/mgfn7-i3d.pkl"  # params_xd_reg_105
        return params["save_dir"]  # path where to save files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MGFN')
    parser.add_argument("-u", '--user', default='cluster', choices=['cluster', 'marc'])  # this gives dir to data and save loc
    parser.add_argument("-p", "--params", required=True, help="Params to load")  # which parameters to load
    parser.add_argument("-c", "--cuda", required=True, help="gpu number")
    args = parser.parse_args()

    token = os.getenv('NEPTUNE_API_TOKEN')
    run = neptune.init_run(
        project="AAM/mgfnxd",
        api_token=token,
    )
    run_id = run["sys/id"].fetch()

    param = params.HYPERPARAMS[args.params]
    param["user"] = args.user
    param["params"] = args.params
    run["params"] = param
    run["model"] = "MGFN"

    save_path = path_inator(param, args)
    save_path = save_config(save_path, run_id, params=param)

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_nloader = DataLoader(Dataset(rgb_list=param["rgb_list"], datasetname=param["datasetname"],
                                        modality=param["modality"], seg_length=param["seg_length"],
                                        add_mag_info=param["add_mag_info"], mode="train", is_normal=True, shuffle=True),
                                batch_size=param["batch_size"], shuffle=False, num_workers=param["workers"],
                                pin_memory=False, drop_last=True)

    train_aloader = DataLoader(Dataset(rgb_list=param["rgb_list"], datasetname=param["datasetname"],
                                        modality=param["modality"], seg_length=param["seg_length"],
                                        add_mag_info=param["add_mag_info"], mode="train", is_normal=False, shuffle=True),
                                batch_size=param["batch_size"], shuffle=False, num_workers=param["workers"],
                                pin_memory=False, drop_last=True)

    val_nloader = DataLoader(Dataset(rgb_list=param["test_rgb_list"], datasetname=param["datasetname"],
                                        modality=param["modality"], seg_length=param["seg_length"],
                                        add_mag_info=param["add_mag_info"], mode="val", is_normal=True, shuffle=True),
                                batch_size=param["batch_size"], shuffle=False, num_workers=param["workers"],
                                pin_memory=False, drop_last=True)

    val_aloader = DataLoader(Dataset(rgb_list=param["test_rgb_list"], datasetname=param["datasetname"],
                                        modality=param["modality"], seg_length=param["seg_length"],
                                        add_mag_info=param["add_mag_info"], mode="val", is_normal=False, shuffle=True),
                                batch_size=param["batch_size"], shuffle=False, num_workers=param["workers"],
                                pin_memory=False, drop_last=True)

    model = mgfn(dims=(param["dims1"], param["dims2"], param["dims3"]),
                    depths=(param["depths1"], param["depths2"], param["depths3"]),
                    mgfn_types=(param["mgfn_type1"], param["mgfn_type2"], param["mgfn_type3"]),
                    channels=param["channels"], ff_repe=param["ff_repe"], dim_head=param["dim_head"],
                    batch_size=param["batch_size"], dropout_rate=param["dropout_rate"],
                    mag_ratio=param["mag_ratio"], dropout=param["dropout"],
                    attention_dropout=param["attention_dropout"], device=device,
                    )

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

        loss_sum, sce_sum, mc_sum, smooth_sum, sparse_sum, con_sum, con_n_sum, con_a_sum\
            = val(nloader=train_nloader, aloader=train_aloader, model=model, params=param, device=device)
        
        run["train/loss"].log(loss_sum/(param["xd_train_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["train/loss_sce"].log(sce_sum/(param["xd_train_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["train/loss_mc"].log(mc_sum/(param["xd_train_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["train/loss_smooth"].log(smooth_sum/(param["xd_train_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["train/loss_sparse"].log(sparse_sum/(param["xd_train_len"]//(param["batch_size"]*2)*param["batch_size"]))

        run["train/loss_con_sum"].log(con_sum/(param["xd_train_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["train/loss_con_n_sum"].log(con_n_sum/(param["xd_train_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["train/loss_con_a_sum"].log(con_a_sum/(param["xd_train_len"]//(param["batch_size"]*2)*param["batch_size"]))

        # if step % 1 == 0 and step > 0:
        loss, sce_sum, mc_sum, smooth_sum, sparse_sum, con_sum, con_n_sum, con_a_sum = \
            val(nloader=val_nloader, aloader=val_aloader, model=model, params=param, device=device)

        run["test/loss"].log(
            loss / (param["xd_test_len"] // (param["batch_size"] * 2) * param["batch_size"]))
        run["test/loss_sce"].log(
            sce_sum / (param["xd_test_len"] // (param["batch_size"] * 2) * param["batch_size"]))
        run["test/loss_mc"].log(
            mc_sum / (param["xd_test_len"] // (param["batch_size"] * 2) * param["batch_size"]))
        run["test/loss_smooth"].log(
            smooth_sum / (param["xd_test_len"] // (param["batch_size"] * 2) * param["batch_size"]))
        run["test/loss_sparse"].log(
            sparse_sum / (param["xd_test_len"] // (param["batch_size"] * 2) * param["batch_size"]))

        run["test/loss_con_sum"].log(
            con_sum / (param["xd_test_len"] // (param["batch_size"] * 2) * param["batch_size"]))
        run["test/loss_con_n_sum"].log(
            con_n_sum / (param["xd_test_len"] // (param["batch_size"] * 2) * param["batch_size"]))
        run["test/loss_con_a_sum"].log(
            con_a_sum / (param["xd_test_len"] // (param["batch_size"] * 2) * param["batch_size"]))

        # break

    run.stop()





from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import argparse
from model import Model
from tqdm import tqdm
import params
import datetime
import os
from utils import save_best_record
from dataset import Dataset
from train import train, val
import neptune

# from config import *
# from test_10crop import test
# from utils import Visualizer

# viz = Visualizer(env='shanghai tech 10 crop', use_incoming_socket=False)
viz = 0
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
        params["save_dir"] = "/home/marc/Documents/sandbox/rftm"  # where to save results + model
        params["rgb_list"] = "/home/marc/Documents/data/UCF/UCF_list/ucf-i3d-train.list"
        params["val_rgb_list"] = "/home/marc/Documents/data/UCF/UCF_list/ucf-i3d-val.list"
        params["test_rgb_list"] = "/home/marc/Documents/data/UCF/UCF_list/ucf-i3d-test.list"
        params["gt"] = "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/" \
                        "results/ucf_gt/gt-ucf.npy"
        return params["save_dir"]  # path where to save files

    elif args.user == "cluster":
        params["save_dir"] = "/home/cv05f23/data/UCF/results/rftm"  # where to save results + model
        return params["save_dir"]  # path where to save files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RFTM')
    parser.add_argument("-u", '--user', default='cluster',
                        choices=['cluster', 'marc'])  # this gives dir to data and save loc
    parser.add_argument("-p", "--params", required=True, help="Params to load")  # which parameters to load
    args = parser.parse_args()

    token = os.getenv('NEPTUNE_API_TOKEN')
    run = neptune.init_run(
        project="AAM/rftm",
        api_token=token,
    )
    run_id = run["sys/id"].fetch()

    param = params.HYPERPARAMS[args.params]
    param["user"] = args.user
    param["params"] = args.params
    run["params"] = param
    run["model"] = "RFTM"

    save_path = path_inator(param, args)
    save_path = save_config(save_path, run_id, params=param)

    train_nloader = DataLoader(Dataset(dataset=param["dataset"], rgb_list=param["rgb_list"], mode="train",
                                        is_normal=True),
                                batch_size=param["batch_size"], shuffle=True,
                                num_workers=0, pin_memory=False, drop_last=True)

    train_aloader = DataLoader(Dataset(dataset=param["dataset"], rgb_list=param["rgb_list"], mode="train",
                                        is_normal=False),
                                batch_size=param["batch_size"], shuffle=True,
                                num_workers=0, pin_memory=False, drop_last=True)

    val_nloader = DataLoader(Dataset(dataset=param["dataset"], rgb_list=param["val_rgb_list"], mode="val",
                                        is_normal=True),
                                batch_size=param["batch_size"], shuffle=True,
                                num_workers=0, pin_memory=False, drop_last=True)

    val_aloader = DataLoader(Dataset(dataset=param["dataset"], rgb_list=param["val_rgb_list"], mode="val",
                                        is_normal=False),
                                batch_size=param["batch_size"], shuffle=True,
                                num_workers=0, pin_memory=False, drop_last=True)


    model = Model(n_features=param["feature_size"], batch_size=param["batch_size"])

    for name, value in model.named_parameters():
        print(name)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("batch-size", param["batch_size"])

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=param["lr"], weight_decay=param["w_decay"])

    val_info = {"epoch": [], "val_loss": []}
    best_val_loss = float("inf")

    for step in tqdm(range(1, param["max_epoch"] + 1), total=param["max_epoch"], dynamic_ncols=True):
        # if step > 1 and param["lr"][step - 1] != param["lr"][step - 2]:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = param["lr"][step - 1]

        # if (step - 1) % len(train_nloader) == 0:
        #     loadern_iter = iter(train_nloader)
        #
        # if (step - 1) % len(train_aloader) == 0:
        #     loadera_iter = iter(train_aloader)

        total_cost, loss_cls_sum, loss_sparse_sum, loss_smooth_sum, loss_rtfm_sum \
            = train(train_nloader, train_aloader, model, param, optimizer, viz, device)

        run["train/loss"].log(total_cost/(param["UCF_train_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["train/loss_cls"].log(loss_cls_sum/(param["UCF_train_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["train/loss_sparse"].log(loss_sparse_sum/(param["UCF_train_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["train/loss_smooth"].log(loss_smooth_sum/(param["UCF_train_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["train/loss_rtfm"].log(loss_rtfm_sum/(param["UCF_train_len"]//(param["batch_size"]*2)*param["batch_size"]))

        val_loss, loss_cls_sum, loss_sparse_sum, loss_smooth_sum, loss_rtfm_sum \
            = val(val_nloader, val_aloader, model, param, device)

        run["val/loss"].log(val_loss/(param["UCF_val_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["val/loss_cls"].log(loss_cls_sum/(param["UCF_val_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["val/loss_sparse"].log(loss_sparse_sum/(param["UCF_val_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["val/loss_smooth"].log(loss_smooth_sum/(param["UCF_val_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["val/loss_rtfm"].log(loss_rtfm_sum/(param["UCF_val_len"]//(param["batch_size"]*2)*param["batch_size"]))

        val_info["epoch"].append(step)
        val_info["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path + param["model_name"] + '{}-i3d.pkl'.format(step))
            save_best_record(val_info, os.path.join(save_path, '{}-step-val_loss.txt'.format(step)))

    torch.save(model.state_dict(), save_path + param["model_name"] + 'final.pkl')

    run.stop()

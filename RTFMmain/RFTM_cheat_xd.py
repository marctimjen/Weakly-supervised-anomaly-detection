from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import argparse
from model import Model
from tqdm import tqdm
import params
import os
from utils import save_best_record
from train import train, val
import neptune
import sys
sys.path.append("../..")  # adds higher directory to python modules path
sys.path.append("..")  # adds higher directory to python modules path
from MGFNmain.config import path_inator, save_config
from dataset import Dataset

# from config import *
# from test_10crop import test
# from utils import Visualizer

# viz = Visualizer(env='shanghai tech 10 crop', use_incoming_socket=False)
viz = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RFTM')
    parser.add_argument("-u", '--user', default='cluster',
                        choices=['cluster', 'marc'])  # this gives dir to data and save loc
    parser.add_argument("-p", "--params", required=True, help="Params to load")  # which parameters to load
    parser.add_argument("-c", "--cuda", required=True, help="gpu number")
    args = parser.parse_args()

    token = os.getenv('NEPTUNE_API_TOKEN')
    run = neptune.init_run(
        project="AAM/RTFM",
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

    for name, value in model.named_parameters():
        print(name)

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("batch-size", param["batch_size"])

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=param["lr"], weight_decay=param["w_decay"])

    # val_info = {"epoch": [], "val_loss": []}
    # best_val_loss = float("inf")

    for step in tqdm(range(param["max_epoch"]), total=param["max_epoch"], dynamic_ncols=True):
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

        run["train/loss"].log(total_cost/(param["xd_train_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["train/loss_cls"].log(loss_cls_sum/(param["xd_train_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["train/loss_sparse"].log(loss_sparse_sum/(param["xd_train_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["train/loss_smooth"].log(loss_smooth_sum/(param["xd_train_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["train/loss_rtfm"].log(loss_rtfm_sum/(param["xd_train_len"]//(param["batch_size"]*2)*param["batch_size"]))

        val_loss, loss_cls_sum, loss_sparse_sum, loss_smooth_sum, loss_rtfm_sum \
            = val(val_nloader, val_aloader, model, param, device)

        run["test/loss"].log(val_loss/(param["xd_test_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["test/loss_cls"].log(loss_cls_sum/(param["xd_test_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["test/loss_sparse"].log(loss_sparse_sum/(param["xd_test_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["test/loss_smooth"].log(loss_smooth_sum/(param["xd_test_len"]//(param["batch_size"]*2)*param["batch_size"]))
        run["test/loss_rtfm"].log(loss_rtfm_sum/(param["xd_test_len"]//(param["batch_size"]*2)*param["batch_size"]))

        # val_info["epoch"].append(step)
        # val_info["val_loss"].append(val_loss)

        torch.save(model.state_dict(), save_path + param["model_name"] + '{}-i3d.pkl'.format(step))
        # save_best_record(val_info, os.path.join(save_path, '{}-step-val_loss.txt'.format(step)))

    torch.save(model.state_dict(), save_path + param["model_name"] + 'final.pkl')

    run.stop()


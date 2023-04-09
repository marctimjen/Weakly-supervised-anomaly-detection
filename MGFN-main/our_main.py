from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils.utils import save_best_record
from tqdm import tqdm
import argparse
from torch.multiprocessing import set_start_method
from models.mgfn import mgfn
from datasets.dataset import Dataset
from train import train
from test import test
import datetime
import params
import os
import neptune

# import option
# args = option.parse_args()
# from config import *
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
        params["save_dir"] = "/home/marc/Documents/sandbox"  # where to save results + model
        params["rgb_list"] = "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFN-main/" \
                             "UCF_list/ucf-i3d.list"

        params["gt"] = "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFN-main/" \
                       "results/ucf_gt/gt-ucf.npy"

        params["test_rgb_list"] = "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/" \
                                  "MGFN-main/UCF_list/ucf-i3d-test.list"

        return "/home/marc/Documents/sandbox"  # path where to wave files

    elif args.user == "cluster":
        params["save_dir"] = ""  # where to save results + model
        params["rgb_list"] = ""
        params["gt"] = ""
        params["test_rgb_list"] = ""
        return ""  # path where to wave files

# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MGFN')
    parser.add_argument("-u", '--user', default='cluster', choices=['cluster', 'marc'])  # this gives dir to data and save loc
    parser.add_argument("-p", "--params", required=True, help="Params to load")  # which parameters to load
    args = parser.parse_args()

    token = os.getenv('NEPTUNE_API_TOKEN')
    run = neptune.init_run(
        project="AAM/anomaly",
        api_token=token,
    )
    run_id = run["sys/id"].fetch()

    param = params.HYPERPARAMS[args.params]
    param["user"] = args.user
    param["params"] = args.params
    run["params"] = param

    save_path = path_inator(param, args)
    save_path = save_config(save_path, run_id, params=param)

    # args = option.parse_args()
    # config = Config(args)
    # TODO: Get randomly samples from train loaders + make sure they output more than 100 samples.
    train_nloader = DataLoader(Dataset(rgb_list=param["rgb_list"], datasetname=param["datasetname"],
                                       modality=param["modality"], seg_length=param["seg_length"],
                                       add_mag_info=param["add_mag_info"], test_mode=False, is_normal=True),
                               batch_size=param["batch_size"], shuffle=False, num_workers=param["workers"],
                               pin_memory=False, drop_last=True)

    train_aloader = DataLoader(Dataset(rgb_list=param["rgb_list"], datasetname=param["datasetname"],
                                       modality=param["modality"], seg_length=param["seg_length"],
                                       add_mag_info=param["add_mag_info"], test_mode=False, is_normal=False),
                               batch_size=param["batch_size"], shuffle=False, num_workers=param["workers"],
                               pin_memory=False, drop_last=True)

    test_loader = DataLoader(Dataset(rgb_list=param["test_rgb_list"], datasetname=param["datasetname"],
                                       modality=param["modality"], seg_length=param["seg_length"],
                                       add_mag_info=param["add_mag_info"], test_mode=True),
                             batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    model = mgfn(depths=(param["depths1"], param["depths2"], param["depths3"]),
                 mgfn_types=(param["mgfn_type1"], param["mgfn_type2"], param["mgfn_type3"]),
                 batch_size=param["batch_size"],
                 dropout_rate=param["dropout_rate"],
                 mag_ratio=param["mag_ratio"]
                 )

    if param["pretrained_ckpt"] is not None:
        model_ckpt = torch.load(param["pretrained_ckpt"])
        model.load_state_dict(model_ckpt)
        print("pretrained loaded")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")

    model = model.to(device)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    optimizer = optim.Adam(model.parameters(), lr=param["lr"][0], weight_decay=param["w_decay"])
    test_info = {"epoch": [], "test_AUC": [], "test_PR": []}

    best_AUC = -1
    best_PR = -1

    # for name, value in model.named_parameters():
    #     print(name)

    iterator = 0
    for step in tqdm(range(1, param["max_epoch"] + 1), total=param["max_epoch"], dynamic_ncols=True):

        # for step in range(1, args.max_epoch + 1):
        if step > 1 and param["lr"][step - 1] != param["lr"][step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param["lr"][step - 1]

        cost, loss_smooth, loss_sparse = train(train_nloader, train_aloader, model, param["batch_size"], optimizer,
                                               device, iterator)

        # log_writer.add_scalar('loss_contrastive', cost, step)
        if step % 1 == 0 and step > 0:
            auc, pr_auc = test(test_loader, model, param, device)  # TODO: ugly
            # log_writer.add_scalar('auc-roc', auc, step)
            # log_writer.add_scalar('pr_auc', pr_auc, step)

            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)
            test_info["test_PR"].append(pr_auc)
            if param["datasetname"] == 'XD':
                if test_info["test_PR"][-1] > best_PR:
                    best_PR = test_info["test_PR"][-1]
                    torch.save(model.state_dict(), save_path + param["model_name"] + f'{step}-i3d.pkl')
                    save_best_record(test_info, os.path.join(save_path, f'{step}-step-AUC.txt'))
            else:
                if test_info["test_AUC"][-1] > best_AUC:
                    best_AUC = test_info["test_AUC"][-1]
                    torch.save(model.state_dict(), save_path + param["model_name"] + f'{step}-i3d.pkl')
                    save_best_record(test_info, os.path.join(save_path, f'{step}-step-AUC.txt'))

    torch.save(model.state_dict(), save_path + param["model_name"] + 'final.pkl')

    run.stop()

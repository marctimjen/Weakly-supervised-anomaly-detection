import os
import datetime

class Config(object):
    def __init__(self, args):
        self.lr = eval(args.lr)
        self.lr_str = args.lr

    def __str__(self):
        attrs = vars(self)
        attr_lst = sorted(attrs.keys())
        return '\n'.join("- %s: %s" % (item, attrs[item]) for item in attr_lst if item != 'lr')

def path_inator(params, args):

    dataset_name = params.get("datasetname")
    if dataset_name.lower() == "xd":
        if args.user == "marc":
            params["save_dir"] = "/home/marc/Documents/sandbox"  # where to save results + model
            params["rgb_list"] = "/home/marc/Documents/data/xd/lists/rgb.list"
            params["test_rgb_list"] = "/home/marc/Documents/data/xd/lists/rgbtest.list"
            params["gt"] = "/home/marc/Documents/data/xd/test_gt/gt-ucf_our.npy"

            params["pretrained_path"] = "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/" \
                                        + "MGFNmain/results/XD_pretrained/mgfn_xd.pkl"

            return "/home/marc/Documents/sandbox"  # path where to wave files

        elif args.user == "cluster":
            params["save_dir"] = f"/home/cv05f23/data/xd/results/{params['model_name']}"
            # params["save_dir"] = ""  # where to save results + model
            # params["rgb_list"] = ""
            # params["gt"] = ""
            # params["test_rgb_list"] = ""
            # params["pretrained_path"] = ""
            return params["save_dir"]  # path where to wave files

    elif dataset_name.lower() == "ucf":
        if args.user == "marc":
            params["save_dir"] = f"/home/marc/Documents/data/xd/results/{params['model_name']}"  # where to save results + model
            params["rgb_list"] = "/home/marc/Documents/data/UCF/UCF_list/ucf-i3d-train.list"
            params["test_rgb_val"] = "/home/marc/Documents/data/UCF/UCF_list/ucf-i3d-val.list"
            params["val_rgb_list"] = "/home/marc/Documents/data/UCF/UCF_list/ucf-i3d-val.list"
            params["test_rgb_list"] = "/home/marc/Documents/data/UCF/UCF_list/ucf-i3d-test.list"

            # params["gt"] = "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/" \
            #                 "results/ucf_gt/gt-ucf.npy"
            params["gt"] = "/home/marc/Documents/data/UCF/UCF_list/gt-ucf_our.npy"

            # params["pretrained_path"] = "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/" \
            #                             "MGFNmain/results/UCF_pretrained/mgfn_ucf.pkl"

            params["pretrained_path"] = "/home/marc/Documents/data/UCF/results/MGFN/Nept_id_MGFN-8/mgfn156-i3d.pkl"
            return params["save_dir"]  # path where to wave files

        elif args.user == "cluster":
            params["save_dir"] = f"/home/cv05f23/data/UCF/results/{params['model_name']}"  # where to save results + model
            params["gt"] = "/home/cv05f23/data/UCF/test_gt/gt-ucf_our.npy"
            params["pretrained_path"] = "/home/cv05f23/data/UCF/lists/UCF_pretrained/mgfn_ucf.pkl"
            return params["save_dir"]  # path where to wave files
    else:
        raise ValueError("Dataset should be UCF og XD")


def save_config(save_path, nept_id, params):
    path = save_path + '/nept_id_' + nept_id + "/"
    os.makedirs(path, exist_ok=True)
    f = open(path + f"config_{datetime.datetime.now()}.txt", 'w')
    for key in params.keys():
        f.write(f'{key}: {params[key]}')
        f.write('\n')

    return path
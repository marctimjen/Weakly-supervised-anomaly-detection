import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
"""
This script is used for creating predictions plots for different UCF model.
"""


with open("/home/marc/Documents/data/s3r/ucf-crime_cheat.pickle", "rb") as f:
    s3r_path_ucf_cheat = pkl.load(f)

with open("/home/marc/Documents/data/s3r/ucf-crime_pretrain.pickle", "rb") as f:
    s3r_path_ucf_pre = pkl.load(f)

with open("/home/marc/Documents/data/s3r/ucf-crime_validation.pickle", "rb") as f:
    s3r_path_ucf_val = pkl.load(f)

# network = "mgfn7"
# pred = np.load(f"/home/marc/Documents/data/UCF/results/MGFN/Nept_id_MGFN-6/{network}-i3d_test.npy")
# pred2 = np.load("/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/results/UCF_pretrained/mgfn_ucf_test.npy")

gt = np.load("/home/marc/Documents/data/UCF/UCF_list/gt-ucf_our.npy")

pred_dict = {
    "RTFM_val": np.load("/home/marc/Documents/data/UCF/results/rftm/nept_id_RTFMUC-38/rftm771-i3d_test.npy"),
    "RTFM_cheat": np.load("/home/marc/Documents/data/UCF/results/rftm/nept_id_RTFMUC-22/rftm91-i3d_test.npy"),
    "MGFN_val": np.load("/home/marc/Documents/data/UCF/results/MGFN/nept_id_MGFN-63/mgfn50-i3d_test.npy"),
    "MGFN_cheat": np.load("/home/marc/Documents/data/UCF/results/MGFN/nept_id_MGFN-38/mgfn95-i3d_test.npy"),
    "MGFN_pre": np.load("/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/results/UCF_pretrained/mgfn_ucf_test.npy"),
    "s3r_cheat": s3r_path_ucf_cheat,
    "s3r_pre": s3r_path_ucf_pre,
    "s3r_val": s3r_path_ucf_val
}


path = rf"/home/marc/Documents/data/UCF/UCF_list/"
with open(path + rf"ucf-i3d-test.list", 'r') as f:  # use a context manager to safely opening and closing files
    val = [line.strip() for line in f.readlines()]  # use strip to get rid of the \n at the end of string
val.sort()  # make sure to sort the values!


start = "/home/marc/Documents/data/UCF/test/"
end = "_i3d.npy"

nept = True
if nept:
    import neptune
    import os
    token = os.getenv('NEPTUNE_API_TOKEN')
    run = neptune.init_run(
        project="AAM/results",
        api_token=token,
    )
leng = 0

c_idx = 0
colors = ['tab:blue', 'tab:cyan', 'tab:green', 'tab:olive', 'tab:gray', 'tab:purple', 'tab:pink', 'tab:orange', "tab:brown"]


for i in val:
    string = "UCF " + i[len(start):-len(end)]
    length = np.load(i).shape[0] * 16
    length += leng
    gt_anno = gt[leng: length]
    fig = plt.figure()
    plt.plot(range(length - leng), gt_anno, color='tab:red', label='gt')
    c_idx = 0
    for key in pred_dict.keys():
        if key[:3] == "s3r":
            pred = np.repeat(pred_dict.get(key)[i[len(start):-len(end)]].squeeze(), 16)
        else:
            pred = pred_dict.get(key)[leng: length]

        plt.plot(range(length - leng), pred, color=colors[c_idx], label=f'{key}')
        c_idx += 1

    plt.ylim(-0.05, 1.05)
    # plt.legend()
    plt.tight_layout(pad=5.0)
    fig.set_size_inches(10, 3)
    plt.title(f"{string[:-5]}")
    plt.xlabel('frame number')
    plt.ylabel('anomaly score')

    # path = f"/home/marc/Documents/data/UCF/results/MGFN/Nept_id_MGFN-6/{network}/" + string[:-3] + ".jpg"
    path = f"/home/marc/Documents/data/UCF/results/MGFN/Plots_their2/" + string[:-5] + ".jpg"
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    leng = length



    if nept:
        run[f"test_pics/{string[:-3]}"].upload(path)

if nept:
    run.stop()

















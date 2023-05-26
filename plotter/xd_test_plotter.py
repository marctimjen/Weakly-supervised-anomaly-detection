import numpy as np
import matplotlib.pyplot as plt

import pickle as pkl

network = "mgfn7"

gt = np.load("/home/marc/Documents/data/xd/test_gt/gt-xd_our.npy")
#pred = np.load(f"/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/results/XD_pretrained/mgfn_xd_test.npy")
# pred = np.load(f"/home/marc/Documents/data/xd/results/MGFN/MGFNXD10/mgfn8-i3d_test.npy")
# pred = np.load("/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/results/UCF_pretrained/mgfn_ucf_test.npy")

with open("/home/marc/Documents/data/s3r/xd-violence_cheat.pickle", "rb") as f:
    s3r_path_xd_cheat = pkl.load(f)

with open("/home/marc/Documents/data/s3r/xd-violence_valid.pickle", "rb") as f:
    s3r_path_xd_val = pkl.load(f)

# with open("/home/marc/Documents/data/s3r/ucf-crime_validation.pickle", "rb") as f:
#     s3r_path_xd_val = pkl.load(f)

pred_dict = {
    "RTFM_val": np.load("/home/marc/Documents/data/xd/results/rtfm/nept_id_RTFM-55/rftm676-i3d_test.npy"),
    "RTFM_cheat": np.load("/home/marc/Documents/data/xd/results/rtfm/nept_id_RTFM-18/rftm30-i3d_test.npy"),
    "MGFN_val": np.load("/home/marc/Documents/data/xd/results/MGFN/MGFNXD149/mgfn56-i3d_test.npy"),
    "MGFN_cheat": np.load("/home/marc/Documents/data/xd/results/MGFN/MGFNXD59/mgfn60-i3d_test.npy"),
    "MGFN_cheat_A": np.load("/home/marc/Documents/data/xd/results/MGFN/MGFNXD10/mgfn8-i3d_test.npy"),
    "MGFN_pre": np.load("/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/results/XD_pretrained/mgfn_xd_test.npy"),
    "s3r_cheat": s3r_path_xd_cheat,
    "s3r_val": s3r_path_xd_val
}


path = "/home/marc/Documents/data/xd/lists/rgbtest.list"
with open(path, 'r') as f:  # use a context manager to safely opening and closing files
    val = [line.strip() for line in f.readlines()]  # use strip to get rid of the \n at the end of string
val = sorted(val)  # make sure to sort the values!

start = "/home/marc/Documents/data/xd/RGBTest/"
end = ".npy"

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
colors = ['tab:blue', 'tab:cyan', 'tab:green', 'tab:olive', 'tab:gray', "tab:brown", 'tab:purple', 'tab:pink', 'tab:orange']

for i in val:
    string = "XD " + i[len(start):-len(end)]
    length = np.load(i).shape[0] * 16
    length += leng

    gt_anno = gt[leng: length]
    fig = plt.figure()
    plt.plot(range(length - leng), gt_anno, color='r', label='gt')
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
    plt.title(f"{string}")
    plt.xlabel('frame number')
    plt.ylabel('anomaly score')

    # path = f"/home/marc/Documents/data/UCF/results/MGFN/Nept_id_MGFN-6/{network}/" + string[:-3] + ".jpg"
    path = f"/home/marc/Documents/data/UCF/results/MGFN/Plots_their/" + string[:-3] + ".jpg"
    # path = f"/home/marc/Documents/data/xd/results/MGFN/MGFNXD10/mgfn8/" + string + ".jpg"
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    leng = length



    if nept:
        run[f"test_pics/{string}"].upload(path)

if nept:
    run.stop()

















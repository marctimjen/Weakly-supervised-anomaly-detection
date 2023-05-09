import numpy as np
import matplotlib.pyplot as plt

network = "mgfn7"

gt = np.load("/home/marc/Documents/data/xd/test_gt/gt-ucf_our.npy")
#pred = np.load(f"/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/results/XD_pretrained/mgfn_xd_test.npy")

pred = np.load(f"/home/marc/Documents/data/xd/results/MGFN/MGFNXD10/mgfn8-i3d_test.npy")

# pred = np.load("/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/results/UCF_pretrained/mgfn_ucf_test.npy")

path = "/home/marc/Documents/data/xd/lists/rgbtest.list"
with open(path, 'r') as f:  # use a context manager to safely opening and closing files
    val = [line.strip() for line in f.readlines()]  # use strip to get rid of the \n at the end of string
val.sort()  # make sure to sort the values!


start = "/home/marc/Documents/data/xd/RGBTest/"
end = ".npy"

nept = False

if nept:
    import neptune
    import os

    token = os.getenv('NEPTUNE_API_TOKEN')
    run = neptune.init_run(
        project="AAM/mgfn",
        api_token=token,
        with_id="MGFN-6"
    )


leng = 0

for i in val:
    string = i[len(start):-len(end)]
    length = np.load(i).shape[0] * 16
    length += leng

    gt_anno = gt[leng: length]
    pred_anno = pred[leng: length]

    plt.plot(range(length - leng), gt_anno, color='b', label='gt')
    plt.plot(range(length - leng), pred_anno, color='r', label='prediction')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.title(f"{string}")
    plt.xlabel('frame number')
    plt.ylabel('anomaly score')

    # path = f"/home/marc/Documents/data/UCF/results/MGFN/Nept_id_MGFN-6/{network}/" + string[:-3] + ".jpg"
    # path = f"/home/marc/Documents/data/UCF/results/MGFN/Plots_their/" + string[:-3] + ".jpg"
    path = f"/home/marc/Documents/data/xd/results/MGFN/MGFNXD10/mgfn8/" + string + ".jpg"
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    leng = length



    if nept:
        run[f"test_pics/{network}/{string}"].upload(path)

if nept:
    run.stop()

















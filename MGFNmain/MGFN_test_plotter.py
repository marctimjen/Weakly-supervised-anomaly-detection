import numpy as np
import matplotlib.pyplot as plt


gt = np.load("/home/marc/Documents/data/UCF/UCF_list/gt-ucf_our.npy")
# pred = np.load("/home/marc/Documents/data/UCF/results/MGFN/nept_id_AN-60/mgfn7-i3d_test.npy")

pred = np.load("/MGFNmain/results/UCF_pretrained/mgfn_ucf_test.npy")

path = rf"/home/marc/Documents/data/UCF/UCF_list/"
with open(path + rf"ucf-i3d-test.list", 'r') as f:  # use a context manager to safely opening and closing files
    val = [line.strip() for line in f.readlines()]  # use strip to get rid of the \n at the end of string
val.sort()  # make sure to sort the values!


start = "/home/marc/Documents/data/UCF/test/"
end = "_i3d.npy"

leng = 0

for i in val:
    string = i[len(start):-len(end)] + ".mp4"
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

    path = f"/home/marc/Documents/data/UCF/results/MGFN/Plots_their/" + string[:-3] + ".jpg"
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    leng = length









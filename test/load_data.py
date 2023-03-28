import numpy as np
import matplotlib.pyplot as plt
import pylab
import imageio
import skvideo.io


dir = r'/home/marc/Downloads/UCF_Train_ten_i3d'
dir_vid = r"/home/marc/Documents/data/crime/Anomaly-Videos-Part-1/Abuse/Abuse018_x264.mp4"
vid = imageio.get_reader(dir_vid,  'ffmpeg')
frame_vid = vid.get_data(0)

np_vid = skvideo.io.vread(dir_vid)

frame = np_vid[100]/255
plt.imshow(frame)
plt.show()


data = np.load(dir + "/Abuse016_x264_i3d.npy")  # used to download the data :)

print("")




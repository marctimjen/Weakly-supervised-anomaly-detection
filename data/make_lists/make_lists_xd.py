import os
# https://github.com/Roc-Ng/XDVioDet/blob/master/list/make_list.py

path = "/home/cv05f23/data/XD/RGB"
files = os.listdir(path)
files = sorted(files)
name = "/home/cv05f23/data/XD/files/rgb.list"

with open(name, 'w+') as f:  ## the name of feature list
    for file in files:
        newline = path+'/'+file+'\n'
        f.write(newline)


path = "/home/cv05f23/data/XD/RGBTest"
files = os.listdir(path)
files = sorted(files)
name = "/home/cv05f23/data/XD/files/rgbtest.list"

with open(name, 'w+') as f:  ## the name of feature list
    for file in files:
        newline = path+'/'+file+'\n'
        f.write(newline)


# path = "/home/cv05f23/data/XD/FlowTest"
# files = os.listdir(path)
# files = sorted(files)
# name = "/home/cv05f23/data/XD/files/flowtest.list"
#
# with open(name, 'w+') as f:  ## the name of feature list
#     for file in files:
#         newline = path+'/'+file+'\n'
#         f.write(newline)
#
#
# path = "/home/cv05f23/data/XD/Flow"
# files = os.listdir(path)
# files = sorted(files)
# name = "/home/cv05f23/data/XD/files/flow.list"
#
# with open(name, 'w+') as f:  ## the name of feature list
#     for file in files:
#         newline = path+'/'+file+'\n'
#         f.write(newline)

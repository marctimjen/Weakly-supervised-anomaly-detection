import os
# https://github.com/Roc-Ng/XDVioDet/blob/master/list/make_list.py

path = "/home/cv05f23/data/XD/i3d-features/RGB"
files = os.listdir(path)
files = sorted(files)
name = '/home/cv05f23/data/XD/i3d-features/lists/rgb.list'

with open(name, 'w+') as f:  ## the name of feature list
    for file in files:
        newline = path+'/'+file+'\n'
        f.write(newline)


path = "/home/cv05f23/data/XD/i3d-features/RGBTest"
files = os.listdir(path)
files = sorted(files)
name = '/home/cv05f23/data/XD/i3d-features/lists/rgbtest.list'

with open(name, 'w+') as f:  ## the name of feature list
    for file in files:
        newline = path+'/'+file+'\n'
        f.write(newline)


path = "/home/cv05f23/data/XD/i3d-features/FlowTest"
files = os.listdir(path)
files = sorted(files)
name = '/home/cv05f23/data/XD/i3d-features/lists/flowtest.list'

with open(name, 'w+') as f:  ## the name of feature list
    for file in files:
        newline = path+'/'+file+'\n'
        f.write(newline)


path = "/home/cv05f23/data/XD/i3d-features/Flow"
files = os.listdir(path)
files = sorted(files)
name = '/home/cv05f23/data/XD/i3d-features/lists/flow.list'

with open(name, 'w+') as f:  ## the name of feature list
    for file in files:
        newline = path+'/'+file+'\n'
        f.write(newline)

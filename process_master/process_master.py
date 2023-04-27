import time
import subprocess


i = 0
tick = 0
# py_path = "/home/marc/anaconda3/envs/pyt/bin/python3"
# path = "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/process_master/process.txt"

py_path = "/home/cv05f23/.conda/envs/weak/bin/python3"
path = "/home/cv05f23/git/Weakly-supervised-anomaly-detection/process_master/process.txt"

while True:
    if tick > 60:
        print("waited an hour end process now")
        break

    with open(path, "r") as f:
        file = []
        for line in f:
            file.append(line.strip("\n"))

    file = [[py_path] + i.split(" ") for i in file]

    if i >= len(file):
        print("sleeping at iteration:", i)
        time.sleep(60)
        tick += 1
        continue
    else:
        tick = 0
        print("At iteration:", i)

    command = file[i]

    print("running command:", command)

    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    # text = p.communicate()
    (output, err) = p.communicate()

    # This makes the wait possible
    p_status = p.wait()

    time.sleep(1)
    i += 1


import time
import subprocess
import argparse

"""
This scipt is used to run the different model runs on either the cluster or a local machine. It is good if you have alot
of different models to test.
"""

i = 0
tick = 0

parser = argparse.ArgumentParser(description='master process')
parser.add_argument("-u", '--user', default='cluster', choices=['cluster', 'marc'])  # this gives dir to data and save loc
parser.add_argument("-l", "--logs", required=True, help="Logs to load")  # which process log to load
args = parser.parse_args()

if args.user == "marc":
    py_path = "/home/marc/anaconda3/envs/pyt/bin/python3"
    path = f"/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/process_master/process_{args.user}.txt"
else:
    py_path = "/home/cv05f23/.conda/envs/weak/bin/python3"
    path = f"/home/cv05f23/git/Weakly-supervised-anomaly-detection/process_master/process_{args.logs}.txt"


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


# ["cd", "/home/marc/Documents/data/xd/results/MGFN/Plots_thier2", "rm *"]

import time
import subprocess


flag = False
while True:
    path = "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/process_master/process.txt"
    with open(path, "r") as f:
        file = []
        for line in f:
            file.append(line.strip("\n"))

    # print(file)

    if not flag:
        command = ["/home/marc/anaconda3/envs/pyt/bin/python3", "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/cheat/cheat_test.py", "-u", "marc", "-p", "params_cheat"] # "/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/process_master/long.py"]

        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        # text = p.communicate()
        (output, err) = p.communicate()

    flag = True



    # This makes the wait possible
    p_status = p.wait()

    # This will give you the output of the command being executed
    print(output)

    print("2")

    # print(p)
    # print(text)
    # print(type(text))



    time.sleep(1)


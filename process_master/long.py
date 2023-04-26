import time
from tqdm import tqdm


print("This file takes time to solve")
for i in tqdm(range(10)):
    # print(i)
    time.sleep(1)
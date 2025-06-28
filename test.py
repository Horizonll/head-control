import time, math
import numpy as np

a = [[[1, 2], 3], [[2,3], 4], [[1, 5], 2]]
n = len(a)
point = np.fromiter((xy for x in a for xy in x[0]), dtype=float, count=n*2).reshape(n, 2)
gap_time = np.fromiter((x[1] for x in a), dtype=float, count=n)
stime = np.cumsum(gap_time)
loop_index = 0

while True:
    loop_time = math.fmod(time.time_ns() / 1e9, stime[-1])
    while stime[loop_index] < loop_time:
        loop_index += 1
    if stime[loop_index - 1] > loop_time:
        loop_index = 0
    alpha = (stime[loop_index] - loop_time) / gap_time[loop_index]
    target = point[loop_index-1] * alpha + point[loop_index] * (1 - alpha)
    print(f"{target}")
    time.sleep(0.1)
    

import numpy as np
import os
import random
import pickle
import re
from ReplayMemory import ReplayMemory
import time
from Tool.SendKey import PressKey, ReleaseKey

MEMORY_SIZE = 200
BATCH_SIZE = 20

act_rmp_correct_temp = ReplayMemory(MEMORY_SIZE, file_name='./act_memory')  # experience pool
move_rmp_correct_temp = ReplayMemory(MEMORY_SIZE, file_name='./move_memory')  # experience pool

while True:
    num_list = []
    p = random.randint(0, 44)
    for x in os.listdir('./move_memory'):
        x = re.split('[_.]', x)  # 文件格式为"memory_x.txt"，分割成['memory','x','txt']
        num_list.append(int(float(x[1])))
    num_list.sort()
    m = random.randint(0, num_list[-1])
    move_rmp_correct_temp.buffer = pickle.load(
        open(r'D:\DQN_HollowKnight-main\move_memory\memory_' + str(m) + '.txt', 'rb'))
    act_rmp_correct_temp.buffer = pickle.load(
        open(r'D:\DQN_HollowKnight-main\act_memory\memory_' + str(m) + '.txt', 'rb'))
    batch_station1, batch_actions1, batch_reward1, batch_next_station1, batch_done1 = move_rmp_correct_temp.sample(
        BATCH_SIZE)
    batch_station2, batch_actions2, batch_reward2, batch_next_station2, batch_done2 = act_rmp_correct_temp.sample(
        BATCH_SIZE)
    np.savez('./share/input' + str(p), batch_station1, batch_actions1, batch_reward1, batch_next_station1,
             batch_done1, batch_station2, batch_actions2, batch_reward2, batch_next_station2, batch_done2)
    time.sleep(60)
    PressKey(0x5A)
    time.sleep(0.01)
    ReleaseKey(0x5A)
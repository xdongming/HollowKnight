from random import randint
from ReplayMemory import ReplayMemory
import numpy as np
import pickle

MEMORY_SIZE = 200

BATCH_SIZE=20

# act_rmp_correct = ReplayMemory(MEMORY_SIZE, file_name='./act_memory')  # experience pool
# move_rmp_correct = ReplayMemory(MEMORY_SIZE, file_name='./move_memory')  # experience pool
#
# setm = set()
# while len(setm)<45:
#     m = randint(0, 158)
#     setm.add(m)
# listm = [*setm]
# for i, m in enumerate(listm):
#     move_rmp_correct.buffer = pickle.load(open(r'D:\DQN_HollowKnight-main\move_memory\memory_'+str(m)+'.txt', 'rb'))
#     act_rmp_correct.buffer = pickle.load(open(r'D:\DQN_HollowKnight-main\act_memory\memory_' + str(m) + '.txt', 'rb'))
#     batch_station1, batch_actions1, batch_reward1, batch_next_station1, batch_done1 = move_rmp_correct.sample(
#         BATCH_SIZE)
#     batch_station2, batch_actions2, batch_reward2, batch_next_station2, batch_done2 = act_rmp_correct.sample(
#         BATCH_SIZE)
#     np.savez('./share/input'+str(i), batch_station1, batch_actions1, batch_reward1, batch_next_station1, batch_done1,
#         batch_station2, batch_actions2, batch_reward2, batch_next_station2, batch_done2)

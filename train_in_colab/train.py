# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import random
import pickle
import os
import cv2
import time
import re
import copy
import collections
import matplotlib.pyplot as plt

from Model import Model
from DQN import DQN
from tensorflow.keras.models import Model as mod
from Agent import Agent
from ReplayMemory import ReplayMemory
from Tool.GetMetrics import get_pass_count, get_se, get_q_value

import Tool.Helper
import Tool.Actions
from Tool.Helper import mean, is_end
from Tool.Actions import take_action, restart, take_direction, TackAction
from Tool.WindowsAPI import grab_screen
from Tool.GetHP import Hp_getter
from Tool.UserInput import User
from Tool.FrameBuffer import FrameBuffer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 注释这行将使用GPU运算
window_size = (0, 0, 1920, 1017)
station_size = (230, 230, 1670, 930)

# HP_WIDTH = 768
# HP_HEIGHT = 407
WIDTH = 400
HEIGHT = 200
ACTION_DIM = 7
FRAMEBUFFERSIZE = 4
INPUT_SHAPE = (FRAMEBUFFERSIZE, HEIGHT, WIDTH, 3)

MEMORY_SIZE = 200  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 24  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 20  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.00001  # 学习率
GAMMA = 0.8

action_name = ["Attack", "Attack_Up",
               "Short_Jump", "Mid_Jump", "Skill_Up",
               "Skill_Down", "Rush", "Cure"]

move_name = ["Move_Left", "Move_Right", "Turn_Left", "Turn_Right"]

DELAY_REWARD = 1


def run_episode(hp, algorithm, agent, act_rmp_correct, move_rmp_correct, PASS_COUNT, paused, model):
    # learn while load game
    # m = random.randint(0, 38)
    # move_rmp_correct.buffer = pickle.load(open(r'D:\DQN_HollowKnight-main\move_memory\memory_'+str(m)+'.txt', 'rb'))
    # act_rmp_correct.buffer = pickle.load(open(r'D:\DQN_HollowKnight-main\act_memory\memory_' + str(m) + '.txt', 'rb'))
    # batch_station1, batch_actions1, batch_reward1, batch_next_station1, batch_done1 = move_rmp_correct.sample(
    #     BATCH_SIZE)
    # batch_station2, batch_actions2, batch_reward2, batch_next_station2, batch_done2 = act_rmp_correct.sample(
    #     BATCH_SIZE)
    # np.savez('./share/input', batch_station1, batch_actions1, batch_reward1, batch_next_station1, batch_done1,
    #          batch_station2, batch_actions2, batch_reward2, batch_next_station2, batch_done2)

    restart()

    if (len(move_rmp_correct) > MEMORY_WARMUP_SIZE and len(act_rmp_correct) > MEMORY_WARMUP_SIZE):
        stop = 0
    else:
        stop = 1

    step = 0
    done = 0
    total_reward = 0

    # start_time = time.time()
    # Delay Reward
    DelayMoveReward = collections.deque(maxlen=DELAY_REWARD)
    DelayActReward = collections.deque(maxlen=DELAY_REWARD)
    DelayStation = collections.deque(maxlen=DELAY_REWARD + 1)  # 1 more for next_station
    DelayActions = collections.deque(maxlen=DELAY_REWARD)
    DelayDirection = collections.deque(maxlen=DELAY_REWARD)

    while True:
        while True:
            with open(r'D:\DQN_HollowKnight-main\37to39\boss_hp.txt', 'r') as f:
                data = f.read()
            if data:
                boss_hp_value = eval(data)
                break
        while True:
            with open(r'D:\DQN_HollowKnight-main\37to39\hp.txt', 'r') as f:
                data = f.read()
            if data:
                self_hp = eval(data)
                break
        if boss_hp_value > 800 and boss_hp_value <= 900 and self_hp >= 1 and self_hp <= 9:
            break

    thread1 = FrameBuffer(1, "FrameBuffer", WIDTH, HEIGHT, maxlen=FRAMEBUFFERSIZE)
    thread1.start()

    last_hornet_y = 0
    flag = 0
    while True:
        step += 1
        # last_time = time.time()
        # no more than 10 mins
        # if time.time() - start_time > 600:
        #     break

        # in case of do not collect enough frames
        while (len(thread1.buffer) < FRAMEBUFFERSIZE):
            time.sleep(0.1)
        stations = thread1.get_buffer()
        while True:
            with open(r'D:\DQN_HollowKnight-main\37to39\boss_hp.txt', 'r') as f:
                data = f.read()
            if data:
                boss_hp_value = eval(data)
                break
        while True:
            with open(r'D:\DQN_HollowKnight-main\37to39\hp.txt', 'r') as f:
                data = f.read()
            if data:
                self_hp = eval(data)
                break
        player_x, player_y = hp.get_play_location()
        hornet_x, hornet_y = hp.get_hornet_location()
        while True:
            with open(r'D:\DQN_HollowKnight-main\37to39\soul.txt', 'r') as f:
                data = f.read()
            if data:
                soul = eval(data)
                break

        hornet_skill1 = False
        if last_hornet_y > 32 and last_hornet_y < 32.5 and hornet_y > 32 and hornet_y < 32.5:
            hornet_skill1 = True
        last_hornet_y = hornet_y

        move, action = agent.sample(stations, soul, hornet_x, hornet_y, player_x, hornet_skill1)
        # action = 0
        take_direction(move)
        take_action(action)

        # print(time.time() - start_time, " action: ", action_name[action])
        # start_time = time.time()

        next_station = thread1.get_buffer()
        while True:
            with open(r'D:\DQN_HollowKnight-main\37to39\hp.txt', 'r') as f:
                data = f.read()
            if data:
                next_self_hp = eval(data)
                break
        while True:
            with open(r'D:\DQN_HollowKnight-main\37to39\boss_hp.txt', 'r') as f:
                data = f.read()
            if data:
                next_boss_hp_value = eval(data)
                break
        next_player_x, next_player_y = hp.get_play_location()
        next_hornet_x, next_hornet_y = hp.get_hornet_location()

        # get reward
        move_reward = Tool.Helper.move_judge(self_hp, next_self_hp, player_x, next_player_x, hornet_x, next_hornet_x,
                                             move, hornet_skill1)
        # print(move_reward)
        act_reward, done = Tool.Helper.action_judge(boss_hp_value, next_boss_hp_value, self_hp, next_self_hp,
                                                    next_player_x, next_hornet_x, next_hornet_x, action, hornet_skill1)
        # print(reward)
        # print( action_name[action], ", ", move_name[d], ", ", reward)

        DelayMoveReward.append(move_reward)
        DelayActReward.append(act_reward)
        DelayStation.append(stations)
        DelayActions.append(action)
        DelayDirection.append(move)

        if len(DelayStation) >= DELAY_REWARD + 1:
            if DelayMoveReward[0] != 0:
                move_rmp_correct.append((DelayStation[0], DelayDirection[0], DelayMoveReward[0], DelayStation[1], done))
            # if DelayMoveReward[0] <= 0:
            #     move_rmp_wrong.append((DelayStation[0],DelayDirection[0],DelayMoveReward[0],DelayStation[1],done))

        if len(DelayStation) >= DELAY_REWARD + 1:
            if mean(DelayActReward) != 0:
                act_rmp_correct.append((DelayStation[0], DelayActions[0], mean(DelayActReward), DelayStation[1], done))
            # if mean(DelayActReward) <= 0:
            #     act_rmp_wrong.append((DelayStation[0],DelayActions[0],mean(DelayActReward),DelayStation[1],done))

        station = next_station
        self_hp = next_self_hp
        boss_hp_value = next_boss_hp_value

        # if (len(act_rmp) > MEMORY_WARMUP_SIZE and int(step/ACTION_SEQ) % LEARN_FREQ == 0):
        #     print("action learning")
        #     batch_station,batch_actions,batch_reward,batch_next_station,batch_done = act_rmp.sample(BATCH_SIZE)
        #     algorithm.act_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)

        total_reward += act_reward
        paused = Tool.Helper.pause_game(paused)

        if done == 1:
            Tool.Actions.Nothing()
            break
        elif done == 2:
            PASS_COUNT += 1
            flag = 1
            Tool.Actions.Nothing()
            # time.sleep(3)
            break

    thread1.stop()

    get_pass_count(flag)

    # act_q_sum = 0
    # move_q_sum = 0
    # act_se_sum = 0
    # move_se_sum = 0
    #
    # if stop == 0:
    #     for i in range(1):
    #         with open('./share/command.txt', 'w') as f:
    #             f.write('1')
    #         while True:
    #             with open('./share/command.txt', 'r') as f:
    #                 command = f.read()
    #             if command == '0':
    #                 # flag = os.path.exists(r'D:\DQN_HollowKnight-main\share\input.npz')
    #                 # if flag:
    #                 #     os.remove(r'D:\DQN_HollowKnight-main\share\input.npz')
    #                 break
    #             time.sleep(0.1)
    #         model.load_model()
    #         with open('./share/output.txt', 'r') as f:
    #             data = f.read()
    #         if data:
    #             data = data.split(' ')
    #             act_q, act_loss, move_q, move_loss = eval(data[0]), eval(data[1]), eval(data[2]), eval(data[3])
    #             act_q_sum = act_q_sum + act_q
    #             move_q_sum = move_q_sum + move_q
    #             act_se_sum = act_se_sum + act_loss
    #             move_se_sum = move_se_sum + move_loss
    #             get_q_value('act', act_q_sum / 1)
    #             get_q_value('move', move_q_sum / 1)
    #             get_se('act', act_se_sum / 1)
    #             get_se('move', move_se_sum / 1)
    #
    # if (len(move_rmp_correct) > MEMORY_WARMUP_SIZE and len(act_rmp_correct) > MEMORY_WARMUP_SIZE):
    #     num_list = []
    #     p = random.randint(0, 44)
    #     for x in os.listdir('./move_memory'):
    #         x = re.split('[_.]', x)  # 文件格式为"memory_x.txt"，分割成['memory','x','txt']
    #         num_list.append(int(float(x[1])))
    #     num_list.sort()
    #     m = random.randint(0, num_list[-1])
    #     move_rmp_correct_temp.buffer = pickle.load(
    #         open(r'D:\DQN_HollowKnight-main\move_memory\memory_' + str(m) + '.txt', 'rb'))
    #     act_rmp_correct_temp.buffer = pickle.load(
    #         open(r'D:\DQN_HollowKnight-main\act_memory\memory_' + str(m) + '.txt', 'rb'))
    #     batch_station1, batch_actions1, batch_reward1, batch_next_station1, batch_done1 = move_rmp_correct_temp.sample(
    #         BATCH_SIZE)
    #     batch_station2, batch_actions2, batch_reward2, batch_next_station2, batch_done2 = act_rmp_correct_temp.sample(
    #         BATCH_SIZE)
    #     os.remove(r'D:\DQN_HollowKnight-main\share\input' + str(p) + '.npz')
    #     np.savez('./share/input' + str(p), batch_station1, batch_actions1, batch_reward1, batch_next_station1,
    #              batch_done1, batch_station2, batch_actions2, batch_reward2, batch_next_station2, batch_done2)

    return total_reward, step, PASS_COUNT, self_hp


if __name__ == '__main__':

    # In case of out of memor出现小小小xy
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    sess = tf.compat.v1.Session(config=config)

    total_remind_hp = 0

    act_rmp_correct = ReplayMemory(MEMORY_SIZE, file_name='./act_memory')  # experience pool
    move_rmp_correct = ReplayMemory(MEMORY_SIZE, file_name='./move_memory')  # experience pool

    act_rmp_correct_temp = ReplayMemory(MEMORY_SIZE, file_name='./act_memory')  # experience pool
    move_rmp_correct_temp = ReplayMemory(MEMORY_SIZE, file_name='./move_memory')  # experience pool

    # new model, if exit save file, load it
    model = Model(INPUT_SHAPE, ACTION_DIM)

    # Hp counter
    hp = Hp_getter()
    # hp = 1

    model.load_model()
    model.load_target_model()
    algorithm = DQN(model, gamma=GAMMA, learnging_rate=LEARNING_RATE)
    agent = Agent(ACTION_DIM, algorithm, e_greed=0.12, e_greed_decrement=1e-6)

    with open('./share/command.txt', 'w') as f:
        f.write('0')
    ex = os.path.exists(r'D:\DQN_HollowKnight-main\share\input.npz')
    if ex:
        os.remove(r'D:\DQN_HollowKnight-main\share\input.npz')

    # get user input, no need anymore
    # user = User()

    # paused at the begining
    # paused = False
    paused = True
    paused = Tool.Helper.pause_game(paused)

    max_episode = 30000
    # 开始训练
    episode = 0
    PASS_COUNT = 0  # pass count
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # 训练
        episode += 1
        total_reward, total_step, PASS_COUNT, remind_hp = run_episode(hp, algorithm, agent, act_rmp_correct,
                                                                      move_rmp_correct, PASS_COUNT, paused, model)
        # run_episode(hp, algorithm, agent, act_rmp_correct,
        #             move_rmp_correct, PASS_COUNT, paused, model)
        # if episode % 5 == 0:
        #     model.load_target_model()
        # if episode % 1 == 0:
        #     model.load_model()
        # if episode % 5 == 0:
        #     move_rmp_correct.save(move_rmp_correct.file_name)
        # if episode % 5 == 0:
        #     act_rmp_correct.save(act_rmp_correct.file_name)
        total_remind_hp += remind_hp
        print("Episode: ", episode, ", pass_count: ", PASS_COUNT, ", hp:", total_remind_hp / episode)

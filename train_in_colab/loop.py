from Tool.GetHP import Hp_getter
import time
# import ctypes
# import win32api
# import win32process
# import win32gui

hp = Hp_getter()
while True:
    boss_hp = hp.get_boss_hp()
    self_hp = hp.get_self_hp()
    soul = hp.get_souls()
    with open(r'D:\DQN_HollowKnight-main\37to39\hp.txt', 'w') as f:
        f.write(str(self_hp))
    with open(r'D:\DQN_HollowKnight-main\37to39\soul.txt', 'w') as f:
        f.write(str(soul))
    with open(r'D:\DQN_HollowKnight-main\37to39\boss_hp.txt', 'w') as f:
        f.write(str(boss_hp))
    time.sleep(0.1)
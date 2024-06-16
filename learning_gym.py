import gymnasium as gym
from gymnasium import spaces
import numpy as np

import pyautogui
from time import sleep

import recog_num

def recog2stat(height, isFuelEmpty, isFlightOver, ap, pe):
    fuelEmpty = 1 if isFuelEmpty else 0
    ret = np.array([height, ap, pe, fuelEmpty])
    return ret

def calc_reward(ap, pe):
    return pe + (ap//10)

def calc_action(action):
    time = 0.75 # アクションにかける時間
    act_dic = {0:"w", 1:"s", 2:"a", 3:"d", 4:"space"}
    if action in act_dic:
        key = act_dic[action]
        pyautogui.keyDown(key)
        sleep(time)
        pyautogui.keyUp(key)
    else:
        key = "nothing"
    print("action:",key)
    sleep(time)


class KSPgym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(KSPgym, self).__init__()
        # 0: w, 1:s, 2:a, 3:d, 4:space, 5:nothing
        self.action_space = gym.spaces.Discrete(6) # エージェントが取りうる行動空間を定義
        # height, AP, PE
        self.observation_space = gym.spaces.Box(low=np.array([-np.inf,-np.inf,-np.inf,0]), high=np.array([np.inf,np.inf,np.inf,1]))  # エージェントが受け取りうる観測空間を定義
        self.reward_range = (-np.inf, np.inf)       # 報酬の範囲[最小値と最大値]を定義

    def reset(self):
        # 環境を初期状態にする関数
        # 初期状態をreturnする
        position_of_uchiagemae = (988, 537)
        position_of_uchiagemae2= (1019,576)
        pyautogui.click(*position_of_uchiagemae) # 打ち上げ前に戻る
        pyautogui.click(*position_of_uchiagemae) # 打ち上げ前に戻る
        pyautogui.click(*position_of_uchiagemae2) # 打ち上げ前に戻る
        pyautogui.click(*position_of_uchiagemae2) # 打ち上げ前に戻る
        sleep(1)
        pyautogui.keyDown("f9")
        sleep(3)
        pyautogui.keyUp("f9")
        print("loading...")
        sleep(10)
        for i in range(5):
            print(f"starting in {5-i} seconds...")
            sleep(1)
        pyautogui.click(360,878)
        print("changed mode")
        sleep(0.2)
        pyautogui.press("t")
        print("pressed T")

        rec = recog_num.recog()
        obs = recog2stat(*rec)
        return obs

    def step(self, action):
        calc_action(action)
        ret = recog_num.recog()
        obs = recog2stat(*ret)

        height, isFuelEmpty, isFlightOver, ap, pe = ret
        done = isFlightOver
        # 近点の高度、　遠点の高度、近点と遠点の差の小ささ
        reward = calc_reward(ap,pe)

        info = {}
        return obs, reward, done, info

    def render(self, mode='human'):   
        pass
  
    def close(self):
        pass

    def seed(self, seed=None):
        pass

    
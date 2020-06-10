import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import adam
from torch.distributions import Categorical

import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(1)

torch.manual_seed(1) #为CPU设置种子用于生成随机数，以使得结果是确定的
plt.ion()


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 20)
        #self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(20, action_space)

        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []


    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x
class PG:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate,
            reward_decay,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.log_prob=[]
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

    def choose_action(self,observation):
        observation = torch.from_numpy(observation).float().unsqueeze(0)
        probs = Net(observation)
        c = Categorical(probs)
        action = c.sample()  # 选择随机行为
        log_prob.append(c.log_prob(action))
        action = action.item()  # 返回张量的元素值 只能返回一个数

        return action

    def storage(self,s,a,r):
        self.ep_as.append(a)
        self.ep_obs.append(s)
        self.ep_rs.append(r)

    def learn(self):
        discount_change=self.discount()
        for discount_change, log_prob in zip(discount_change, log_prob):  # 按对应位置合并
            policy_loss.append(-log_prob * reward)
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return discount_change

    def discount(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs




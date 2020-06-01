import numpy as np
import gym
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import adam
from torch.distributions import Categorical

env = gym.make('MountainCar-v0') #创建环境
env = env.unwrapped #解除封印
env.seed(1) #解除封印2

torch.manual_seed(1) #为CPU设置种子用于生成随机数，以使得结果是确定的
plt.ion() #打开交互模式 在plt.show()之前一定不要忘了加plt.ioff()，如果不加，界面会一闪而过，并不会停留
'''在交互模式下：

1、plt.plot(x)或plt.imshow(x)是直接出图像，不需要plt.show()

2、如果在脚本中使用ion()命令开启了交互模式，没有使用ioff()关闭的话，则图像会一闪而过，并不会常留。要想防止这种情况，需要在plt.show()之前加上ioff()命令。

在阻塞模式下：

1、打开一个窗口以后必须关掉才能打开下一个新的窗口。这种情况下，默认是不能像Matlab一样同时开很多窗口进行对比的。

2、plt.plot(x)或plt.imshow(x)是直接出图像，需要plt.show()后才能显示图像
'''

#Hyperparameters
learning_rate = 0.02
gamma = 0.995
episodes = 1000

eps = np.finfo(np.float32).eps.item() #finfo函数是根据括号中的类型来获得信息，获得符合这个类型的数型
#eps是取非负的最小值。当计算的结果为0或为负（但从代码上来看不太可能为负），使用np.finfo(np.float32).eps来替换

action_space = env.action_space.n #3 left 0 right 2.不动 1
state_space = env.observation_space.shape[0] #2 position speed


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.fc1 = nn.Linear(state_space, 20)
        #self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(20, action_space)

        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):

        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x

policy = Policy()
optimizer = adam.Adam(policy.parameters(), lr=learning_rate)

def selct_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0) #生成返回的tensor会和ndarry共享数据.
    # unsequeeze返回一个新的张量，对输入的指定位置插入维度1。注意： 返回张量与输入张量共享内存，所以改变其中一个的内容会改变另一个。
    # 当概率密度函数相对于其参数可微分时，我们只需要sample()和log_prob()
    # △θ = αr▽θ​logπθ​(a∣s)
    # θ是参数，α是学习速率，r是奖励，πθ(a∣s)是在给定策略πθπθ​下在状态ss执行动作a的概率。
    # 在实践中，我们将从网络输出中采样一个动作，将这个动作应用到环境中，然后使用log_prob构造一个等效的损失函数。
    # 请注意，我们使用负数是因为优化器使用梯度下降，而上面的规则假设梯度上升。

    probs = policy(state)
    #print(probs)
    c = Categorical(probs)

    action = c.sample() #选择随机行为
    #print(action)


    policy.saved_log_probs.append(c.log_prob(action))
    action = action.item()#返回张量的元素值 只能返回一个数
    return action

def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]: #反转操作
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Formalize reward 正则化 用方差等操作（一般固定操作）
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean())/(rewards.std() + eps) #.std计算标准差 eps防止分母为0

    # get loss
    for reward, log_prob in zip(rewards, policy.saved_log_probs): #按对应位置合并
        policy_loss.append(-log_prob * reward)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()#矩阵拼接
    policy_loss.backward()
    optimizer.step()



    del policy.rewards[:] #重置记忆空间
    del policy.saved_log_probs[:] #

def plot(steps):
    ax = plt.subplot(111)
    ax.cla()
    ax.set_title('Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Time')
    ax.plot(steps)
    RunTime = len(steps)
    path =  './PG_MountainCar-v0/'+'RunTime'+str(RunTime)+'.jpg'
    if len(steps) % 100 == 0:
        plt.savefig(path)
    plt.pause(0.0000001)



def main():
    running_reward = 0
    steps = []
    for episode in count(60000):
        state = env.reset()

        for t in range(10000):
            action = selct_action(state)
            state, reward ,done, info = env.step(action)
            reward = reward * policy.gamma - t * 0.01

            env.render() #刷新环境
            policy.rewards.append(reward)
            if done:
                print("Episode {}, live time = {}".format(episode, t))
                steps.append(t)
                plot(steps)
                break
        if episode % 50 == 0:
            torch.save(policy, 'policyNet_copy.pkl')
        k=running_reward
        running_reward = running_reward * policy.gamma - t*0.01
        finish_episode()

if __name__ == '__main__':
    main()
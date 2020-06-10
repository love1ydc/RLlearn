import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
TARGET_REPLACE_ITER = 100

n_action=5
n_features=2
class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()
        self.n_features=n_features
        self.fc1 = nn.Linear(n_features, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, n_action)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DeepQN(object):
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 memory_set=2000,
                 batch_size=32,):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.memory_size = memory_set
        self.batch_size = batch_size
        self.learn_step_counter=0
        self.memory_counter=0
        self.cost_count=[]
        self.eval_net, self.target_net = Net(), Net()
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < self.epsilon_max:  # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]  # return the argmax index
            #action=np.argmax(actions_value)
        else:  # random
            action = np.random.randint(0, self.n_actions)

        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.memory_size, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_features])
        b_a = torch.LongTensor(b_memory[:, self.n_features:self.n_features + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_features + 1:self.n_features + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_features:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        self.cost=loss
        self.cost_count.append(loss)
        print(self.cost_count)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_count)), self.cost)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()





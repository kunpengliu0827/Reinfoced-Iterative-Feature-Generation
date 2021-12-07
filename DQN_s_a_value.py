import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
BATCH_SIZE = 8
LR = 0.01
EPSILON = 0.95
GAMMA = 0.99
TARGET_REPLACE_ITER = 100 # After how much time you refresh target network
# MEMORY_CAPACITY = 20 # The size of experience replay buffer
# ACTION_DIM = 2
# STATE_DIM = 4

class Net(nn.Module):

    def __init__(self, STATE_DIM, ACTION_DIM):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM+ACTION_DIM, 100)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization, set seed to ensure the same result
        self.out = nn.Linear(100, 1)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class DQN1(object):

    def __init__(self, STATE_DIM,ACTION_DIM,MEMORY_CAPACITY):
        self.eval_net, self.target_net = Net(STATE_DIM, ACTION_DIM), Net(STATE_DIM, ACTION_DIM)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, STATE_DIM * 2 + ACTION_DIM*2+1))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.STATE_DIM = STATE_DIM
        self.ACTION_DIM = ACTION_DIM
        self.MEMORY_CAPACITY = MEMORY_CAPACITY

    # def choose_action(self, x):
    #     x = torch.unsqueeze(torch.FloatTensor(x), 0)
    #     if np.random.uniform() < EPSILON:
    #         action_value = self.eval_net.forward(x)
    #         action = torch.max(action_value, 1)[1].data.numpy()
    #         action = action[0]
    #     else:
    #         action = np.random.randint(0, self.ACTION_DIM)
    #     return action

    def get_q_value(self, state, action):
        state = torch.tensor(state,dtype=torch.float32)
        action = torch.tensor(action,dtype=torch.float32)
        return self.eval_net(torch.cat((state,action)))


    def store_transition(self, s, a, r, s_, a_):
        transition = np.hstack((s, a, [r], s_,a_))
        index = self.memory_counter % self.MEMORY_CAPACITY  # If full, restart from the beginning
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.STATE_DIM])
        b_a = torch.LongTensor(b_memory[:, self.STATE_DIM:self.STATE_DIM + self.ACTION_DIM])
        b_r = torch.FloatTensor(b_memory[:, self.STATE_DIM + self.ACTION_DIM:self.STATE_DIM+self.ACTION_DIM + 1])
        b_s_ = torch.FloatTensor(b_memory[:, self.STATE_DIM+self.ACTION_DIM + 1: self.STATE_DIM*2+self.ACTION_DIM + 1])
        b_a_ =  torch.LongTensor(b_memory[:, -self.ACTION_DIM:])

        net_input = torch.cat((b_s,b_a),axis=1)
        q_eval = self.eval_net(net_input)
        net_input_ = torch.cat((b_s_,b_a_),axis=1)
        q_next = self.target_net(net_input_)
        q_target = b_r + GAMMA * q_next.view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
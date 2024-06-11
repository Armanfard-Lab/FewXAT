import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)



class Policy(nn.Module):
    def __init__(self, nb_states, nb_actions,img_size, hidden1=400, hidden2=300, init_w=3e-3):
        super(Policy, self).__init__()
        self.nb_actions = nb_actions
        self.img_size = img_size
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions *2 *img_size)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
        logstds_param = nn.Parameter(torch.full((nb_actions,), 0.1))
        self.register_parameter("logstds", logstds_param)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x, x1, x2, x3):
        x = x.float()
        x1 = x1.float()
        x2 = x2.float()
        x3 = x3.float()
        x = torch.cat((x,x1,x2,x3), 1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out).reshape([out.shape[0], self.nb_actions, self.img_size, 2])
        out1 = out[:,:,:,0]
        out2 = out[:,:,:,1]
        out1 = F.softmax(out1, dim=-1)
        out2 = F.softmax(out2, dim=-1)
        return out1, out2


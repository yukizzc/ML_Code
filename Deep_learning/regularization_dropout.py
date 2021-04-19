import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

#dropout,在层与层之间加入
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32) 
        self.fc2 = nn.Linear(32, 10)  
        self.fc3 = nn.Linear(10, 5)  
    
    def forward(self, x):
        out = self.fc1(x)
        # 删减50%的神经元链接
        out = nn.Dropout(0.5)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out
net = NeuralNet()
#L2正则化，weight_decay就是正则化系数lamda
optimizer = optim.SGD(net.parameters(),lr = 0.001,weight_decay=0.01)

#momentum
optimizer = optim.SGD(net.parameters(),lr = 0.001,momentum = 0.78,weight_decay=0.01)


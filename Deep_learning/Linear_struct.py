
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


#数据整理
class DiabetesDataset(Dataset):
    def __init__(self, x_, y_):
        self.len = x_.shape[0]
        self.x_data = torch.from_numpy(x_)
        # 这部很关键， 特征类型要求float类型
        self.x_data = self.x_data.float()
        # 标签需要一维的，并且必须是整形，所以不需要one-hot编码
        self.y_data = torch.from_numpy(y_)

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len

data = pd.read_csv('./iris.csv')
x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values
dataset = DiabetesDataset(x, y)
train_loader = DataLoader(dataset=dataset, batch_size=20, shuffle=True)

#网络结构
class NeuralNet(nn.Module):
    def __init__(self, input_size_, hidden_size_, num_classes_):
        super().__init__()
        self.fc1 = nn.Linear(input_size_, hidden_size_) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_, num_classes_)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out
        
input_size = 28*28
out_size = 1

model = NeuralNet(input_size, 64, 1)
#损失函数
criterion = nn.mse_loss(reduction='mean')
#优化算法
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(100):
    for i, (x, y) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        x = x.reshape(-1, input_size)
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

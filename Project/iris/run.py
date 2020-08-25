import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd

# Hyper-parameters
input_size = 4
hidden_size = 2
num_classes = 3
num_epochs = 50
batch_size = 5
learning_rate = 0.001

data = pd.read_csv('./iris.csv')
x_data = data.iloc[:, 1:-1].values
y_data = data.iloc[:, -1].values
class NeuralNet(nn.Module):
    def __init__(self, input_size_, hidden_size_, num_classes_):
        super().__init__()
        self.fc1 = nn.Linear(input_size_, hidden_size_)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_, num_classes_)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 加载模型
model = torch.load('.\model.pkl')
print(model)
print(model.parameters())
# 得到预测值，注意预测值是类似one-hot的概率分布值
predicted = model(torch.from_numpy(x_data).float())
predicted_label = predicted.argmax(dim=1).detach().numpy()
print(predicted_label)
print(y_data)

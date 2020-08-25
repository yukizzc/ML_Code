import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn.functional as F
# Hyper-parameters
input_size = 4
hidden_size = 2
num_classes = 3
num_epochs = 100
batch_size = 20
learning_rate = 0.01


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
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


class NeuralNet(nn.Module):
    def __init__(self, input_size_, hidden_size_, num_classes_):
        super().__init__()
        self.fc1 = nn.Linear(input_size_, hidden_size_)
        self.fc2 = nn.Linear(hidden_size_, num_classes_)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out


# Logistic regression model
model = NeuralNet(input_size, hidden_size, num_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, input_size)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


# 保存模型
# torch.save(model, 'model.pkl')
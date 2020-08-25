from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch


class DiabetesDataset(Dataset):
    def __init__(self, x_, y_):
        self.len = x_.shape[0]
        self.x_data = torch.from_numpy(x_)
        self.x_data = self.x_data.float()
        self.y_data = torch.from_numpy(y_)
        self.y_data = self.y_data.float()

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len


df = pd.read_csv('./iris.csv')
x_train = df.iloc[:, 1:-1].values
y_train = df.iloc[:, -1].values
dataset = DiabetesDataset(x_train, y_train)
# shuffle是否打乱，建议开启，这里看效果所以关闭了
train_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=False)

for epoch in range(20):
    for i, (x, y) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        print(i, x, y)
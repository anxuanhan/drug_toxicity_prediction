import argparse
import pandas as pd
import torch
import torch.nn as nn

threshold = 0.5

parser = argparse.ArgumentParser(description="输入csv文件的绝对路径")
parser.add_argument('path', type=str, help='文件绝对路径')
args = parser.parse_args()
path = args.path
print(path)
features = pd.read_pickle(path)
features = torch.tensor(features)
print(features.shape)

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(in_features, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 32)
        self.output = nn.Linear(32, out_features)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        out = torch.sigmoid(self.output(x))
        return out


def test_label(yhat_list, threshold):
    for i in range(len(yhat_list)):
        if yhat_list[i] >= threshold:
            yhat_list[i] = 1.0
        else:
            yhat_list[i] = 0.0
    return yhat_list



best_model = Model(729, 1)
best_model.load_state_dict(torch.load('./best_model.dat'))

yhat_list = []

yhat = best_model.forward(features).flatten().data.numpy()
yhat_list.extend(yhat)

yhat_label = test_label(yhat_list,threshold)
print(yhat_label)




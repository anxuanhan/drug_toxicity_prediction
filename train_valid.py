#导库
import torch
import pickle
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader,TensorDataset,SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm

#设超参数
lr = 0.000001
epochs = 250
bs = 32
threshold = 0.5

#导入数据
X_train = pickle.load(open("C:/Users/82511/Desktop/future/project/Deep_learning_short_course-main/Deep_learning_short_course-main/Drug_Toxicity/Data_file/train_feature.pkl","rb"))
y_train = pickle.load(open("C:/Users/82511/Desktop/future/project/Deep_learning_short_course-main/Deep_learning_short_course-main/Drug_Toxicity/Data_file/hERG.pkl","rb"))

X_train = torch.tensor(X_train,dtype = torch.float32)
y_train = torch.tensor(y_train,dtype = torch.float32)

#查看数据结构
print(len(X_train))
print(len(X_train[0]))
print(len(y_train.unique()))

#将特征和标签对应合在一起
dataset = TensorDataset(X_train,y_train)

#按照9:1的比例，将数据分为训练集和验证集
samples_num = len(dataset)
split_num = int(0.9 * samples_num)
data_index = np.arange(samples_num)
np.random.seed(124)
np.random.shuffle(data_index)
train_index = data_index[:split_num]
valid_index = data_index[split_num:]
train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(valid_index)
train_batchdata = DataLoader(dataset=dataset, batch_size=bs, sampler=train_sampler)
valid_batchdata = DataLoader(dataset=dataset, batch_size=bs, sampler=valid_sampler)


# 评估指标返回值
def evaluate_metrics(y_list, yhat_list, threshold):
    """将y_list中的概率转化为0,1"""
    for i in range(len(yhat_list)):
        if yhat_list[i] >= threshold:
            yhat_list[i] = 1.0
        else:
            yhat_list[i] = 0.0
    accuracy = skm.accuracy_score(y_list, yhat_list)
    precision = skm.precision_score(y_list, yhat_list)
    recall = skm.recall_score(y_list, yhat_list)
    f1_score = skm.f1_score(y_list, yhat_list)

    return accuracy, precision, recall, f1_score


# 搭建模型
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
def train_epoch(net, train_batchdata, optimizer,criterion):
    batch_loss = 0
    samples = 0
    for batch_idx, (x, y) in enumerate(train_batchdata):
        yhat = net.forward(x).flatten()  # x的形状[32,729]  sigma
        loss = criterion(yhat, y)  #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = batch_loss + loss.item()  # 这里注意下怎么求损失函数的值
        samples = samples + x.shape[0]

    train_epoch_loss = batch_loss / samples
    return train_epoch_loss
def valid_epoch(net, valid_batchdata,criterion):
    batch_loss = 0
    samples = 0
    yhat_list=[]
    y_list=[]
    with torch.no_grad():
         for batch_idx, (x, y) in enumerate(valid_batchdata):
             yhat = net.forward(x).flatten()
             loss = criterion(yhat, y)  #
             batch_loss = batch_loss + loss.item()  # 这里注意下怎么求损失函数的值
             samples = samples + x.shape[0]
             yhat_list.extend(yhat.data.numpy())
             y_list.extend(y.data.numpy())
         accuracy,precsion,recall,f1_score = evaluate_metrics(y_list,yhat_list,threshold)
    valid_epoch_loss = batch_loss / samples
    return valid_epoch_loss,accuracy,precsion,recall,f1_score
def train_valid(net,train_batchdata,valid_batchdata):

    train_loss = []
    valid_loss = []
    valid_accuracy = []
    valid_precision = []
    valid_recall = []
    valid_f1_score = []
    max_score = 0
    moment_accuracy = 0
    moment_precision = 0
    moment_recall = 0
    moment_epoch = 0

    for epoch in range(epochs):
        train_epoch_loss = train_epoch(net, train_batchdata, optimizer, criterion)
        train_loss.append(train_epoch_loss)
        valid_epoch_loss, accuracy, precision, recall, f1_score = valid_epoch(net, valid_batchdata, criterion)
        valid_loss.append(valid_epoch_loss)
        valid_accuracy.append(accuracy)
        valid_precision.append(precision)
        valid_recall.append(recall)
        valid_f1_score.append(f1_score)

        if f1_score > max_score:
            max_score = f1_score
            moment_accuracy = accuracy
            moment_precision = precision
            moment_recall = recall
            moment_epoch = epoch+1
            torch.save(net.state_dict(),"./best_model.dat")


        #如何找到哪一次循环使f1_score最大呢？

    return train_loss,valid_loss,valid_accuracy,valid_precision,\
        valid_recall,valid_f1_score,max_score, moment_accuracy,moment_precision,moment_recall,moment_epoch


#实例化模型的类，调用loss函数
torch.manual_seed(123)
net = Model(729, 1)
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
train_loss,valid_loss,valid_accuracy,valid_precision,valid_recall,valid_f1_score,max_score,\
    moment_accuracy,moment_precision,moment_recall,moment_epoch = train_valid(net,train_batchdata,valid_batchdata)

print("当Epoch为" + str(moment_epoch) + "时，f1_score最大，最大值为" + str(max_score))
print("此时")
print("Accuracy:", str(moment_accuracy))
print("Precision:", str(moment_precision))
print("Recall:", str(moment_recall))

#画损失函数
plt.plot(train_loss,c="red")
plt.title("train_loss")
plt.savefig("./train_loss.jpg")
plt.show()

plt.plot(valid_loss,c="blue")
plt.title("valid_loss")
plt.savefig("./valid_loss.jpg")
plt.show()

plt.plot(train_loss,c="red",label="train_loss")
plt.plot(valid_loss,c="blue",label="valid_loss")
plt.title("train_valid_loss")
plt.legend()
plt.savefig("./train_valid_loss.jpg")
plt.show()

plt.plot(valid_f1_score,c="orange")
plt.savefig("./valid_f1_score.jpg")
plt.show()
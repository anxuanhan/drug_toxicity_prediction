import torch
import pickle
import torch.nn as nn

X_test = pickle.load(open("C:/Users/82511/Desktop/future/project/Deep_learning_short_course-main/Deep_learning_short_course-main/Drug_Toxicity/Data_file/test_feature.pkl","rb"))
X_test = torch.tensor(X_test, dtype=torch.float32)
print(X_test.shape)

threshold = 0.5

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

def test_label(yhat_test_list, threshold):
    for i in range(len(yhat_test_list)):
        if yhat_test_list[i] >= threshold:
            yhat_test_list[i] = 1.0
        else:
            yhat_test_list[i] = 0.0
    return yhat_test_list



best_model = Model(729, 1)
best_model.load_state_dict(torch.load('./best_model.dat'))

yhat_test_list = []
yhat_test = best_model.forward(X_test).flatten().data.numpy()
yhat_test_list.extend(yhat_test)

yhat_test_label = test_label(yhat_test_list,threshold)
print(yhat_test_label)






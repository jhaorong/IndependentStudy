import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import pandas as pd
from sklearn import preprocessing
#parameter
batch_size=64
num_layers = 0
epochs = 70

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

filename = "output2.txt"
file = open(filename, "r")
first_line = file.readline()  # store first line label
file_lines = file.readlines()  # Store all data each row in a list
numberOfLines = len(file_lines)
requestTimeList = []  # A list to store requestTime(float type)
sectorNumberList = []  # A list to store SectorNumber(int type)
cacheLabelList = []  # A list to store Cache or not Cache with optimal cache method(int type)(1 -> cache 0->not cache)
past_countlist = []         #A list to store currently request's request_count in future
# get train data
i=0
for line in file_lines:
    formLine = line.split('\t')  # split the line with tab
    #print(formLine)
    requestTimeList.append(float(formLine[0]))
    sectorNumberList.append(int(formLine[1]))
    cacheLabelList.append(int(formLine[2]))
    past_countlist.append(int(formLine[5]))
    #print(requestTimeList[i])
    #print(sectorNumberList[i])
    #print(cacheLabelList[i])
    #print(past_countlist[i])
#train data split 70% of all data
train_requestTimeList = requestTimeList[:int(0.7*numberOfLines)]
train_sectorNumberList = sectorNumberList[:int(0.7*numberOfLines)]
train_cacheLabelList =  cacheLabelList[:int(0.7*numberOfLines)]
train_past_countlist =  past_countlist[:int(0.7*numberOfLines)]
#test data split 30% of all data
test_requestTimeList = requestTimeList[int(0.7*numberOfLines):]
test_sectorNumberList = sectorNumberList[int(0.7*numberOfLines):]
test_cacheLabelList =  cacheLabelList[int(0.7*numberOfLines):]
test_past_countlist =  past_countlist[int(0.7*numberOfLines):]


train_data_size = int(0.7*numberOfLines)-int(0.7*numberOfLines)%batch_size #drop last must cut the remainder data that is lower than batchsize
print("train data size",train_data_size)
test_data_size = int(0.3*numberOfLines)  - int(0.3*numberOfLines)%batch_size #drop last must cut the remainder data that is lower than batchsize
print("test data size",test_data_size)
data_size = numberOfLines - numberOfLines%batch_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #decide whether use gpu to run
class LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=100, output_size=1):

        #LSTM二分類任務
        #:param input_size: 輸入數據的維度
        #:param hidden_layer_size:隱藏層的數目
        #:param output_size: 輸出的個數
        super().__init__()
        self.hidden_layer_size = hidden_layer_size #represent the feature dimension of hidden layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        #LSTM parameter
        #input_size：x的特徵維度
        #hidden_size：隱藏層的特徵維度
        #num_layers：lstm隱層的層數，默认为1
        #bias：False則bih = 0和bhh = 0.默認為True
        #batch_first：True則輸入輸出得數據格式為(batch, seq, feature)
        #dropout：除最后一層，每一层的输出都進行dropout，默認為: 0
        #bidirectional：True則為雙向lstm默認為False
        self.linear = nn.Linear(hidden_layer_size, output_size)
        """
            linear layer
            transform dimension (input_size,hidden_size) to (input_size,output_size) 
            y=xAT+*b
        """
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x):
        #print("input size = ", (input_x.shape))  # shape(batch_size,1,in_feature)
        input_x = input_x.view(len(input_x), 1, -1)
        #print("input size = ", (input_x.shape)) #shape(batch_size,1,in_feature)
        if torch.cuda.is_available():
            hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size,device=device),  # shape: (n_layers, batch, hidden_size)
                           torch.zeros(1, 1, self.hidden_layer_size,device=device))
        else:
            hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),  # shape: (n_layers, batch, hidden_size)
                           torch.zeros(1, 1, self.hidden_layer_size))
        lstm_out, (h_n, h_c) = self.lstm(input_x, hidden_cell)
        #print("output size = ", (lstm_out.shape))
        linear_out = self.linear(lstm_out.view(len(input_x), -1))  # self.linear(lstm_out[batch,hidden_layer_size])
        predictions = self.sigmoid(linear_out)
        return predictions

def get_data(requestTimeList,sectorNumberList,past_countlist,cacheLabelList):
    def get_tensor_from_pd(dataframe_series) -> torch.Tensor:
        return torch.tensor(data=dataframe_series.values)

    # 生成训练数据x并做标准化后，构造成dataframe格式，再转换为tensor格式
    requestTimeListToNdarray = np.array(requestTimeList)
    sectorNumberListToNdarray = np.array(sectorNumberList)
    past_countlistToNdarray = np.array(past_countlist)
    cacheLabelListToNdarray = np.array(cacheLabelList)
    input = np.vstack([requestTimeListToNdarray,sectorNumberListToNdarray,past_countlistToNdarray]).reshape(-1,3) #let input feature combine to an 2-D array with rows
    #print("input size",input.shape)
    df = pd.DataFrame(data=preprocessing.StandardScaler().fit_transform(input))#Use pandas to preproceessing data (standardization)
    y = pd.Series(cacheLabelListToNdarray)
    return get_tensor_from_pd(df).float(), get_tensor_from_pd(y).float()
x, y = get_data(train_requestTimeList,train_sectorNumberList,train_past_countlist,train_cacheLabelList)
train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x, y),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=num_layers,  # 多进程（multiprocess）来读数据
        drop_last = True,
    )
#preprocessing unbalanced data label by setting weight
#find rate
train_cacheLabel_count = 0
for i,label in enumerate(train_cacheLabelList):
    #print("i=",i,"cache label,",label)
    if  label == 1:
        train_cacheLabel_count+=1
train_cacheLabel_scale = train_cacheLabel_count/train_data_size
#print("train_cacheLabel_scale:",train_cacheLabel_scale)
model = LSTM().to(device)  # 模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器

f_accuracy = open("accuracy.txt","w")
# 开始训练
model.train()
train_accuracy = 0
total_train_loss = 0
total_train_accuracy = 0
current_epoch_train_accuracy = 0
i=0
j=0
weight = [0] * batch_size
fpredict = open("predict.txt","w")
fpredict.write("train predict\n")
if __name__ == '__main__':

    for i in range(epochs):
        #step = 0
        current_epoch_train_accuracy = 0
        for step,data in enumerate(train_loader):
            optimizer.zero_grad()
            seq, labels = data
            #print("seq = ",seq)
            seq = seq.to(device)
            labels = labels.to(device)
            y_pred = model(seq).squeeze(dim=1)  # 压缩维度：得到输出，并将维度为1的去除  model(seq)
            #print("seq shape= ", seq.shape)
            #print("y_pred shape= ", y_pred.shape)
            # 若想要获得类别，二分类问题使用四舍五入的方法即可：print(torch.round(y_pred))
            print("train y_pred", y_pred)
            #print("train y_pred round", torch.round(y_pred))
            fpredict.write(str(y_pred))
            for j,label in enumerate(labels) :
                if label == 1:
                    weight[j] = (100 - train_cacheLabel_scale * 100)
                else:
                    weight[j] = (train_cacheLabel_scale * 100)
            weight_tensor = torch.Tensor(weight)
            #print(weight_tensor)
            #print(weight_tensor)
            loss_function = nn.BCELoss(weight=weight_tensor).to(device)  # loss
            single_loss = loss_function(y_pred, labels)
            total_train_loss = total_train_loss + single_loss.item()
            #print("labels ",labels)
            #print("detach {}".format(np.argmax(y_pred.detach().cpu().numpy(),axis=0) == labels))
            train_accuracy = (np.argmax(y_pred.round().detach().cpu().numpy(), axis=0) == labels).sum()
            current_epoch_train_accuracy = current_epoch_train_accuracy + train_accuracy
            total_train_accuracy = total_train_accuracy + train_accuracy
            #print("train accuracy {}".format(train_accuracy))
            single_loss.backward()
            optimizer.step()
            #print("Train Epoch: ",i,"Step",step," loss: ", single_loss)
            print("train epoch：", i, "的第", step, "个inputs", seq.shape, "labels", labels.shape)
            step+=1
        f_accuracy.write("train epoch {} accuracy {}\n".format(i, current_epoch_train_accuracy / (train_data_size)))
#print("整體訓練集上的的Loss: {}".format(total_train_loss))
#print("total train accuracy {}".format(total_train_accuracy))
#print("total train size {}".format(train_data_size*epochs))
#print("整體訓練集上的正確率: {}".format(total_train_accuracy / (train_data_size*epochs)))
f_accuracy.write("total train accuracy = {}\n".format(total_train_accuracy / (train_data_size*epochs)))

# 开始验证
x, y = get_data(test_requestTimeList,test_sectorNumberList,test_past_countlist,test_cacheLabelList)
test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x, y),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=num_layers,  # 多进程（multiprocess）来读数据
        drop_last=True
    )
i=0
j=0
weight = [0] * batch_size
total_test_loss = 0
total_test_accuracy = 0
current_epoch_test_accuracy = 0
test_accuracy = 0
model.eval()
fpredict.write("test predict")
if __name__ == '__main__':
    for i in range(epochs):
        #step = 0
        current_epoch_test_accuracy = 0
        for step,data in enumerate(test_loader):
            seq, labels = data
            seq = seq.to(device)
            labels = labels.to(device)
            y_pred = model(seq).squeeze(dim=1)  # 压缩维度：得到输出，并将维度为1的去除
            print("test Y pred", y_pred)
            #print("test y_pred round", torch.round(y_pred))
            """
            if i == epochs-1:
                result = torch.cat((seq, torch.round(y_pred).view(len(y_pred),1)),dim=1)
                for info in result:
                    print("predict {}".format(info[3]))
                    if info[3] == 1:
                        f_result.write("{}\t{}\t{}\t{}\n".format(info[0],info[1],info[2],int(info[3])))
            """
            for j, label in enumerate(labels):
                if label == 1:
                    weight[j] = (100 - train_cacheLabel_scale * 100)
                else:
                    weight[j] = (train_cacheLabel_scale * 100)
            weight_tensor = torch.Tensor(weight)
            # print(weight_tensor)
            #print(labels)
            loss_function = nn.BCELoss(weight=weight_tensor).to(device)  # loss
            single_loss = loss_function(y_pred, labels)
            #print("Test Epoch: ",i,"Step",step," loss: ", single_loss)
            total_test_loss = total_test_loss + single_loss.item()
            test_accuracy = (np.argmax(y_pred.round().detach().cpu().numpy(),axis=0) == labels).sum()
            #print("test accuracy {}".format(test_accuracy))
            current_epoch_test_accuracy = current_epoch_test_accuracy + test_accuracy
            total_test_accuracy = total_test_accuracy + test_accuracy
            print("test epoch：", i, "的第", step, "个inputs", seq.shape, "labels", labels.shape)
            step+=1
        f_accuracy.write("test epoch {} accuracy {}\n".format(i, current_epoch_test_accuracy / (test_data_size)))
    #print("整體測試集上的的Loss: {}".format(total_test_loss))
    #print("total test accuracy {}".format(total_test_accuracy))
    #print("total test size {}".format(test_data_size*epochs))
    #print("整體測試集上的正確率: {}".format(total_test_accuracy / (test_data_size*epochs)))
f_accuracy.write("total test accuracy = {}\n".format(total_test_accuracy / (test_data_size*epochs)))
torch.save(model.state_dict(),"modelparameter.pth")

x, y = get_data(requestTimeList,sectorNumberList,past_countlist,cacheLabelList)
data_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x, y),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=batch_size,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=num_layers,  # 多进程（multiprocess）来读数据
        drop_last=True
    )
#result
f_result = open("result.txt","w")
f_result.write("request_time\tsector_number\tpast_count\n")
#step = 0
total_data_loss = 0
data_accuracy  = 0
current_epoch_data_accuracy = 0
for step,data in enumerate(data_loader):
    seq, labels = data
    seq = seq.to(device)
    labels = labels.to(device)
    y_pred = model(seq).squeeze(dim=1)  # 压缩维度：得到输出，并将维度为1的去除
    print("result Y pred", y_pred)
    # print("test y_pred round", torch.round(y_pred))
    for k,pred in enumerate(y_pred):
        if pred.round() == 1:
            print(pred)
            f_result.write("{}\t{}\t{}\n".format(requestTimeList[64*step+k],sectorNumberList[64*step+k],past_countlist[64*step+k]))
    for j, label in enumerate(labels):
        if label == 1:
            weight[j] = (100 - train_cacheLabel_scale * 100)
        else:
            weight[j] = (train_cacheLabel_scale * 100)
    weight_tensor = torch.Tensor(weight)
    # print(weight_tensor)
    # print(labels)
    loss_function = nn.BCELoss(weight=weight_tensor).to(device)  # loss
    single_loss = loss_function(y_pred, labels)
    # print("Test Epoch: ",i,"Step",step," loss: ", single_loss)
    total_data_loss = total_data_loss + single_loss.item()
    data_accuracy = (np.argmax(y_pred.round().detach().cpu().numpy(), axis=0) == labels).sum()
    # print("test accuracy {}".format(test_accuracy))
    current_epoch_data_accuracy = current_epoch_data_accuracy + data_accuracy
    step += 1
f_accuracy.write("result accuracy {}\n".format(current_epoch_data_accuracy / (data_size)))

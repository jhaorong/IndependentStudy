import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.data as Data
import numpy as np
import pandas as pd
from sklearn import preprocessing
#parameter
batch_size = 100
epochs = 1000
num_layers = 0

requestTimeList = []  # A list to store requestTime(float type)
sectorNumberList = []  # A list to store SectorNumber(int type)
past_countlist = []         #A list to store currently request's request_count in the past
stayCacheTimeList = []       #A list to store how long is the request stay in cache
hitList = []

f_input = open("input.txt", "r")
first_line = f_input.readline()  # store first line label
input_lines = f_input.readlines()  # Store all data each row in a list
numberOfLines = len(input_lines)
# get train data
#i=0
for id,line in enumerate(input_lines):
    formLine = line.split('\t')  # split the line with tab
    #print(formLine)
    requestTimeList.append(float(formLine[0]))
    sectorNumberList.append(int(formLine[1]))
    past_countlist.append(int(formLine[2]))
    stayCacheTimeList.append(int(formLine[3]))
    hitList.append(int(formLine[4]))
    #print(requestTimeList[i],"\t",sectorNumberList[i],"\t",past_countlist[i],stayCacheTimeList[i],"\n")
    #i+=1
#print("hit list = {}".format(hitList))

#nomalize stay time
"""
stayTimeNormal = [0]*numberOfLines
for id,data in enumerate(stayCacheTimeList):
    normalized = (data-min)/(max-min)#MinMax
    stayTimeNormal[id] = normalized
"""
"""
#divide category
catergory = 5
interval = (max - min)/5
labelList = [6]*numberOfLines
for id,data in enumerate(stayCacheTimeList):
    if data >= min and data < min+interval: #label 0
        labelList[id] = 0
    elif data >= min+interval and data < min+2*interval: #label 1
        labelList[id] = 1
    elif data >= min+2*interval and data < min+3*interval: #label 2
        labelList[id] = 2
    elif data >= min+3*interval and data < min+4*interval: #label 3
        labelList[id] = 3
    elif data >= min+4*interval and data <= min+5*interval: #label 4
        labelList[id] = 4
"""





#train data split 70% of all data
train_requestTimeList = requestTimeList[:int(0.7*numberOfLines)]
train_sectorNumberList = sectorNumberList[:int(0.7*numberOfLines)]
train_past_countlist =  past_countlist[:int(0.7*numberOfLines)]
train_stayCacheTimeList =  stayCacheTimeList[:int(0.7*numberOfLines)]
train_hitList =  hitList[:int(0.7*numberOfLines)]
#test data split 30% of all data
test_requestTimeList = requestTimeList[int(0.7*numberOfLines):]
test_sectorNumberList = sectorNumberList[int(0.7*numberOfLines):]
test_past_countlist =  past_countlist[int(0.7*numberOfLines):]
test_stayCacheTimeList =  stayCacheTimeList[int(0.7*numberOfLines):]
test_hitList =  hitList[int(0.7*numberOfLines):]

train_data_size = int(0.7*numberOfLines)-int(0.7*numberOfLines)%batch_size #drop last must cut the remainder data that is lower than batchsize
print("train data size",train_data_size)
test_data_size = int(0.3*numberOfLines)  - int(0.3*numberOfLines)%batch_size #drop last must cut the remainder data that is lower than batchsize
print("test data size",test_data_size)
total_data_size = train_data_size + test_data_size
print("total data size",total_data_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #decide whether use gpu to run
print("device is {}".format(device))
total_time = 0
for time in train_stayCacheTimeList:
    total_time += time
#print(total_time/train_data_size)
class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=100, output_size=1):

        #LSTM二分類任務
        #:param input_size: 輸入數據的維度
        #:param hidden_layer_size:隱藏層的特徵數
        #:param output_size: 輸出的個數
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size  # represent the feature dimension of hidden layer
        self.hidden_layer_num = 1

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_layer_size,
            batch_first=True,
            num_layers=self.hidden_layer_num
        )
        #LSTM parameter
        #input_size：x的特徵維度
        #hidden_size：隱藏層的特徵維度
        #num_layers：lstm隱層的層數，默认为1
        #bias：False則bih = 0和bhh = 0.默認為True
        #batch_first：True則輸入輸出得數據格式為(batch, seq, feature)
        #dropout：除最后一層，每一层的输出都進行dropout，默認為: 0
        #bidirectional：True則為雙向lstm默認為False
        self.linear = nn.Linear(in_features=hidden_layer_size, out_features=output_size)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, input_x):
        #print("input size = ", (input_x.shape))  # shape(batch_size,1,in_feature)
        #input_x = input_x.view(len(input_x), 1, -1)
        #print("input size = ", (input_x.shape)) #shape(batch_size,1,in_feature)
        if torch.cuda.is_available():
            hidden_cell = (torch.zeros(self.hidden_layer_num, batch_size, self.hidden_layer_size,device=device).requires_grad_(),  # shape: (n_layers, batch, hidden_size)
                           torch.zeros(self.hidden_layer_num, batch_size, self.hidden_layer_size,device=device).requires_grad_())
        else:
            hidden_cell = (torch.zeros(self.hidden_layer_num, batch_size, self.hidden_layer_size).requires_grad_(),  # shape: (n_layers, batch, hidden_size)
                           torch.zeros(self.hidden_layer_num, batch_size, self.hidden_layer_size).requires_grad_())
        #self.sigmoid(input_x)
        lstm_out, (h_n, c_n) = self.lstm(input_x, hidden_cell)
        #print("output size = ", (lstm_out.shape))
        linear_out = self.linear(h_n[0]).flatten()  # self.linear(lstm_out[batch,hidden_layer_size])
        #print("linear out = {}\n".format(linear_out))
        #print("linear out shape = {}\n".format(linear_out.shape))
        #predictions = self.sigmoid(linear_out)

        return linear_out

class SequenceDataset(Dataset):
    def __init__(self,x,y,sequence_length=5):
        self.sequence_length = sequence_length
        self.X = x
        self.y = y
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]

def get_data(request,sector,past_count,stayCacheTime,hit):
    def get_tensor_from_pd(dataframe_series) -> torch.Tensor:
        return torch.tensor(data=dataframe_series.values)
    # 生成训练数据x并做标准化后，构造成dataframe格式，再转换为tensor格式
    requestTimeListToNdarray = np.array(request)
    sectorNumberListToNdarray = np.array(sector)
    past_countlistToNdarray = np.array(past_count)
    stayCacheTimeListToNdarray = np.array(stayCacheTime)
    hitListToNdarray = np.array(hit)
    input = np.hstack([requestTimeListToNdarray,sectorNumberListToNdarray,past_countlistToNdarray,hitListToNdarray]).reshape(4,len(request)).transpose() #let input feature combine to an 2-D array with rows
    #print("input = ",input)
    #print("input size",input.shape)
    df = pd.DataFrame(data=preprocessing.MinMaxScaler().fit_transform(input))#Use pandas to preproceessing data (standardization)
    #df = pd.DataFrame(input)#Use pandas to preproceessing data (standardization)
    #print("df = \n",df)
    #df = pd.DataFrame(data = input)
    #stayCacheTime_np = np.array(stayCacheTimeList).reshape(-1,1)
    y = pd.Series(stayCacheTimeListToNdarray)
    return get_tensor_from_pd(df).float(), get_tensor_from_pd(y).float()

x, y = get_data(train_requestTimeList,train_sectorNumberList,train_past_countlist,train_stayCacheTimeList,train_hitList)
#print(y)


sequence_length = 5

train_dataset = SequenceDataset(
    x=x,
    y=y,
    sequence_length=sequence_length
)


train_loader = Data.DataLoader(
        dataset=train_dataset,  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=num_layers,  # 多进程（multiprocess）来读数据
        drop_last = True,
    )
# 开始验证
x, y = get_data(test_requestTimeList,test_sectorNumberList,test_past_countlist,test_stayCacheTimeList,test_hitList)
test_dataset = SequenceDataset(
    x=x,
    y=y,
    sequence_length=sequence_length
)
test_loader = Data.DataLoader(
        dataset=test_dataset,  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=num_layers,  # 多进程（multiprocess）来读数据
        drop_last = True,
    )

#X, y = next(iter(train_loader))
#print(y.shape)
#print(y)
total_train_loss = 0
model = LSTM().to(device)  # 模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # 优化器
fpredict = open("predict.txt","w")
torch.Tensor.ndim = property(lambda self: len(self.shape))
f_loss = open("loss.txt", "w")

if __name__ == '__main__':
    for i in range(epochs):
        #step = 0
        #current_epoch_train_accuracy = 0
        print("epoch = {}".format(i))
        total_train_loss = 0
        model.train()
        for step,data in enumerate(train_loader):
            #optimizer.zero_grad()
            seq, labels = data
            #print("seq = ",seq)
            #print("seq = \n", seq)
            #print("label = \n", labels)
            seq = seq.to(device)
            labels = labels.to(device)
            y_pred = model(seq)
            #y_pred = y_pred.squeeze(dim=1) # 压缩维度：得到输出，并将维度为1的去除  model(seq)
            #print("seq shape= ", seq.shape)
            #print("y_pred shape= ", y_pred.shape)
            # 若想要获得类别，二分类问题使用四舍五入的方法即可：print(torch.round(y_pred))
            #print("train y_pred", y_pred)
            #print("train y_pred round", torch.round(y_pred))
            #fpredict.write(str(y_pred))
            #print(weight_tensor)
            #print(weight_tensor)
            loss_function = nn.MSELoss().to(device)  # loss
            single_loss = loss_function(y_pred, labels)
            #print(single_loss.item())
            total_train_loss = total_train_loss + single_loss.item()
            optimizer.zero_grad()
            single_loss.backward()
            optimizer.step()

            #print("Train Epoch: ",i,"Step",step," loss: ", single_loss)
            fpredict.write("train epoch： {} 的第 {}\n labels {}\n".format(i,step,labels))
            fpredict.write("train epoch： {} 的第 {}\n predict {}\n".format(i, step, y_pred))
            #fpredict.write("train epoch： {} 的第 {}\n loss {}\n".format(i, step, single_loss))
            #print("train epoch：", i, "的第", step, "个inputs", seq.shape, "labels", labels.shape)
            #step+=1
        avg_loss = total_train_loss / len(train_loader)
        print("train epoch {} avg loss = {}".format(i,avg_loss))
        f_loss.write("train epoch = {} avg_loss = {}\n".format(i,avg_loss))
        #test
        total_test_loss = 0
        model.eval()
        for step, data in enumerate(test_loader):
            with torch.no_grad():
                seq, labels = data
                seq = seq.to(device)
                # print(seq)
                labels = labels.to(device)
                y_pred = model(seq)
                # print("test Y pred", y_pred)
                loss_function = nn.MSELoss().to(device)  # loss
                single_loss = loss_function(y_pred, labels)
                total_test_loss = total_test_loss + single_loss.item()
        avg_loss = total_test_loss / len(test_loader)
        print("test epoch = {} avg_loss = {}".format(i, avg_loss))
        f_loss.write("test epoch = {} avg_loss = {}\n".format(i, avg_loss))

"""
total_test_loss = 0
model.eval()
f_loss.write("test loss\n")
if __name__ == '__main__':
    for i in range(epochs):
        total_test_loss = 0
        for step,data in enumerate(test_loader):
            with torch.no_grad():
                seq, labels = data
                seq = seq.to(device)
                #print(seq)
                labels = labels.to(device)
                y_pred = model(seq)
                #print("test Y pred", y_pred)
                loss_function = nn.MSELoss().to(device)  # loss
                single_loss = loss_function(y_pred, labels)
                total_test_loss = total_test_loss + single_loss.item()
        avg_loss = total_test_loss / len(test_loader)
        print("test epoch = {} avg_loss = {}".format(i,avg_loss))
        f_loss.write("test epoch = {} avg_loss = {}".format(i,avg_loss))
"""
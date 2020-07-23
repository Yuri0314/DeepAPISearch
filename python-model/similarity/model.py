import torch
from torch import nn

# 语义相似度GRU模型
class GRU(nn.Module):
    #其中input_size是指词表的大小
    def __init__(self,input_size,hidden_size,out_size):
        super(GRU,self).__init__()
        # self.batch_size=batch_size
        self.hidden_size=hidden_size
        self.embedding=nn.Embedding(input_size,hidden_size)
        self.gru=nn.GRU(hidden_size,hidden_size,num_layers=3)
        self.linear1=nn.Linear(hidden_size,512)
        self.linear2=nn.Linear(512,256)
        self.linear3=nn.Linear(256,256)
        self.out = nn.Linear(256, out_size)
        self.dropout= nn.Dropout(p=0.6)
        # self.bn = nn.BatchNorm1d(hidden_size)
    def forward(self,input,hidden):
        # 输入形状是(批量大小, 时间步数)。将输出互换样本维和时间步维
        #理解此处的0与1维度互换，因为是GRU序列，按照文本序列一个词一个词的输入进循环网络
        embedded=self.embedding(input.long()).permute(1,0,2)# (seq_len, batch, input_size)
        output=embedded
        output,state=self.gru(output,hidden)
        output=self.dropout(output[-1])
        # output = self.bn(output[-1])
        output=torch.relu(self.linear1(output))
        output=torch.relu(self.linear2(output))
        output=torch.relu(self.linear3(output))
        output=torch.relu(self.linear3(output))
        output=torch.relu(self.linear3(output))
        output=torch.relu(self.linear3(output))
        output=torch.relu(self.linear3(output))
        output=torch.relu(self.linear3(output))
        output=self.out(output)
        output=self.dropout(output)
        #移除时间步维，输出形状为(批量大小, 输出词典大小)
        #疑问：state是否是最后linear层的输入,输出的尺寸应该为（GRU层数，batch_size，num_hidden）
        return output.squeeze(dim=0)

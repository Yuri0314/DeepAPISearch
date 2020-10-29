import torch
from torch import nn

class GRU_API(nn.Module):
    #其中input_size是指词表的大小
    def __init__(self,batch_size,input_size,hidden_size,out_size):
        super().__init__()
        self.hidden_size=hidden_size
        self.embedding=nn.Embedding(input_size,hidden_size)
        self.gru=nn.GRU(hidden_size,hidden_size,num_layers=2,bidirectional=True)
        self.linear=nn.Linear(hidden_size*2,out_size)
        self.LeakyReLU= nn.LeakyReLU(negative_slope=0.01)
    def forward(self,input_api,input_method,hidden):
        output_api=self.vecInput(input_api,hidden)
        output_method=self.vecInput(input_method,hidden)
        output=torch.cat((output_api,output_method),1)
        output=self.linear(output)
        return output
    
    def vecInput(self,input,hidden):
        # 输入形状是(批量大小, 时间步数)。将输出互换样本维和时间步维
        #理解此处的0与1维度互换，因为是GRU序列，按照文本序列一个词一个词的输入进循环网络
        embedded=self.embedding(input.long()).permute(1,0,2)# (seq_len, batch, input_size)
        output=embedded
        output,state=self.gru(output,hidden)
        output=state[-1]
        #移除时间步维，输出形状为(批量大小, 输出词典大小)
        return output.squeeze(dim=0)


# 语义相似度GRU模型
class GRU_QUERY(nn.Module):
    #其中input_size是指词表的大小
    def __init__(self,batch_size,input_size,hidden_size,out_size):
        super().__init__()
        self.hidden_size=hidden_size
        self.embedding=nn.Embedding(input_size,hidden_size)
        self.gru=nn.GRU(hidden_size,hidden_size,num_layers=2,bidirectional=True)
        self.out = nn.Linear(hidden_size, out_size)
    def forward(self,input,hidden):
        # 输入形状是(批量大小, 时间步数)。将输出互换样本维和时间步维
        #理解此处的0与1维度互换，因为是GRU序列，按照文本序列一个词一个词的输入进循环网络
        embedded=self.embedding(input.long()).permute(1,0,2)# (seq_len, batch, input_size)
        output=embedded
        _, state=self.gru(output,hidden)
        output=state[-1]
        output=self.out(output)     
        #移除时间步维，输出形状为(批量大小, 输出词典大小)
        return output.squeeze(dim=0)



class Similarity(nn.Module):
    def __init__(self,model1,model2):
        super().__init__()

        self.rnn_api=model1
        self.rnn_query=model2
    
    def forward(self,query,api,method,hidden_size,device):
        w = torch.empty(4,query.shape[0],hidden_size).to(device)
        init_state=torch.nn.init.orthogonal_(w, gain=1)
        q_state=init_state
        api_state=init_state
        
        q_output=self.rnn_query(query,q_state)
        api_output=self.rnn_api(api,method,api_state)

        return q_output,api_output
        



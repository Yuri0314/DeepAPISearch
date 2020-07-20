#注意！！！！！ 需要更改训练函数的维度以及最后两个向量的相似度计算！！！！与AI研习社中的计算方法相同#

import collections
import os
import io
import re
import math
import torch
from torch import nn
import torch.nn.functional as F
import torchtext.vocab as Vocab
import torch.utils.data as Data
import sys
import pandas as pd

PAD,EOS='<pad>','<eos>'
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#处理数据对
def processline(x,y):
    # 去除各种杂乱的标点符号
    x=re.sub('[.(),->]',' ',x)
    #使用strip去除y尾部的换行符
    y=re.sub(r'[()]',' ',y.strip())
    #  把#部分与前面类名用空格分开,其中line为字符串
    x=re.sub('[#]',' #',x)
    #使用line1存储处理过后的x字符串，line2存储处理过后的y字符串
    line1=''
    line2=''
    for i in x.split(' '):
        if i=='':
            continue
        if not re.match('#.*',i):
            line1+=i+' '
        else:      
            #下面语句是将方法名按照大写字母分割开,对于y为空的语句将#后面切割之后的方法名替代
            if y=='':
                y=re.sub("[A-Z]",lambda x:" "+x.group(0),i[1:])
            line1+=(re.sub("[A-Z]",lambda x:" "+x.group(0),i[1:]))+' '
    for j in y.split(' '):
        if j=='':
            continue
        else:
            line2+=j+' '   
    # 返回处理之后的x与y字符串
    return line1.lower().strip(),line2.lower().strip()

# 将一个序列中所有的词记录在all_tokens中以便之后构造词典，然后在该序列后面添加PAD直到序列
# 长度变为max_seq_len，然后将序列保存在all_seqs中
def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    all_tokens.extend(seq_tokens)
    seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)
    all_seqs.append(seq_tokens)

# 使用所有的词来构造词典。并将所有序列中的词变换为词索引后构造Tensor
def build_data(all_tokens, all_seqs):
    vocab = Vocab.Vocab(collections.Counter(all_tokens),
                        specials=[PAD, EOS])
    indices = [[vocab.stoi[w] for w in seq] for seq in all_seqs]
    return vocab, torch.tensor(indices)


# 开始读取数据，构建词典，以及数据集
def read_data(path,query_max_length,api_max_length):
    in_tokens,out_tokens,in_seqs,out_seqs=[],[],[],[]
    with io.open(path) as f:
        lines=f.readlines()
    for line in lines:
        #将每一行语句按照'：'将api与对应的query查询分开
        out_seq,in_seq=line.split(':::')
        #使用processline处理输入输出数据的各种标点，输出干净字符串
        out_seq,in_seq=processline(out_seq,in_seq)
        
        in_seq_tokens,out_seq_tokens=in_seq.split(' '),out_seq.split(' ')
       #针对描述语句过长的进行截断，下面减一操作是因为要对句末添加EOS
        if len(in_seq_tokens)>query_max_length-1:
            in_seq_tokens=in_seq_tokens[:query_max_length-1]
        #api序列名数据比较规整可以使用最大长度为api_max_length，对于短于最大值的api序列进行padding
#       if len(out_seq_tokens)>api_max_length-1:
#          out_seq_tokens=out_seq_tokens[:api_max_length]
        process_one_seq(in_seq_tokens,in_tokens,in_seqs,query_max_length)
        process_one_seq(out_seq_tokens,out_tokens,out_seqs,api_max_length)
    in_vocab,in_data=build_data(in_tokens,in_seqs)
    out_vocab,out_data=build_data(out_tokens,out_seqs)

    return in_vocab,out_vocab,Data.TensorDataset(in_data,out_data)


# 语义相似度GRU模型
class GRU(nn.Module):
    #其中input_size是指词表的大小
    def __init__(self,input_size,hidden_size,out_size):
        super(GRU,self).__init__()
        
        self.hidden_size=hidden_size
        self.embedding=nn.Embedding(input_size,hidden_size)
        self.gru=nn.GRU(hidden_size,hidden_size)
        self.out = nn.Linear(hidden_size, out_size)
    def forward(self,input,hidden):
        # 输入形状是(批量大小, 时间步数)。将输出互换样本维和时间步维
        #理解此处的0与1维度互换，因为是GRU序列，按照文本序列一个词一个词的输入进循环网络
        embedded=self.embedding(input.long()).permute(1,0,2)# (seq_len, batch, input_size)
        output=embedded
        output,state=self.gru(output,hidden)
        #移除时间步维，输出形状为(批量大小, 输出词典大小)
        #疑问：state是否是最后linear层的输入,输出的尺寸应该为（GRU层数，batch_size，num_hidden）
        return self.out(state).squeeze(dim=0)


   # def initRNN(self):
   #      return None
# 例子：创建一个批量大小为4、时间步数为7的小批量序列输入。设门控循环单元的隐藏层个数为2，隐藏单元个数为16。
# encoder = Encoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
# output, state = encoder(torch.zeros((4, 7)), encoder.begin_state())
# output.shape, state.shape # GRU的state是h, 而LSTM的是一个元组(h, c)

# 其中输出形状output为(时间步数, 批量大小, 隐藏单元个数)  torch.Size([7, 4, 16])
# 返回的状态为state为(隐藏层个数, 批量大小, 隐藏单元个数) torch.Size([2, 4, 16])

#小批量计算损失
def batch_loss(rnn_q,rnn_api,query,api,tag,loss, tmperature = 0.3):
    batch_size=query.shape[0]
#   初始化GRU的隐藏层状态
    q_state=None
    api_state=None
    q_output=rnn_q(query,q_state)
    api_output=rnn_api(api,api_state)
    #此时输出的维度猜测应该是（batch_size,hidden_size）
    #下方为AI研习社语义任务之后修改的向量维度
    #
    # print(q_output.shape)
    # print(api_output.shape)
    q_output=q_output/torch.norm(q_output,dim=1).unsqueeze(1)
    api_output=api_output/torch.norm(api_output,dim=1).unsqueeze(1)
    
    pre_score=torch.mm(q_output,api_output.permute(1,0))
    #此处进行sigmoid方程与结果的概率进行（0-1）匹配。可能存在一部分问题，待确定
    score=torch.sigmoid(torch.flatten(pre_score))
  #！！！！！这个l的初始化可能有问题！！！！
    #tag应该是一个1维向量（由对角矩阵拉直）
    l=loss(score / tmperature,tag/tmperature).mean()
    return l

#训练过程
def train(rnn_q,rnn_api,dataset,lr,batch_size,num_epochs):
    query_optimizer=torch.optim.Adam(rnn_q.parameters(),lr=lr)
    api_optimizier=torch.optim.Adam(rnn_api.parameters(),lr=lr)
    #将损失函数由交叉熵损失改为MSE损失
    loss=nn.MSELoss(reduction='none')

    for epoch in range(num_epochs):
        data_iter=Data.DataLoader(dataset,batch_size,shuffle=True)
        ##print("epoch为：{}, iter: {}".format(epoch, len(data_iter)))
        l_sum=0.0
        for iter, (query,api) in enumerate(data_iter):
            query_optimizer.zero_grad()
            api_optimizier.zero_grad()
            res_tag=torch.flatten(torch.eye(query.shape[0],query.shape[0]))
            l=batch_loss(rnn_q,rnn_api,query,api,res_tag,loss)
            l.backward()
            query_optimizer.step()
            api_optimizier.step()
            l_sum+=l.item()
            if (iter + 1) % 10 == 0:
                print("iter[{}/{}], loss {:.4f}".format((iter+1)*batch_size, len(dataset), l_sum / (iter+1)))
        print("epoch[{}/{}], loss {:.4f}".format(epoch, num_epochs, l_sum/len(data_iter)))

#设置相关超参数
hidden_size,out_size = 128, 64
lr, batch_size, num_epochs =0.01, 8, 20

#注意此时 res_tag的形状即为（batch_size,batch_size）


#假设此时的batch_size为2，应该是一个对角矩阵（对角线均为1.0）
query_max_length=20
#需要提前计算api数据集中api序列的最大长度
api_max_length=10
path="./test.txt"
in_vocab,out_vocab,dataset=read_data(path,query_max_length,api_max_length)
print(in_vocab)
rnn_q = GRU(len(in_vocab),hidden_size,out_size)
rnn_api = GRU(len(in_vocab),hidden_size,out_size)
train(rnn_q, rnn_api, dataset, lr, batch_size, num_epochs)


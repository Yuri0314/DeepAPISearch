import torch
from torch import nn
import torch.utils.data as Data
import numpy as np
import json
from traintry import *
import modeltry


def collection(rnn_api,data_iter,batch_size, hidden_size,dataset,device):
    # api_output=torch.tensor([0.0]).expand(batch_size,out_size).to(device)
    #使用apis收集对应的api_idx序列
    api_output = []
    apis=[]
    for iter, (query,api,method) in enumerate(data_iter):
        query=query.to(device)
        api=api.to(device)
        method=method.to(device)
    #初始化rnn_api的初始隐层状态
#     !!!!注意 apis中添加的应该是一个原生的字符串序列
        apis.append(api.cpu().detach().numpy())

        # 初始化网络的隐藏层
        w = torch.empty(2,query.shape[0],hidden_size).to(device)
        init_state=torch.nn.init.orthogonal_(w, gain=1)
        api_state=init_state
        output=rnn_api(api,method,api_state)
        api_output.append(output.cpu().detach().numpy())
        if (iter + 1) % 1000 == 0:
            print("iter[{}/{}]".format((iter+1)*batch_size, len(dataset)))
        # api_output=torch.cat((api_output,output))
    #对生成的向量进行裁剪，去除最开始初始化为零的api_output部分
    #   注 意！！！
#   需要把输出的api_output初始expand函数产生的全为零的部分清除,之后才可以保存到csv文件中(已裁剪)
    api_output=np.concatenate(api_output, axis=0)
    apis=np.concatenate(apis,axis=0)
    # print(api_output.shape)
    # print(np.sum(api_output * api_output, axis=1).shape)
    api_output = api_output / np.sum(api_output * api_output, axis=1).reshape([-1, 1])
    # api_output=api_output/torch.norm(api_output,dim=1).unsqueeze(1)
    return api_output,apis


#根据build_data函数更改，转化字符串为index序列
def build_query(vocab,query_seq):
    query_idx=[vocab.stoi[w] for w in query_seq]
    return torch.tensor(query_idx)

#计算输入查询的index表示并将其输入进模型中生成特征向量，与api序列的特征向量进行查询相似度，输出相似度最高的api接口
def caculate(vocab,rnn_q,api_output,apis,query,k,query_max_length,api_dict,hidden_size,device):
    query_seq=query.split(' ')
    if len(query_seq)>query_max_length-1:
            query_seq=query_seq[:query_max_length-1]
    query_data=build_query(vocab,query_seq).unsqueeze(0)
    query_data=query_data.to(device)
    # 初始化网络的隐藏层
    w = torch.empty(2,1,hidden_size).to(device)
    init_state=torch.nn.init.orthogonal_(w, gain=1)
    query_state=init_state
    q_output=rnn_q(query_data,query_state)
    #q_output形状应为（1，output_size）
    q_output=q_output/torch.norm(q_output.unsqueeze(0),dim=1).unsqueeze(1)

    #此处传进来的api_output形状应为(所有api个数，output_size)
    api_output=torch.from_numpy(api_output)
    api_output=api_output.to(device)
    pre_score=torch.mm(q_output,api_output.permute(1,0))
    #此处进行sigmoid方程与结果的概率进行（0-1）匹配。
    score=torch.sigmoid(pre_score)
    #之后针对分数最高的那个API进行输出序列
    _,index=torch.topk(score,k,dim=1)
    index=index.cpu().numpy().tolist()
    index=index[0]
    # _,idx=torch.max(score,1)
#     求出最大的那个api index
    # idx=int(idx)
    #根据idx查询出apis中的index，之后通过字典查询出对应的原生字符串
    # print(tuple(apis[idx].tolist()))
    for idx in index:
        print(api_dict[tuple(apis[idx].tolist())])

# 存储与取出所有的api向量
def jsonwrite(file_name, obj):
    with open(file_name, "w") as f:
        json.dump(f, obj)

def jsonread(file_name):
    with open(file_name, "r") as f:
        obj = json.load(f)
    o = np.array(obj)
    return obj

#处理query语句使其大转小写
if __name__=='__main__':
    api_dict=np.load('./api_dict.npy',encoding='bytes',allow_pickle=True)
    api_dict=api_dict.item()
    api_output=np.load('./api_output.npy',encoding='bytes',allow_pickle=True)
    apis=np.load('./apis.npy',encoding='bytes',allow_pickle=True)

    method_max_length=10
    #假设此时的batch_size为2，应该是一个对角矩阵（对角线均为1.0）
    query_max_length=20
    #需要提前计算api数据集中api序列的最大长度
    api_max_length=8
    path="./all.txt"
    api_dict1={}
    in_vocab,dataset=read_data(path,query_max_length,api_max_length,method_max_length,api_dict1)
    model=torch.load('100.pth')
    rnn_q=model.rnn_query
    hidden_size = 256
    k=20
    query=""
    print("输入exit()退出")
    while True:
        str = input("请输入：")
        if str == 'exit()':
            break
        else:
            query = str
    # data_iter=Data.DataLoader(dataset,batch_size,shuffle=True)
    # rnn_q=rnn_q.cpu()

    # api_output,apis=collection(rnn_api,data_iter,device)
    # api_output=api_output.cpu()

            caculate(in_vocab,rnn_q,api_output,apis,query.lower(),k,query_max_length,api_dict,hidden_size,device)

    # 对collection函数进行调用，得到api_output(特征向量)，api（api的字符串序列）
    print('收集结束。。。。。')
